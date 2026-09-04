"""Turning a detection into a plan the agent can apply.

Shaped after `assist.py` on the tabular branch, and for the same reasons:

**The heuristic path is the base case, not the fallback path.** It is computed
first, every time, before any model is consulted. That means data preparation
works with no API key, no network, and no account — which the desktop build
depends on — and it means a model failure degrades to a worse plan rather than
to no plan.

**The model ranks and names; it never decides structure.** Detection already read
the files. When it is confident, the model may not override the format it found —
a YOLO root is a YOLO root regardless of what a language model believes about it.
What a model genuinely adds is judgement the file scan cannot supply: which of
several plausible tasks suits the data, whether a column called `target` holds
classes or prose, and a readable rationale.

**Nothing the model names is trusted without checking it exists.** Every label
must appear in the detection's candidates, every column in its column list, every
transform in the allowed set. A model that invents a class taxonomy is the single
most damaging failure mode here — it would produce a dataset whose labels match
nothing in the files — so labels the data does not support are dropped rather
than corrected.
"""

import json
from typing import Any

from app.schemas import (
    DatasetFieldMapping,
    DatasetPrepPlan,
    DatasetPreprocessConfig,
    DatasetSplitConfig,
    PrepDecision,
    PrepEngine,
)
from app.services.datasets.constants import (
    ALLOWED_PREPROCESS_TRANSFORMS,
    IMAGE_TASK_TYPES,
    LLM_TASK_TYPES,
    NLP_TASK_TYPES,
    PREPROCESS_PRESETS,
)
from app.services.datasets.prep.detect import Detection
from app.services.datasets.prep.readiness import MULTICLASS_TASKS

#: Tasks the planner may propose. `language_modeling` is excluded deliberately:
#: no project can declare it (it exists for architecture-studio graphs), so
#: proposing it would produce a plan that apply always rejects.
ALLOWED_PLAN_TASKS = IMAGE_TASK_TYPES | NLP_TASK_TYPES | LLM_TASK_TYPES

#: Detection at or above this confidence has read an explicit declaration — a
#: manifest, a `data.yaml`, an annotation sidecar. A model may not overrule it.
STRUCTURE_LOCKED_CONFIDENCE = 0.85

#: Tasks whose labels are real classes, so balancing them across splits means
#: something. Summarization and question answering carry a *pseudo*-label
#: (`["summary"]`, `["answer"]`) that marks the annotation kind rather than a
#: class — stratifying on a single constant puts every item in one stratum and
#: degenerates to a plain shuffle, so it is worse than not asking.
STRATIFIABLE_TASKS = {
    "classification",
    "text_classification",
    "object_detection",
    "segmentation",
}

#: Which tasks each modality can carry. Detection reads the *files*, and that
#: reading holds at every confidence: a table of rows is not images, however
#: unsure the scan is about which task those rows support. Without this gate a
#: model answering "classification" for a CSV produces a plan whose importer
#: looks for image files, finds none, and applies cleanly to an empty dataset —
#: the one failure this agent exists to prevent.
MODALITY_TASKS: dict[str, set[str]] = {
    "image": IMAGE_TASK_TYPES,
    "text": NLP_TASK_TYPES | LLM_TASK_TYPES,
    "record": NLP_TASK_TYPES | LLM_TASK_TYPES,
    "table": NLP_TASK_TYPES | LLM_TASK_TYPES,
}

#: The mapping roles each task reads. Apply consults these and ignores the rest,
#: so a mapping with all nine filled is not a richer mapping — it is an unread
#: one, and on the Overview screen it reads as nine columns Orinth understood.
TASK_MAPPING_ROLES: dict[str, tuple[str, ...]] = {
    "text_classification": ("text", "label"),
    "summarization": ("text", "summary"),
    "question_answering": ("text", "question", "answer"),
    "llm_finetune": ("instruction", "input", "output", "messages"),
}

#: The subset of those roles without which no example can be built at all. It is
#: narrower than `TASK_MAPPING_ROLES` because some roles are optional: question
#: answering takes an optional `text` context, and `llm_finetune` accepts either
#: record shape, so it is checked separately.
TASK_REQUIRED_ROLES: dict[str, tuple[str, ...]] = {
    "text_classification": ("text", "label"),
    "summarization": ("text", "summary"),
    "question_answering": ("question", "answer"),
}

SYSTEM_PROMPT = (
    "You are a data preparation assistant for a machine learning platform. "
    "You are given the result of a deterministic scan of a dataset a user just "
    "uploaded, and you propose how to prepare it for training. "
    "Respond with ONLY a JSON object — no prose, no markdown fences. "
    "Base every claim on the scan; never invent a column, a label, or a file."
)

_RESPONSE_SHAPE = (
    'Return {"task_type": one of '
    '"classification"|"object_detection"|"segmentation"|"text_classification"|'
    '"summarization"|"question_answering"|"llm_finetune"|null, '
    '"confidence": 0.0-1.0, "labels": [string], '
    '"field_mapping": {"text"|"label"|"summary"|"question"|"answer"|'
    '"instruction"|"input"|"output"|"messages": string}, '
    '"rationale": string, "warnings": [string]}. '
    "Every label must already appear in candidate_labels. "
    "Every field_mapping value must already appear in columns. "
    "Fill only the field_mapping keys the task you chose actually needs — "
    "text+label for text_classification, text+summary for summarization, "
    "question+answer for question_answering, instruction+output or messages "
    "for llm_finetune — and leave the rest out. "
    "Use task_type null when none of the listed tasks fits; null is the right "
    "answer for a table of numeric features, and a better one than a task whose "
    "columns you had to guess at."
)


def heuristic_plan(
    detection: Detection,
    *,
    allowed_task_types: list[str] | None = None,
) -> DatasetPrepPlan:
    """The plan the file scan alone supports. Always computed; never raises."""
    decisions: list[PrepDecision] = []
    warnings: list[str] = []
    evidence = detection.signals[0] if detection.signals else None

    task = detection.task_type if detection.task_type in ALLOWED_PLAN_TASKS else None
    needs_input = detection.needs_input

    # A project only offers the tasks it declared. Proposing one it did not is a
    # plan that apply would reject with a 409, so it is caught here instead,
    # where the message can point at project settings.
    if task and allowed_task_types is not None and task not in allowed_task_types:
        warnings.append(
            f"This looks like a {task.replace('_', ' ')} dataset, but this project does not "
            "have that task type enabled."
        )
        needs_input = (
            f"Enable the {task.replace('_', ' ')} task in project settings, or pick a "
            "different task for this data."
        )
        task = None

    decisions.append(
        PrepDecision(
            field="task_type",
            value=task,
            source="detected",
            confidence=detection.confidence,
            rationale=(
                f"The files look like a {task.replace('_', ' ')} dataset."
                if task
                else "The files did not match any task Orinth can train."
            ),
            evidence=evidence,
        )
    )

    stored_format = detection.format
    if stored_format == "coco" and task in {"object_detection", "segmentation"}:
        # COCO is how the data arrived, not how it can train. `_image_paths`
        # looks under `<split>/` for COCO and `<split>/images/` for everything
        # else, and the YOLO runner reads `labels/*.txt`; storing as YOLO puts
        # the files where both already look. Apply converts the annotations.
        stored_format = "yolo"

    if stored_format:
        decisions.append(
            PrepDecision(
                field="format",
                value=stored_format,
                source="detected",
                confidence=detection.confidence,
                rationale=(
                    "COCO annotations are converted to the YOLO layout the training "
                    "runner reads."
                    if stored_format != detection.format
                    else f"Stored on disk as {stored_format.replace('_', ' ')}."
                ),
                evidence=evidence,
            )
        )

    labels = list(detection.candidate_labels)
    if labels:
        decisions.append(
            PrepDecision(
                field="labels",
                value=labels,
                source="detected",
                confidence=detection.confidence,
                rationale=f"Found {len(labels)} class(es) in the data.",
                evidence=next(
                    (signal for signal in detection.signals if "class" in signal or "folder" in signal),
                    evidence,
                ),
            )
        )

    preprocess, preprocess_decision = _preprocess_for(task, detection)
    decisions.append(preprocess_decision)

    split, split_decision = _split_for(task, labels)
    decisions.append(split_decision)

    mapping, mapping_decision = _mapping_for(detection, task)
    if mapping_decision is not None:
        decisions.append(mapping_decision)

    return DatasetPrepPlan(
        source="heuristic",
        confidence=detection.confidence,
        task_type=task,  # type: ignore[arg-type]
        format=stored_format,  # type: ignore[arg-type]
        labels=labels,
        field_mapping=mapping,
        preprocess=preprocess,
        split=split,
        rationale=" ".join(detection.signals) if detection.signals else "",
        warnings=warnings,
        decisions=decisions,
        engine=PrepEngine(mode="heuristic"),
        needs_input=needs_input,
    )


def _preprocess_for(
    task: str | None, detection: Detection
) -> tuple[DatasetPreprocessConfig, PrepDecision]:
    """Cleaning defaults.

    Text gets whitespace and case normalization, which is safe and usually
    wanted. Images and records get nothing: resizing is the training runner's job
    and it already knows the model's input size, so doing it here would bake one
    model's resolution into the dataset.

    Augmentation is never proposed. `materialize` copies every image on disk, and
    a user who dropped a folder in did not ask to multiply it.
    """
    if task in NLP_TASK_TYPES:
        config = DatasetPreprocessConfig(
            enabled=True,
            preset="nlp_clean",
            transforms=list(PREPROCESS_PRESETS["nlp_clean"]),
        )
        rationale = "Text is lowercased and whitespace-normalized before training."
    else:
        config = DatasetPreprocessConfig(enabled=False, preset="none")
        rationale = (
            "No preprocessing: the training runner resizes images to whatever the "
            "chosen model expects."
            if task in IMAGE_TASK_TYPES
            else "No preprocessing needed for this data."
        )

    return config, PrepDecision(
        field="preprocess",
        value=config.model_dump(mode="json"),
        source="heuristic",
        rationale=rationale,
    )


def _split_for(task: str | None, labels: list[str]) -> tuple[DatasetSplitConfig, PrepDecision]:
    """A 70/20/10 split, stratified when there is a label to stratify on.

    Matches `defaultSplitConfig()` in the frontend, so the agent's choice and the
    form's default are the same number rather than two conventions.
    """
    stratify = bool(labels) and task in STRATIFIABLE_TASKS
    config = DatasetSplitConfig(train=0.7, valid=0.2, test=0.1, seed=42, stratify=stratify)
    return config, PrepDecision(
        field="split",
        value=config.model_dump(mode="json"),
        source="heuristic",
        rationale=(
            "Split 70/20/10 into train, validation and test, keeping each class "
            "balanced across the three."
            if stratify
            else "Split 70/20/10 into train, validation and test."
        ),
    )


def _mapping_for(
    detection: Detection, task: str | None
) -> tuple[DatasetFieldMapping, PrepDecision | None]:
    """Which detected column feeds which part of an example.

    Detection decided this already, over the full profile rather than the capped
    prompt sample, so the roles are taken from it. Only the record-shape keys —
    which are fixed names by definition — are read off the column list here.
    """
    columns = detection.columns
    if not columns:
        return DatasetFieldMapping(), None

    roles = dict(detection.column_roles)
    mapping = DatasetFieldMapping(
        text=roles.get("text"),
        label=roles.get("label"),
        summary=roles.get("summary"),
        question=roles.get("question"),
        answer=roles.get("answer"),
        instruction="instruction" if "instruction" in columns else None,
        input="input" if "input" in columns else None,
        output="output" if "output" in columns else None,
        messages=next(
            (name for name in ("messages", "conversations", "chosen") if name in columns), None
        ),
    )

    mapping = _prune_mapping(mapping, task)
    used = {key: value for key, value in mapping.model_dump().items() if value}
    if not used:
        return mapping, None

    return mapping, PrepDecision(
        field="mapping",
        value=used,
        source="detected",
        rationale="Matched columns to the parts of a training example.",
        evidence=f"Columns present: {', '.join(columns[:8])}.",
    )


def _pretty(task: str) -> str:
    return task.replace("_", " ")


def _tasks_for_modality(detection: Detection) -> set[str]:
    """The tasks the scanned modality can carry.

    An unrecognized modality constrains nothing: with no reading of the files,
    there is no evidence to contradict the model with.
    """
    return MODALITY_TASKS.get(detection.modality, ALLOWED_PLAN_TASKS)


def _project_tasks(allowed_task_types: list[str] | None) -> set[str]:
    """The tasks this project declared, narrowed to the ones a plan may name."""
    if allowed_task_types is None:
        return set(ALLOWED_PLAN_TASKS)
    return set(allowed_task_types) & ALLOWED_PLAN_TASKS


def _prune_mapping(mapping: DatasetFieldMapping, task: str | None) -> DatasetFieldMapping:
    """Drop the roles the chosen task never reads.

    Image tasks read no columns, and a plan with no task reads none either —
    apply refuses it, and the transform stage writes its own mapping over the
    top. So both keep nothing: a mapping nothing will consume is not neutral on
    screen, it is nine columns claiming to have been understood.
    """
    roles = TASK_MAPPING_ROLES.get(task or "")
    if roles is None:
        return DatasetFieldMapping()
    return DatasetFieldMapping(
        **{role: getattr(mapping, role) for role in roles if getattr(mapping, role)}
    )


def _mapping_satisfies(task: str, mapping: DatasetFieldMapping) -> bool:
    """Whether apply could build one example out of a row with this mapping."""
    if task == "llm_finetune":
        # Two record shapes, either of which is enough: a turn list, or an
        # instruction paired with its output.
        return bool(mapping.messages or (mapping.instruction and mapping.output))
    required = TASK_REQUIRED_ROLES.get(task)
    if required is None:
        return True
    return all(getattr(mapping, role) for role in required)


def blocking_reason(plan: DatasetPrepPlan, detection: Detection) -> str | None:
    """Why applying this plan would yield nothing, or None if it would work.

    Every field the model is allowed to touch is one that can break the plan's
    internal agreement — a task whose importer reads columns, paired with a
    mapping naming none of them, imports zero rows and reports success. Apply
    cannot catch that: an empty upload and an unreadable one look identical by
    the time it runs. So the plan is checked here, as a whole, once the merge
    that could have broken it is done.
    """
    task = plan.task_type
    if not task or plan.needs_input:
        # Already stuck, and honest about it. `plan_for` reads this state as the
        # signal to generate a transform.
        return None

    if task not in _tasks_for_modality(detection):
        return f"Orinth cannot train {_pretty(task)} on {detection.modality} data"

    if not plan.format:
        return f"no storage format fits {_pretty(task)} here"

    if detection.columns and not _mapping_satisfies(task, plan.field_mapping):
        return (
            "no column in this file was matched to the parts of a "
            f"{_pretty(task)} example"
        )

    if task in MULTICLASS_TASKS and len(plan.labels) < 2:
        return (
            f"{_pretty(task)} needs at least two classes and the data yielded "
            f"{len(plan.labels)}"
        )

    return None


# --- LLM refinement -----------------------------------------------------------


def analyze(
    detection: Detection,
    *,
    allowed_task_types: list[str] | None = None,
    api_key: str | None = None,
    model: str | None = None,
    client: Any = None,
) -> DatasetPrepPlan:
    """The heuristic plan, refined by a model when one is configured.

    Never raises. Every failure path returns the heuristic plan with a notice
    saying what went wrong, because a data-prep step that dies when OpenRouter
    has an outage is worse than one that produces a slightly duller plan.
    """
    fallback = heuristic_plan(detection, allowed_task_types=allowed_task_types)

    if not (api_key and model) and client is None:
        fallback.engine.notice = (
            "No OpenRouter key is configured, so this plan comes from file inspection "
            "alone. Add a key in Settings for model-assisted analysis."
        )
        return fallback

    # Imported here, not at module scope: `app.services.recipes.__init__` imports
    # `DatasetService`, which reaches this module, so a top-level provider import
    # would close a cycle.
    from app.services.providers import (
        ChatMessage,
        LlmAuthError,
        LlmError,
        LlmRateLimitError,
        OpenRouterClient,
    )

    resolved = client or OpenRouterClient(api_key=api_key)
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=build_prompt(detection, allowed_task_types)),
    ]

    try:
        body, usage = resolved.chat_json(messages, model=model or "", temperature=0.0)
    except LlmAuthError:
        fallback.engine.notice = (
            "OpenRouter rejected the configured API key, so this plan comes from file "
            "inspection alone."
        )
        return fallback
    except LlmRateLimitError:
        fallback.engine.notice = (
            "OpenRouter is rate limiting, so this plan comes from file inspection alone."
        )
        return fallback
    except LlmError as error:
        fallback.engine.notice = (
            f"The OpenRouter call failed ({error}), so this plan comes from file "
            "inspection alone."
        )
        return fallback

    refined = _merge(fallback, body, detection, allowed_task_types)
    refined.engine = PrepEngine(
        mode="llm",
        model=model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        estimated_cost_usd=usage.cost_usd,
    )
    return refined


def build_prompt(detection: Detection, allowed_task_types: list[str] | None) -> str:
    """The scan, rendered for a model. Only what the scan actually found."""
    # Only the tasks that are both enabled for the project and possible for the
    # modality the scan read. Offering the rest invites a reply that has to be
    # thrown away, and the thrown-away reply is the one that reads best.
    allowed = sorted(_project_tasks(allowed_task_types) & _tasks_for_modality(detection))
    lines = [
        "Deterministic scan result:",
        f"- modality: {detection.modality}",
        f"- detected task: {detection.task_type or 'none'} (confidence {detection.confidence:.2f})",
        f"- detected format: {detection.format or 'none'}",
        f"- candidate_labels: {json.dumps(detection.candidate_labels)}",
        f"- columns: {json.dumps(detection.columns)}",
        f"- files: {json.dumps(detection.file_counts)}",
        "- signals:",
        *[f"    · {signal}" for signal in detection.signals],
        "",
        f"Tasks available for this data: {json.dumps(allowed)}",
        "Choose from that list or null. It is already narrowed to what this "
        "project enabled and what this kind of data can train.",
    ]
    if detection.sample_rows:
        lines += ["", "Sample rows (values truncated):", json.dumps(detection.sample_rows[:10], ensure_ascii=False)]
    if detection.confidence >= STRUCTURE_LOCKED_CONFIDENCE:
        lines += [
            "",
            "The scan read an explicit declaration (a manifest, a data.yaml, or annotation "
            "sidecars). Do not contradict the detected task or format; refine the labels, "
            "the mapping, and the rationale only.",
        ]
    lines += ["", _RESPONSE_SHAPE]
    return "\n".join(lines)


def _merge(
    fallback: DatasetPrepPlan,
    body: dict[str, Any],
    detection: Detection,
    allowed_task_types: list[str] | None,
) -> DatasetPrepPlan:
    """Fold a model reply into the heuristic plan, dropping anything unsupported."""
    plan = fallback.model_copy(deep=True)
    plan.source = "llm"

    allowed = _project_tasks(allowed_task_types)
    for_modality = _tasks_for_modality(detection)
    structure_locked = detection.confidence >= STRUCTURE_LOCKED_CONFIDENCE

    proposed = body.get("task_type")
    task = proposed if isinstance(proposed, str) else None
    if task and task in allowed and not structure_locked and task != plan.task_type:
        if task not in for_modality:
            # The scan read the files. Which task those files support is a
            # judgement the model may make; what kind of files they are is not.
            plan.warnings.append(
                f"The model proposed {_pretty(task)}, which was not used: Orinth "
                f"cannot train that on {detection.modality} data."
            )
            task = None
    else:
        task = None

    if task:
        plan.task_type = task  # type: ignore[assignment]
        plan.needs_input = None
        plan.decisions = [
            decision for decision in plan.decisions if decision.field != "task_type"
        ]
        plan.decisions.insert(
            0,
            PrepDecision(
                field="task_type",
                value=task,
                source="llm",
                confidence=_confidence(body.get("confidence")),
                rationale=str(body.get("rationale") or "")[:400]
                or "The model read the sample and chose this task.",
                evidence=detection.signals[0] if detection.signals else None,
            ),
        )
        # The task changed, so the split's stratification premise changed with it.
        plan.split, split_decision = _split_for(task, plan.labels)
        plan.decisions = [d for d in plan.decisions if d.field != "split"] + [split_decision]

        # And so did the mapping: the roles the heuristic filled were the roles
        # the *old* task reads. Re-deriving from the scan is what lets a switch
        # to `llm_finetune` pick up the `instruction`/`output` columns that a
        # text-classification mapping had no place for.
        plan.field_mapping, mapping_decision = _mapping_for(detection, task)
        plan.decisions = [d for d in plan.decisions if d.field != "mapping"]
        if mapping_decision is not None:
            plan.decisions.append(mapping_decision)

    # Labels: only ones the data actually contains. A model that invents a class
    # taxonomy would produce labels matching nothing in the files.
    suggested = [str(label) for label in body.get("labels") or [] if isinstance(label, str)]
    known = set(detection.candidate_labels)
    kept = [label for label in suggested if label in known]
    dropped = [label for label in suggested if label not in known]
    if dropped:
        plan.warnings.append(
            f"Ignored {len(dropped)} suggested label(s) that do not appear in the data: "
            f"{', '.join(dropped[:5])}."
        )
    if kept and kept != plan.labels:
        plan.labels = kept
        plan.decisions = [d for d in plan.decisions if d.field != "labels"] + [
            PrepDecision(
                field="labels",
                value=kept,
                source="llm",
                rationale="Selected from the classes found in the data.",
                evidence=f"Candidates in the data: {', '.join(sorted(known)[:8])}.",
            )
        ]

    # Mapping: only columns the scan actually saw.
    raw_mapping = body.get("field_mapping")
    if isinstance(raw_mapping, dict) and detection.columns:
        columns = set(detection.columns)
        named = {
            key: value
            for key, value in raw_mapping.items()
            if key in DatasetFieldMapping.model_fields
            and isinstance(value, str)
            and value in columns
        }
        # Overlaid on the scan's mapping rather than replacing it. Dropping one
        # bad column should not discard the roles the scan read correctly — a
        # model that names `text` and invents `label` would otherwise leave a
        # text-classification plan with nothing to annotate rows from.
        overlaid = plan.field_mapping.model_dump() | named
        # A model asked for nine roles will fill nine roles. Only the ones the
        # chosen task reads survive, so the mapping shown on Overview is the
        # mapping apply will use rather than a longer, more confident-looking one.
        pruned = _prune_mapping(DatasetFieldMapping(**overlaid), plan.task_type)
        cleaned = {key: value for key, value in pruned.model_dump().items() if value}
        if cleaned:
            plan.field_mapping = pruned
            plan.decisions = [d for d in plan.decisions if d.field != "mapping"] + [
                PrepDecision(
                    field="mapping",
                    value=cleaned,
                    source="llm",
                    rationale="Matched columns to the parts of a training example.",
                    evidence=f"Columns present: {', '.join(detection.columns[:8])}.",
                )
            ]

    if isinstance(body.get("rationale"), str):
        plan.rationale = body["rationale"][:1000]
    for warning in body.get("warnings") or []:
        if isinstance(warning, str) and warning.strip():
            plan.warnings.append(warning.strip()[:300])

    confidence = _confidence(body.get("confidence"))
    if confidence is not None and not structure_locked:
        plan.confidence = confidence

    _sanitize_preprocess(plan)

    blocker = blocking_reason(plan, detection)
    if blocker is None:
        return plan

    # The refined plan cannot produce a single example, so the heuristic one is
    # restored — including its `needs_input`, which the task branch above clears.
    # That is not a lost opportunity: a plan with no task is what `plan_for`
    # reads as the signal to generate a transform, and a feature table the rules
    # could not map is precisely what that stage was built for.
    reverted = fallback.model_copy(deep=True)
    reverted.source = "llm"
    # The diagnostics the merge accumulated describe the *reply* — invented
    # labels, columns that do not exist — so they survive the plan being
    # discarded. Only the plan's own fields are rolled back.
    reverted.warnings.extend(
        warning for warning in plan.warnings if warning not in fallback.warnings
    )
    reverted.warnings.append(
        f"The model's plan ({_pretty(str(plan.task_type))}) was not used: {blocker}."
    )
    _sanitize_preprocess(reverted)
    return reverted


def _sanitize_preprocess(plan: DatasetPrepPlan) -> None:
    """Keep only transforms the pipeline actually implements."""
    allowed = [
        transform
        for transform in plan.preprocess.transforms
        if transform in ALLOWED_PREPROCESS_TRANSFORMS
    ]
    if allowed != plan.preprocess.transforms:
        plan.preprocess.transforms = allowed


def _confidence(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(number, 0.0), 1.0)
