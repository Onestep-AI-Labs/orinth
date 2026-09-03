"""Whether a dataset can actually be trained on, and if not, what to do about it.

Nothing in the platform used to answer that question. ``metadata.processed_at``
is stamped on every split and read by no one; the training page filtered on task
type alone and then disabled its Start button for four different unstated
reasons. Worst of it: ``keras_common.load_split`` silently skips images whose
annotation file is missing or empty, so an unlabelled classification dataset
trains on zero items and reports success.

Every rule below is therefore derived from what a runner actually requires, and
each cites the code that enforces it. The point is to fail loudly here rather
than quietly there.

Two design constraints shape this module:

**It must be pure over the fields already on ``DatasetSummary``.** ``summary()``
runs for every dataset on every catalog list, and ``_split_summary`` already
walks items and reads annotation JSON per item. Any check needing to open a file
would double that cost, so those live in ``eda_summary`` and surface as
advisories on the detail screen only. The split is forced rather than stylistic:
``_split_summary`` sets ``annotation_count = len(items)`` unconditionally for
``llm_finetune``, so "every record has a non-empty output" is genuinely not
derivable here.

**It imports nothing that imports back.** ``datasets/service.py`` imports this
module, so it holds to ``app.schemas`` plus ``datasets.constants`` — the latter
declares no imports of its own and so cannot close a cycle.
"""

from dataclasses import dataclass

from app.schemas import (
    DatasetReadiness,
    DatasetReadinessCheck,
    DatasetSplitSummary,
    ReadinessAction,
    ReadinessState,
)
from app.services.datasets.constants import (
    IMAGE_TASK_TYPES,
    LLM_FORMATS,
    LLM_TASK_TYPES,
    TRAINING_SPLITS,
)

#: `llm_sft.py` raises below this; see `MIN_TRAIN_RECORDS` there. Duplicated
#: rather than imported because importing a training runner would pull torch and
#: transformers into the API process, which `docs/ai/rules.md` forbids.
MIN_LLM_TRAIN_RECORDS = 10

#: The one prep state during which the dataset is genuinely being rewritten
#: underneath us, so no verdict computed now would still be true by the time it
#: was read. Narrowed from the original {detecting, planning, applying}: those
#: first two only *read* the staging directory, and treating them as blocking is
#: what made a ready dataset report "Preparing…" — and, if the run then stopped
#: at `planned`, "Needs prep" — merely because someone asked Orinth to look at it
#: a second time. Analysis is now reported through `DatasetReadiness.busy`, which
#: annotates the verdict instead of replacing it.
_PREP_REWRITING = {"applying"}

#: Non-destructive stages. They set `busy` and change nothing else.
_PREP_ANALYZING = {"detecting", "planning"}

#: Tasks whose supervision lives in an annotation sidecar, so a non-zero item
#: count says nothing about whether the split is usable.
_ANNOTATION_BACKED = IMAGE_TASK_TYPES | {
    "text_classification",
    "summarization",
    "question_answering",
}

#: Tasks that need at least two classes to be worth training. Detection and
#: segmentation are single-class-viable; a classifier with one class is not.
#: `plan.py` refuses to propose these without two classes, so that a plan and
#: the readiness verdict on its result cannot disagree about what trains.
MULTICLASS_TASKS = {"classification", "text_classification"}


@dataclass(frozen=True)
class _Rule:
    """A check plus what it implies, so state and action stay beside the test."""

    check: DatasetReadinessCheck
    action: ReadinessAction = "none"
    state: ReadinessState = "needs_prep"


def _rule(
    *,
    id: str,
    label: str,
    passed: bool,
    detail: str = "",
    severity: str = "blocking",
    action: ReadinessAction = "none",
    state: ReadinessState = "needs_prep",
) -> _Rule:
    return _Rule(
        check=DatasetReadinessCheck(
            id=id,
            label=label,
            passed=passed,
            severity=severity,  # type: ignore[arg-type]
            detail="" if passed else detail,
        ),
        action=action,
        state=state,
    )


def _count(splits: dict[str, DatasetSplitSummary], split: str, field: str) -> int:
    summary = splits.get(split)
    return int(getattr(summary, field, 0)) if summary is not None else 0


def readiness_for(
    *,
    task_type: str,
    format: str,
    labels: list[str],
    splits: dict[str, DatasetSplitSummary],
    metadata: dict | None = None,
) -> DatasetReadiness:
    """Judge a dataset from its manifest and split counts alone.

    Returns the first failing blocking check's ``detail`` as ``summary`` — the
    one sentence the training page shows in place of a silently disabled button.
    """
    metadata = metadata or {}

    prep = metadata.get("prep") or {}
    prep_state = str(prep.get("state") or "")
    # When the agent stopped because it needed a person, say what it needed.
    # "Run Prep" is actively misleading there: prep already ran, and running it
    # again will stop in exactly the same place.
    plan = prep.get("plan") if isinstance(prep, dict) else None
    blocked_on = str((plan or {}).get("needs_input") or "") if isinstance(plan, dict) else ""

    total_items = sum(_count(splits, split, "item_count") for split in splits)

    # The data rules are built first, and their verdict is what decides how hard
    # the prep rules push. That ordering is the whole fix: whether a dataset can
    # be trained on is a property of the files on disk, and a prep run that ended
    # somewhere other than `ready` is not evidence about them.
    data_rules: list[_Rule] = [
        # Nothing to train on at all. Checked before labels, because "add data"
        # is more useful than "add labels" when the dataset is empty.
        _rule(
            id="has_items",
            label="Dataset has items",
            passed=total_items > 0,
            detail="This dataset is empty. Upload data to get started.",
            action="upload",
            state="needs_input",
        )
    ]
    if task_type in LLM_TASK_TYPES:
        data_rules.extend(_llm_rules(format=format, splits=splits))
    else:
        data_rules.extend(
            _supervised_rules(task_type=task_type, labels=labels, splits=splits)
        )

    # Advisory everywhere: items sitting in the inbox are simply not used, which
    # is invisible at training time and looks like missing data in the metrics.
    unassigned = _count(splits, "unassigned", "item_count")
    data_rules.append(
        _rule(
            id="inbox_drained",
            label="Inbox is empty",
            passed=unassigned == 0,
            detail=(
                f"{unassigned} item{'s' if unassigned != 1 else ''} are still unassigned "
                "and will not be used for training."
            ),
            severity="advisory",
        )
    )

    data_is_trainable = not any(
        rule.check.severity == "blocking" and not rule.check.passed for rule in data_rules
    )

    analyzing = prep_state in _PREP_ANALYZING
    rewriting = prep_state in _PREP_REWRITING
    # A run in flight overrides the verdict only when there is no standing
    # verdict worth preserving: `applying` is rewriting the splits under us, and
    # analysis of a dataset that has nothing in its splits yet has no prior
    # answer to keep. Re-analysing a dataset that already trains does — which is
    # why asking Orinth to look again no longer makes it report "Preparing…".
    rules: list[_Rule] = [
        _rule(
            id="prep_idle",
            label="No prep run in progress",
            passed=not (rewriting or (analyzing and total_items == 0)),
            detail="Orinth is still preparing this dataset.",
            action="wait",
            state="blocked",
        )
    ]

    # Raw files staged but never applied. Blocking only while the data cannot
    # stand on its own: once the splits are populated and annotated, a leftover
    # `planned` or `failed` state from a re-run says nothing about them. Making
    # this unconditional is what turned a ready dataset into "Needs prep" the
    # moment Orinth was asked to re-read it and stopped short of applying.
    stalled = prep_state in {"draft", "planned", "failed"}
    rules.append(
        _rule(
            id="prep_applied",
            label="Prep has been applied",
            passed=not stalled,
            detail=(
                "Orinth's last run stopped before applying anything. The dataset is "
                "unchanged and still trainable."
                if data_is_trainable
                else blocked_on
                or (
                    "Raw files are uploaded but not prepared yet. Run Prep to make this "
                    "dataset trainable."
                )
            ),
            # It still appears in the check list either way; it just no longer
            # overrides a verdict the files themselves already answered.
            severity="advisory" if data_is_trainable else "blocking",
            action="label" if blocked_on and not data_is_trainable else "run_prep",
            state="needs_input" if blocked_on else "needs_prep",
        )
    )

    rules.extend(data_rules)
    return _verdict(rules, busy=analyzing or rewriting)


def _supervised_rules(
    *, task_type: str, labels: list[str], splits: dict[str, DatasetSplitSummary]
) -> list[_Rule]:
    """Rules for every task whose supervision is labels plus annotations."""
    rules: list[_Rule] = []

    minimum = 2 if task_type in MULTICLASS_TASKS else 1
    if task_type in MULTICLASS_TASKS or task_type in IMAGE_TASK_TYPES:
        rules.append(
            _rule(
                id="labels_defined",
                label=f"At least {minimum} label{'s' if minimum > 1 else ''}",
                passed=len(labels) >= minimum,
                detail=(
                    f"This task needs at least {minimum} class "
                    f"label{'s' if minimum > 1 else ''}; "
                    f"{len(labels)} defined."
                ),
                action="label",
                state="needs_input",
            )
        )

    train_items = _count(splits, "train", "item_count")
    rules.append(
        _rule(
            id="train_non_empty",
            label="Train split has items",
            passed=train_items > 0,
            detail="The train split is empty. Run Prep to distribute items into splits.",
            action="run_prep",
        )
    )

    if task_type in _ANNOTATION_BACKED:
        # `keras_common.load_split` drops unannotated images silently, and the
        # NLP loaders skip rows with no annotation sidecar. A split with items
        # but no annotations trains on nothing and reports success.
        rules.append(
            _rule(
                id="train_annotated",
                label="Train items are annotated",
                passed=_count(splits, "train", "annotation_count") > 0,
                detail=(
                    "No items in the train split are annotated; training would silently "
                    "run on nothing. Label them in the Data tab."
                ),
                action="label",
                state="needs_input",
            )
        )

    # `DatasetService.training_root` raises FileNotFoundError without
    # `valid/images`, so for image tasks an empty valid split is fatal, not slow.
    is_image = task_type in IMAGE_TASK_TYPES
    rules.append(
        _rule(
            id="valid_non_empty",
            label="Valid split has items",
            passed=_count(splits, "valid", "item_count") > 0,
            detail=(
                "The valid split is empty; image training requires it. Run Prep to "
                "split the data."
                if is_image
                else "The valid split is empty, so the run will report no eval metrics."
            ),
            severity="blocking" if is_image else "advisory",
            action="run_prep",
        )
    )

    if is_image:
        rules.append(
            _rule(
                id="valid_annotated",
                label="Valid items are annotated",
                passed=_count(splits, "valid", "annotation_count") > 0,
                detail=(
                    "No items in the valid split are annotated, so validation would "
                    "score against nothing."
                ),
                action="label",
                state="needs_input",
            )
        )

    return rules


def _llm_rules(*, format: str, splits: dict[str, DatasetSplitSummary]) -> list[_Rule]:
    """Rules for `llm_finetune`, whose supervision is the record itself."""
    rules: list[_Rule] = [
        _rule(
            id="llm_format",
            label="Record format is trainable",
            passed=format in LLM_FORMATS,
            detail=(
                f"`{format}` records cannot be fine-tuned; expected an instruction or "
                "chat JSONL dataset."
            ),
            action="upload",
            state="needs_input",
        )
    ]

    train_items = _count(splits, "train", "item_count")
    # `llm_sft.resolve_train_valid_records` falls back to the inbox when train is
    # empty and carves its own holdout, so counting only `train` would block runs
    # the runner would happily complete. Mirror that tolerance.
    pool = train_items or _count(splits, "unassigned", "item_count")

    rules.append(
        _rule(
            id="llm_min_records",
            label=f"At least {MIN_LLM_TRAIN_RECORDS} train records",
            passed=pool >= MIN_LLM_TRAIN_RECORDS,
            detail=(
                f"LLM fine-tuning needs at least {MIN_LLM_TRAIN_RECORDS} train records; "
                f"this dataset has {pool}."
            ),
            action="upload",
            state="needs_input",
        )
    )

    rules.append(
        _rule(
            id="train_non_empty",
            label="Train split has records",
            passed=train_items > 0,
            detail=(
                "Records are still unassigned. Training would fall back to the inbox; "
                "run Prep to split them properly."
            ),
            severity="advisory",
            action="run_prep",
        )
    )

    return rules


def _verdict(rules: list[_Rule], *, busy: bool = False) -> DatasetReadiness:
    checks = [rule.check for rule in rules]
    for rule in rules:
        if rule.check.severity == "blocking" and not rule.check.passed:
            return DatasetReadiness(
                state=rule.state,
                trainable=False,
                summary=rule.check.detail,
                next_action=rule.action,
                checks=checks,
                busy=busy,
            )

    advisories = [
        rule for rule in rules if rule.check.severity == "advisory" and not rule.check.passed
    ]
    summary = "Ready to train." if not advisories else advisories[0].check.detail
    return DatasetReadiness(
        state="ready",
        trainable=True,
        summary=summary,
        next_action="none",
        checks=checks,
        busy=busy,
    )


def training_splits_present(splits: dict[str, DatasetSplitSummary]) -> bool:
    """True when every training split holds at least one item."""
    return all(_count(splits, split, "item_count") > 0 for split in TRAINING_SPLITS)
