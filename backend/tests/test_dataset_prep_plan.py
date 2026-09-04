"""Planning: the heuristic base case, and what the LLM is allowed to change.

The LLM tests use a scripted fake client rather than HTTP. `OpenRouterClient`
takes an injected transport for the same reason, but at this level the whole
client is the seam — these tests are about what `_merge` accepts and rejects,
not about how a request is sent.
"""

import pytest

from app.services.datasets.prep.detect import Detection
from app.services.datasets.prep.plan import (
    STRUCTURE_LOCKED_CONFIDENCE,
    analyze,
    build_prompt,
    heuristic_plan,
)
from app.services.providers import (
    LlmAuthError,
    LlmError,
    LlmRateLimitError,
    LlmUsage,
)


class FakeClient:
    """Returns a scripted body, or raises a scripted error."""

    def __init__(self, body=None, error=None, usage=None):
        self.body = body or {}
        self.error = error
        self.usage = usage or LlmUsage(prompt_tokens=120, completion_tokens=40, cost_usd=0.0004)
        self.calls: list[dict] = []

    def chat_json(self, messages, *, model, temperature=0.0, **kwargs):
        self.calls.append({"messages": messages, "model": model})
        if self.error:
            raise self.error
        return self.body, self.usage


def image_detection(**overrides) -> Detection:
    base = dict(
        modality="image",
        task_type="classification",
        format="image_folder",
        confidence=0.85,
        candidate_labels=["kista", "normal"],
        signals=["120 images across 2 folders: kista, normal."],
    )
    base.update(overrides)
    return Detection(**base)


def csv_detection(**overrides) -> Detection:
    base = dict(
        modality="text",
        task_type="text_classification",
        format="text_folder",
        confidence=0.80,
        candidate_labels=["negative", "positive"],
        columns=["text", "label", "id"],
        sample_rows=[
            {"text": f"review body number {i}", "label": "positive" if i % 2 else "negative", "id": str(i)}
            for i in range(8)
        ],
        column_roles={"text": "text", "label": "label"},
        signals=["`c.csv` carries `text` and `label` columns."],
    )
    base.update(overrides)
    return Detection(**base)


def decision(plan, field):
    return next((d for d in plan.decisions if d.field == field), None)


# --- heuristic base case ------------------------------------------------------


def test_heuristic_plan_carries_the_detection_forward():
    plan = heuristic_plan(image_detection())
    assert plan.source == "heuristic"
    assert plan.task_type == "classification"
    assert plan.format == "image_folder"
    assert plan.labels == ["kista", "normal"]
    assert plan.engine.mode == "heuristic"


def test_heuristic_split_is_seeded_and_matches_the_frontend_default():
    plan = heuristic_plan(image_detection())
    assert (plan.split.train, plan.split.valid, plan.split.test) == (0.7, 0.2, 0.1)
    assert plan.split.seed == 42


def test_split_stratifies_when_there_are_labels():
    assert heuristic_plan(image_detection()).split.stratify is True


def test_split_does_not_stratify_llm_records_which_have_no_label():
    plan = heuristic_plan(
        Detection(modality="record", task_type="llm_finetune", format="instruction_jsonl", confidence=0.9)
    )
    assert plan.split.stratify is False


def test_text_gets_cleaning_and_images_do_not():
    """Resizing here would bake one model's input size into the dataset."""
    assert heuristic_plan(csv_detection()).preprocess.enabled is True
    assert heuristic_plan(csv_detection()).preprocess.preset == "nlp_clean"
    assert heuristic_plan(image_detection()).preprocess.enabled is False


def test_augmentation_is_never_proposed():
    """`materialize` copies every image on disk; nobody asked for that."""
    for detection in (image_detection(), csv_detection()):
        plan = heuristic_plan(detection)
        assert plan.preprocess.augmentation_mode != "materialize"


def test_every_decision_carries_a_rationale():
    plan = heuristic_plan(csv_detection())
    assert plan.decisions
    for entry in plan.decisions:
        assert entry.rationale, f"{entry.field} has no rationale"


def test_detected_decisions_carry_evidence_from_the_scan():
    """Evidence is what makes a decision checkable rather than merely asserted."""
    task = decision(heuristic_plan(image_detection()), "task_type")
    assert task is not None
    assert task.source == "detected"
    assert task.evidence and "2 folders" in task.evidence


def test_language_modeling_is_never_proposed():
    """No project can declare it, so a plan naming it could never be applied."""
    plan = heuristic_plan(
        Detection(modality="text", task_type="language_modeling", format="text_folder", confidence=0.6)
    )
    assert plan.task_type is None


def test_a_task_the_project_disallows_is_dropped_with_a_pointer():
    plan = heuristic_plan(image_detection(), allowed_task_types=["text_classification"])
    assert plan.task_type is None
    assert plan.needs_input is not None
    assert "project settings" in plan.needs_input
    assert plan.warnings


def test_detection_needs_input_propagates_to_the_plan():
    plan = heuristic_plan(
        image_detection(
            candidate_labels=[],
            confidence=0.4,
            needs_input="These images carry no labels.",
        )
    )
    assert plan.needs_input == "These images carry no labels."


def test_mapping_is_derived_from_detected_columns():
    plan = heuristic_plan(csv_detection())
    assert plan.field_mapping.text == "text"
    assert plan.field_mapping.label == "label"


# --- no model configured ------------------------------------------------------


def test_without_a_key_the_plan_is_heuristic_and_says_so():
    plan = analyze(image_detection(), api_key=None, model=None)
    assert plan.source == "heuristic"
    assert plan.engine.mode == "heuristic"
    assert plan.engine.notice and "no openrouter key" in plan.engine.notice.lower()
    # Crucially, a usable plan still came out.
    assert plan.task_type == "classification"
    assert plan.labels == ["kista", "normal"]


# --- model failures all degrade, never raise ----------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LlmAuthError("bad key"), "rejected"),
        (LlmRateLimitError("slow down"), "rate limiting"),
        (LlmError("connection reset"), "failed"),
    ],
)
def test_every_model_failure_falls_back_with_a_stated_reason(error, expected):
    plan = analyze(csv_detection(), client=FakeClient(error=error), model="m")
    assert plan.source == "heuristic"
    assert plan.engine.notice and expected in plan.engine.notice.lower()
    assert plan.task_type == "text_classification"


# --- what the model is allowed to change --------------------------------------


def test_the_model_may_refine_labels_that_exist_in_the_data():
    detection = csv_detection(candidate_labels=["negative", "neutral", "positive"])
    client = FakeClient({"labels": ["positive", "negative"], "rationale": "Neutral is unused."})
    plan = analyze(detection, client=client, model="m")
    assert plan.source == "llm"
    assert plan.labels == ["positive", "negative"]
    assert decision(plan, "labels").source == "llm"


def test_invented_labels_are_dropped_and_reported():
    """The most damaging failure mode: labels matching nothing in the files."""
    detection = csv_detection(candidate_labels=["negative", "neutral", "positive"])
    client = FakeClient({"labels": ["positive", "neutral", "sarcastic", "wistful"]})
    plan = analyze(detection, client=client, model="m")
    assert plan.labels == ["positive", "neutral"]
    assert any("do not appear in the data" in warning for warning in plan.warnings)


def test_narrowing_a_classifier_to_one_class_is_refused():
    """`labels` assigns class ids, so dropping a class present in the data does
    not drop those rows — it relabels every one of them as the class left."""
    client = FakeClient({"labels": ["positive"], "rationale": "Only one class matters."})
    plan = analyze(csv_detection(), client=client, model="m")
    assert plan.labels == ["negative", "positive"]
    assert any("at least two classes" in warning for warning in plan.warnings)


def test_a_mapping_column_that_does_not_exist_is_dropped():
    client = FakeClient({"field_mapping": {"text": "text", "label": "nonexistent_column"}})
    plan = analyze(csv_detection(), client=client, model="m")
    assert plan.field_mapping.text == "text"
    assert plan.field_mapping.label != "nonexistent_column"


def test_dropping_a_bad_column_falls_back_to_the_scans_reading_of_that_role():
    """Text classification with no label column annotates nothing, so the role
    the scan read is kept rather than left empty."""
    client = FakeClient({"field_mapping": {"text": "text", "label": "nonexistent_column"}})
    plan = analyze(csv_detection(), client=client, model="m")
    assert plan.field_mapping.label == "label"


def test_an_unknown_mapping_key_is_ignored():
    client = FakeClient({"field_mapping": {"text": "text", "sentiment_axis": "label"}})
    plan = analyze(csv_detection(), client=client, model="m")
    assert plan.field_mapping.text == "text"


def test_the_model_may_change_the_task_when_detection_was_unsure():
    detection = csv_detection(confidence=0.5)
    client = FakeClient({
        "task_type": "summarization",
        "field_mapping": {"text": "text", "summary": "label"},
        "rationale": "The target is prose.",
    })
    plan = analyze(detection, client=client, model="m")
    assert plan.task_type == "summarization"
    assert decision(plan, "task_type").source == "llm"


def test_changing_the_task_without_columns_to_feed_it_is_refused():
    """A summarization plan over columns with no summary writes text items and
    no annotations: it applies cleanly and trains on nothing."""
    detection = csv_detection(confidence=0.5)
    client = FakeClient({"task_type": "summarization", "rationale": "The target is prose."})
    plan = analyze(detection, client=client, model="m")
    assert plan.task_type == "text_classification"
    assert any("was not used" in warning for warning in plan.warnings)


def feature_table_detection(**overrides) -> Detection:
    """A table the rules could not map — the state `_classify_rows` falls through
    to. Reconstructed from the student-exam upload that produced this test."""
    columns = [
        "student_id", "age", "gender", "education_level", "school_type",
        "family_income", "study_hours_per_day", "study_method", "motivation_level",
        "exam_anxiety_level", "exam_score", "performance_grade", "pass_status",
        "performance_level",
    ]
    base = dict(
        modality="table",
        task_type=None,
        format=None,
        confidence=0.30,
        columns=columns,
        sample_rows=[{column: "1" for column in columns}],
        signals=[f"`student_exam.csv` has {len(columns)} columns with no recognizable mapping."],
        needs_input="Orinth could not map these columns to a task it can train.",
    )
    base.update(overrides)
    return Detection(**base)


def test_an_image_task_is_refused_for_a_table_of_rows():
    """The reported failure: the model answered `classification` for a CSV, the
    plan applied, and the image importer found no images — an empty dataset that
    reported success."""
    client = FakeClient({
        "task_type": "classification",
        "confidence": 0.7,
        "labels": ["pass_status", "performance_level"],
        "field_mapping": {"text": "student_id", "label": "pass_status"},
        "rationale": "These features can be used for classification.",
    })
    plan = analyze(feature_table_detection(), client=client, model="m")
    assert plan.task_type is None
    assert any("table data" in warning for warning in plan.warnings)


def test_a_refused_plan_keeps_the_needs_input_that_triggers_a_transform():
    """`plan_for` reads a task-less plan as the signal to generate a transform,
    so clearing `needs_input` on the way past would strand the dataset."""
    client = FakeClient({"task_type": "classification", "confidence": 0.9})
    plan = analyze(feature_table_detection(), client=client, model="m")
    assert plan.needs_input


def test_a_mapping_is_pruned_to_the_roles_its_task_reads():
    """Asked for nine roles, a model fills nine roles. Apply reads two."""
    client = FakeClient({
        "field_mapping": {
            "text": "text", "label": "label", "summary": "id", "question": "id",
            "answer": "id", "instruction": "id", "input": "id", "output": "id",
            "messages": "id",
        },
    })
    plan = analyze(csv_detection(), client=client, model="m")
    used = {key for key, value in plan.field_mapping.model_dump().items() if value}
    assert used == {"text", "label"}


def test_a_table_with_no_task_carries_no_mapping():
    """Nine roles on a plan nothing will apply read as nine columns understood."""
    client = FakeClient({
        "task_type": "classification",
        "field_mapping": {"text": "student_id", "label": "pass_status"},
    })
    plan = analyze(feature_table_detection(), client=client, model="m")
    assert not any(plan.field_mapping.model_dump().values())


def test_the_prompt_offers_only_tasks_the_modality_can_carry():
    prompt = build_prompt(feature_table_detection(), None)
    tasks = prompt.split("Tasks available for this data:")[1].splitlines()[0]
    assert "text_classification" in tasks
    assert "object_detection" not in tasks
    assert "segmentation" not in tasks


def test_the_model_may_not_overrule_a_confident_structural_read():
    """A YOLO root is a YOLO root, whatever a language model believes."""
    detection = image_detection(
        task_type="object_detection", format="yolo", confidence=STRUCTURE_LOCKED_CONFIDENCE + 0.05
    )
    client = FakeClient({"task_type": "text_classification", "confidence": 0.99})
    plan = analyze(detection, client=client, model="m")
    assert plan.task_type == "object_detection"
    assert plan.format == "yolo"


def test_a_task_outside_the_projects_types_is_rejected_even_from_the_model():
    detection = csv_detection(confidence=0.5)
    client = FakeClient({"task_type": "segmentation"})
    plan = analyze(detection, client=client, model="m", allowed_task_types=["text_classification", "summarization"])
    assert plan.task_type != "segmentation"


def test_changing_the_task_recomputes_the_split_premise():
    detection = Detection(
        modality="text", task_type="text_classification", format="text_folder",
        confidence=0.5, candidate_labels=[], columns=["instruction", "output"],
    )
    client = FakeClient({"task_type": "llm_finetune"})
    plan = analyze(detection, client=client, model="m")
    assert plan.task_type == "llm_finetune"
    # No labels and an LLM task: stratifying would silently do nothing.
    assert plan.split.stratify is False


# --- accounting and prompt ----------------------------------------------------


def test_token_and_cost_usage_is_recorded():
    client = FakeClient({"labels": ["positive"]})
    plan = analyze(csv_detection(), client=client, model="openai/gpt-5.4-mini")
    assert plan.engine.mode == "llm"
    assert plan.engine.model == "openai/gpt-5.4-mini"
    assert plan.engine.prompt_tokens == 120
    assert plan.engine.estimated_cost_usd == 0.0004


def test_the_prompt_carries_the_scan_and_never_invented_context():
    detection = csv_detection()
    prompt = build_prompt(detection, ["text_classification"])
    assert "text_classification" in prompt
    assert "`c.csv` carries `text` and `label` columns." in prompt
    assert '"text", "label", "id"' in prompt or "text" in prompt


def test_a_locked_prompt_tells_the_model_not_to_contradict_the_scan():
    prompt = build_prompt(image_detection(confidence=0.95), None)
    assert "Do not contradict" in prompt


def test_an_empty_model_reply_leaves_the_heuristic_plan_intact():
    plan = analyze(csv_detection(), client=FakeClient({}), model="m")
    assert plan.task_type == "text_classification"
    assert plan.labels == ["negative", "positive"]


def test_pseudo_labels_do_not_trigger_stratification():
    """`["answer"]` and `["summary"]` mark an annotation kind, not a class.

    Stratifying on a single constant puts every item in one stratum, which is a
    plain shuffle wearing a misleading name.
    """
    for task, label in (("question_answering", "answer"), ("summarization", "summary")):
        plan = heuristic_plan(
            Detection(
                modality="text",
                task_type=task,
                format="text_folder",
                confidence=0.85,
                candidate_labels=[label],
            )
        )
        assert plan.split.stratify is False, task


def test_real_classes_still_trigger_stratification():
    for task in ("classification", "text_classification", "object_detection", "segmentation"):
        plan = heuristic_plan(
            Detection(
                modality="image",
                task_type=task,
                format="image_folder",
                confidence=0.85,
                candidate_labels=["a", "b"],
            )
        )
        assert plan.split.stratify is True, task
