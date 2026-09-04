"""Turning a feature table into something the platform can train on.

Both fixtures follow the column shapes of the Kaggle datasets that motivated
this stage — an early-stage diabetes risk table (categorical target named
`class`) and a student-performance export (a `result` column beside the
`exam_score` it was derived from). Neither maps to a task by any rule in
`detect.py`, which is the whole reason the stage exists.
"""

import csv
import io
import random
from pathlib import Path

import pytest

from app.services.datasets import DatasetService
from app.services.datasets.prep import transform as transform_module
from app.services.datasets.prep.detect import profile_columns
from app.services.datasets.prep.service import DatasetPrepService
from app.services.datasets.prep.staging import derived_root, staging_root
from app.services.providers.types import LlmUsage

SYMPTOMS = [
    "Polyuria", "Polydipsia", "sudden weight loss", "weakness", "Polyphagia",
    "Genital thrush", "visual blurring", "Itching", "Irritability",
    "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity",
]


def diabetes_csv(rows: int = 200) -> bytes:
    rng = random.Random(11)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["Age", "Gender", *SYMPTOMS, "class"])
    writer.writeheader()
    for _ in range(rows):
        positive = rng.random() < 0.6
        row = {"Age": rng.randint(16, 90), "Gender": rng.choice(["Male", "Female"])}
        for name in SYMPTOMS:
            row[name] = "Yes" if rng.random() < (0.7 if positive else 0.2) else "No"
        row["class"] = "Positive" if positive else "Negative"
        writer.writerow(row)
    return buffer.getvalue().encode()


def students_csv(rows: int = 200, *, include_result: bool = True) -> bytes:
    rng = random.Random(3)
    fields = [
        "student_id", "gender", "study_hours_per_week", "attendance_percentage",
        "previous_scores", "exam_score",
    ]
    if include_result:
        fields.append("result")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for index in range(rows):
        hours = round(rng.uniform(0, 12), 1)
        attendance = round(rng.uniform(50, 100), 1)
        prior = rng.randint(35, 100)
        score = max(0.0, min(100.0, 0.35 * prior + 3.2 * hours + 0.25 * attendance + rng.gauss(0, 6)))
        row = {
            "student_id": f"S{index:05d}",
            "gender": rng.choice(["male", "female"]),
            "study_hours_per_week": hours,
            "attendance_percentage": attendance,
            "previous_scores": prior,
            "exam_score": round(score, 2),
        }
        if include_result:
            row["result"] = "Pass" if score >= 50 else "Fail"
        writer.writerow(row)
    return buffer.getvalue().encode()


@pytest.fixture
def prep(settings, storage) -> DatasetPrepService:
    return DatasetPrepService(settings, DatasetService(settings, storage))


def ingest(prep: DatasetPrepService, name: str, payload: bytes) -> str:
    dataset_id = prep.create_draft(name=Path(name).stem)
    prep.stage_file(dataset_id, stream=io.BytesIO(payload), filename=name, relative_path=name)
    return dataset_id


# --- the case the rules never covered -----------------------------------------


def test_detection_alone_cannot_place_a_feature_table(prep):
    """The precondition. If this ever passes, the transform stage is dead code."""
    dataset_id = ingest(prep, "diabetes.csv", diabetes_csv())
    detection = prep.detect_dataset(dataset_id)
    assert detection.modality == "table"
    assert detection.task_type is None
    assert detection.needs_input


def test_a_feature_table_becomes_a_trainable_dataset(prep):
    dataset_id = ingest(prep, "diabetes.csv", diabetes_csv())
    summary, plan = prep.run(dataset_id)

    assert plan.task_type == "text_classification"
    assert plan.format == "text_folder"
    assert plan.labels == ["Negative", "Positive"]
    assert plan.needs_input is None
    assert plan.transform is not None
    assert plan.transform.engine == "builtin"
    assert plan.transform.target_column == "class"

    assert summary.readiness.trainable
    assert summary.splits["train"].item_count > 0
    assert summary.splits["valid"].item_count > 0


def test_the_target_comes_from_the_name_not_the_class_count(prep):
    """`gender` and `result` both hold two values; only one is the target.

    Picking the column with the fewest classes was the first rule tried, and it
    trains a gender classifier on study habits without ever looking wrong.
    """
    dataset_id = ingest(prep, "students.csv", students_csv())
    _, plan = prep.run(dataset_id)
    assert plan.transform is not None
    assert plan.transform.target_column == "result"
    assert plan.labels == ["Fail", "Pass"]


def test_a_continuous_target_is_bucketed_and_says_so(prep):
    dataset_id = ingest(prep, "scores.csv", students_csv(include_result=False))
    _, plan = prep.run(dataset_id)
    assert plan.transform is not None
    assert plan.transform.target_column == "exam_score"
    assert plan.labels == ["high", "low", "medium"]
    assert "regression" in plan.transform.rationale


def test_an_identifier_column_is_never_a_feature(prep):
    dataset_id = ingest(prep, "students.csv", students_csv())
    _, plan = prep.run(dataset_id)
    assert plan.transform is not None
    assert "student_id" not in plan.transform.feature_columns


def test_a_column_that_dominates_the_target_is_reported(prep):
    """`result` was computed from `exam_score`, so one feature explains it.

    Reported as an observation rather than a verdict: nothing computable from
    the table separates a derived column from the best honest predictor in it —
    at fine enough bucketing iris petal length scores a flat 1.00 — so the note
    states the fact and asks the question.
    """
    dataset_id = ingest(prep, "students.csv", students_csv())
    _, plan = prep.run(dataset_id)
    assert plan.transform is not None
    assert "exam_score" in plan.transform.rationale
    assert "carries nearly all the signal" in plan.transform.rationale


def test_the_prepared_items_carry_column_bound_tokens(prep):
    """`polyuria_yes`, not `Polyuria: Yes`.

    The NLP preset strips punctuation with `[^\\w\\s]`, which would sever every
    column from its value and leave a bag of bare `yes`/`no` with the pairing —
    the entire signal — destroyed. Underscores survive that regex.
    """
    dataset_id = ingest(prep, "diabetes.csv", diabetes_csv())
    prep.run(dataset_id)
    items = prep.datasets.list_items(dataset_id, split="train", limit=1)
    text = items[0].text_preview or ""
    assert "polyuria_" in text
    assert ": " not in text


# --- staging is never mutated -------------------------------------------------


def test_the_raw_upload_survives_and_the_output_lands_beside_it(prep):
    dataset_id = ingest(prep, "diabetes.csv", diabetes_csv())
    root = prep.datasets._location(dataset_id).root
    before = (staging_root(root) / "diabetes.csv").read_bytes()

    prep.run(dataset_id)

    assert (staging_root(root) / "diabetes.csv").read_bytes() == before
    assert (derived_root(root) / "prepared.jsonl").is_file()


def test_undo_drops_the_transform_output(prep):
    dataset_id = ingest(prep, "diabetes.csv", diabetes_csv())
    root = prep.datasets._location(dataset_id).root
    prep.run(dataset_id)
    prep.undo(dataset_id)
    assert not derived_root(root).exists()
    assert (staging_root(root) / "diabetes.csv").is_file()


def test_re_running_rebuilds_rather_than_doubling(prep):
    dataset_id = ingest(prep, "diabetes.csv", diabetes_csv())
    first, _ = prep.run(dataset_id)
    second, _ = prep.run(dataset_id)
    assert second.splits["train"].item_count == first.splits["train"].item_count


def test_discarding_staged_files_clears_the_derived_output(prep):
    dataset_id = ingest(prep, "diabetes.csv", diabetes_csv())
    root = prep.datasets._location(dataset_id).root
    summary, _ = prep.run(dataset_id)
    trained_items = summary.splits["train"].item_count

    summary = prep.discard_staged(dataset_id)
    assert not staging_root(root).exists()
    assert not derived_root(root).exists()
    # The prepared dataset itself is untouched; only the raw source is reclaimed.
    assert summary.splits["train"].item_count == trained_items


# --- the project gate ---------------------------------------------------------


def test_a_project_that_cannot_train_text_gets_no_transform(prep):
    dataset_id = ingest(prep, "diabetes.csv", diabetes_csv())
    _, plan = prep.run(dataset_id, allowed_task_types=["object_detection"])
    assert plan.transform is None
    assert plan.task_type is None
    assert plan.needs_input


# --- the model engine ---------------------------------------------------------


class FakeClient:
    """An OpenRouter stand-in with a scripted reply."""

    def __init__(self, body: dict) -> None:
        self.body = body
        self.calls = 0

    def chat_json(self, messages, *, model, temperature=0.0, **kwargs):
        self.calls += 1
        return self.body, LlmUsage(prompt_tokens=900, completion_tokens=300, cost_usd=0.003)


GOOD_CODE = """
def transform(rows):
    out = []
    for row in rows:
        label = (row.get('class') or '').strip()
        if not label:
            continue
        parts = [
            '%s_%s' % (key.replace(' ', '_').lower(), str(value).lower())
            for key, value in row.items()
            if key != 'class' and value not in (None, '')
        ]
        out.append({'text': ' '.join(parts), 'label': label})
    return out
"""


def plan_with_client(prep: DatasetPrepService, dataset_id: str, body: dict):
    from app.services.datasets.prep.plan import heuristic_plan

    detection = prep.detect_dataset(dataset_id)
    plan = heuristic_plan(detection)
    return transform_module.refine_with_transform(
        plan,
        detection,
        root=prep.datasets._location(dataset_id).root,
        model="test/model",
        client=FakeClient(body),
    )


def test_a_model_written_transform_is_used_when_it_holds_up(prep):
    dataset_id = ingest(prep, "diabetes.csv", diabetes_csv())
    plan = plan_with_client(
        prep,
        dataset_id,
        {
            "task_type": "text_classification",
            "code": GOOD_CODE,
            "target_column": "class",
            "rationale": "One token per symptom.",
        },
    )
    assert plan.transform is not None
    assert plan.transform.engine == "llm"
    assert plan.transform.prompt_tokens == 900
    assert plan.labels == ["Negative", "Positive"]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # A taxonomy the data does not support — the damaging failure mode.
        (
            {
                "task_type": "text_classification",
                "code": "def transform(rows):\n    return [{'text': 'x', 'label': 'diabetic'} for r in rows]\n",
            },
            "single label",
        ),
        # Selective to the point of losing the dataset.
        (
            {
                "task_type": "text_classification",
                "code": "def transform(rows):\n    return [{'text': str(r), 'label': r['class']} for r in rows[:5]]\n",
            },
            "loses most of the dataset",
        ),
        # Code that will not run at all.
        (
            {"task_type": "text_classification", "code": "def transform(rows)\n    return rows\n"},
            "did not run",
        ),
        # Code that reaches for the network.
        (
            {
                "task_type": "text_classification",
                "code": "import urllib.request\ndef transform(rows):\n    return rows\n",
            },
            "did not run",
        ),
        # A task this stage does not produce.
        ({"task_type": "object_detection", "code": GOOD_CODE}, "cannot train"),
        # No code at all.
        ({"task_type": "text_classification"}, "no transform code"),
    ],
)
def test_a_bad_model_transform_falls_back_and_names_the_reason(prep, body, expected):
    dataset_id = ingest(prep, "diabetes.csv", diabetes_csv())
    plan = plan_with_client(prep, dataset_id, body)
    assert plan.transform is not None
    assert plan.transform.engine == "builtin"
    assert expected in (plan.transform.notice or "")
    # Degraded, never dead: the dataset is still trainable.
    assert plan.task_type == "text_classification"
    assert plan.labels == ["Negative", "Positive"]


def test_without_a_key_the_plan_says_so(prep):
    dataset_id = ingest(prep, "diabetes.csv", diabetes_csv())
    _, plan = prep.run(dataset_id)
    assert plan.transform is not None
    assert "No OpenRouter key" in (plan.transform.notice or "")


# --- target selection in isolation --------------------------------------------


def rows_of(payload: bytes) -> list[dict]:
    return list(csv.DictReader(payload.decode().splitlines()))


def test_choose_target_prefers_the_last_column_when_nothing_is_named():
    rows = [
        {"feature_one": str(index % 7), "feature_two": str(index % 5), "zzz": "a" if index % 2 else "b"}
        for index in range(60)
    ]
    columns = list(rows[0])
    choice = transform_module.choose_target(columns, profile_columns(rows, columns), rows)
    assert choice is not None
    assert choice[0] == "zzz"
    assert "last column" in choice[2]


def test_choose_target_avoids_demographics_when_nothing_declares_itself():
    rows = [
        {
            "gender": "male" if index % 2 else "female",
            "internet_access": "Yes" if index % 3 else "No",
            "hours": str(index),
        }
        for index in range(60)
    ]
    columns = list(rows[0])
    choice = transform_module.choose_target(columns, profile_columns(rows, columns), rows)
    assert choice is not None
    assert choice[0] != "gender"
