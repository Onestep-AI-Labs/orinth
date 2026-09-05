"""The agent end to end: ingest raw files, plan, apply, and be trainable.

These drive the real `DatasetService` over a temp storage root — no mocks below
the LLM boundary — because the thing worth proving is that the files land where
the training runners actually look for them.
"""

import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from app.schemas import DatasetPrepPlan
from app.services.datasets import DatasetService
from app.services.datasets.prep.apply import PrepApplyError
from app.services.datasets.prep.service import DatasetPrepService, PrepBusyError
from app.services.datasets.prep.staging import safe_relative_path, staging_root


@pytest.fixture
def prep(settings, storage) -> DatasetPrepService:
    return DatasetPrepService(settings, DatasetService(settings, storage))


@pytest.fixture
def prep_async(settings, storage):
    """A service with a real pool, so `start()` takes its background path."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-prep")
    yield DatasetPrepService(settings, DatasetService(settings, storage), None, executor)
    executor.shutdown(wait=True)


def wait_for_prep(service: DatasetPrepService, dataset_id: str, timeout: float = 60.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = service.prep_status(dataset_id).state
        if state not in {"detecting", "planning", "applying"}:
            return state
        time.sleep(0.05)
    raise AssertionError("prep did not settle")


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def stage(prep: DatasetPrepService, dataset_id: str, relative: str, payload: bytes) -> bool:
    return prep.stage_file(
        dataset_id,
        stream=io.BytesIO(payload),
        filename=Path(relative).name,
        relative_path=relative,
    )


STUDENT_COLUMNS = [
    "student_id", "age", "gender", "study_hours_per_day", "attendance_pct",
    "sleep_hours", "previous_scores", "tutoring_sessions", "exam_score",
    "pass_status",
]


def student_exam_csv(rows: int = 240) -> bytes:
    """A feature table shaped like the upload that produced this test: numeric
    features, an outcome column, and no column any detection rule maps."""
    lines = [",".join(STUDENT_COLUMNS)]
    for index in range(rows):
        passed = index % 3 != 0
        lines.append(
            ",".join(
                [
                    f"S{index:04d}",
                    str(18 + index % 5),
                    "male" if index % 2 else "female",
                    str(1 + index % 8),
                    str(60 + index % 40),
                    str(5 + index % 5),
                    str(40 + index % 55),
                    str(index % 4),
                    str(75 + index % 20 if passed else 30 + index % 20),
                    "Pass" if passed else "Fail",
                ]
            )
        )
    return ("\n".join(lines) + "\n").encode()


# --- staging preserves the thing detection reads ------------------------------


def test_ingest_preserves_directory_structure(prep):
    dataset_id = prep.create_draft(name="scans")
    stage(prep, dataset_id, "normal/a.png", png_bytes())
    stage(prep, dataset_id, "kista/b.png", png_bytes())

    root = staging_root(prep.datasets._location(dataset_id).root)
    assert (root / "normal" / "a.png").is_file()
    assert (root / "kista" / "b.png").is_file()


def test_a_draft_is_not_trainable_and_says_it_needs_prep(prep):
    dataset_id = prep.create_draft(name="scans")
    stage(prep, dataset_id, "normal/a.png", png_bytes())
    summary = prep.datasets.summary(dataset_id)
    assert summary.prep is not None and summary.prep.state == "draft"
    assert summary.readiness.trainable is False
    assert summary.readiness.next_action == "run_prep"


@pytest.mark.parametrize(
    "hostile",
    ["../../etc/passwd", "/etc/passwd", "..\\..\\windows\\system32", "~/secrets"],
)
def test_traversal_paths_are_neutralized(prep, hostile, tmp_path):
    dataset_id = prep.create_draft(name="x")
    stage(prep, dataset_id, hostile, b"x")
    root = staging_root(prep.datasets._location(dataset_id).root).resolve()
    for path in root.rglob("*"):
        if path.is_file():
            assert root in path.resolve().parents


def test_safe_relative_path_strips_traversal_but_keeps_real_nesting():
    assert safe_relative_path("a/../../b/c.txt", "c.txt") == Path("a/b/c.txt")
    assert safe_relative_path("train/images/x.jpg", "x.jpg") == Path("train/images/x.jpg")
    assert safe_relative_path("", "fallback.txt") == Path("fallback.txt")
    assert safe_relative_path("../..", "") is None


# --- the whole agent, per modality --------------------------------------------


def test_class_folders_become_a_trainable_classification_dataset(prep):
    dataset_id = prep.create_draft(name="scans")
    for label in ("normal", "kista"):
        for index in range(6):
            stage(prep, dataset_id, f"{label}/{index}.png", png_bytes())

    summary, plan = prep.run(dataset_id)

    assert plan.task_type == "classification"
    assert plan.labels == ["kista", "normal"]
    assert summary.task_type == "classification"
    assert summary.labels == ["kista", "normal"]
    # The point of the whole exercise.
    assert summary.readiness.trainable is True
    assert summary.prep is not None and summary.prep.state == "ready"


def test_applied_images_land_where_the_training_runner_looks(prep):
    """`keras_common.load_split` reads `<split>/images` + `<split>/annotations`."""
    dataset_id = prep.create_draft(name="scans")
    for label in ("normal", "kista"):
        for index in range(6):
            stage(prep, dataset_id, f"{label}/{index}.png", png_bytes())

    prep.run(dataset_id)
    root = prep.datasets._location(dataset_id).root

    images = list((root / "train" / "images").glob("*.png"))
    assert images
    for image_path in images:
        sidecar = root / "train" / "annotations" / f"{image_path.stem}.json"
        assert sidecar.is_file(), f"{image_path.name} has no annotation"
        rows = json.loads(sidecar.read_text())["annotations"]
        # An empty annotation list is exactly what `load_split` skips silently.
        assert rows and rows[0]["class_name"] in {"normal", "kista"}


def test_alpaca_jsonl_becomes_a_trainable_llm_dataset(prep):
    dataset_id = prep.create_draft(name="sft")
    rows = "\n".join(
        json.dumps({"instruction": f"q{i}", "input": "", "output": f"a{i}"}) for i in range(40)
    )
    stage(prep, dataset_id, "data.jsonl", rows.encode())

    summary, plan = prep.run(dataset_id)

    assert plan.task_type == "llm_finetune"
    assert plan.format == "instruction_jsonl"
    assert summary.readiness.trainable is True
    # `llm_sft.load_jsonl_records` reads `<split>/data.jsonl`.
    root = prep.datasets._location(dataset_id).root
    assert (root / "train" / "data.jsonl").is_file()
    assert summary.splits["train"].item_count > 0


def test_a_labelled_csv_becomes_a_trainable_text_classifier(prep):
    dataset_id = prep.create_draft(name="reviews")
    rows = ["text,label"] + [
        f"review number {i},{'positive' if i % 2 else 'negative'}" for i in range(20)
    ]
    stage(prep, dataset_id, "reviews.csv", "\n".join(rows).encode())

    summary, plan = prep.run(dataset_id)

    assert plan.task_type == "text_classification"
    assert set(plan.labels) == {"negative", "positive"}
    assert summary.readiness.trainable is True
    root = prep.datasets._location(dataset_id).root
    texts = list((root / "train" / "texts").glob("*.txt"))
    assert texts
    sidecar = root / "train" / "annotations" / f"{texts[0].stem}.json"
    rows_out = json.loads(sidecar.read_text())["annotations"]
    assert rows_out[0]["class_name"] in {"negative", "positive"}


def test_a_yolo_upload_keeps_its_labels_and_declared_splits(prep):
    dataset_id = prep.create_draft(name="det")
    stage(prep, dataset_id, "data.yaml", b"names: ['lesion', 'cyst']\n")
    for split in ("train", "valid"):
        for index in range(4):
            stage(prep, dataset_id, f"{split}/images/{index}.png", png_bytes())
            stage(prep, dataset_id, f"{split}/labels/{index}.txt", b"0 0.5 0.5 0.2 0.2\n")

    summary, plan = prep.run(dataset_id)

    assert plan.task_type == "object_detection"
    assert plan.labels == ["lesion", "cyst"]
    root = prep.datasets._location(dataset_id).root
    # An author's declared train/valid boundary is respected, not reshuffled.
    assert summary.splits["train"].item_count == 4
    assert summary.splits["valid"].item_count == 4
    label_files = list((root / "train" / "labels").glob("*.txt"))
    assert any(path.read_text().strip() for path in label_files), "box geometry was lost"
    assert (root / "data.yaml").is_file()


# --- refusing to guess --------------------------------------------------------


def test_unlabelled_images_stop_at_needs_input_rather_than_inventing_labels(prep):
    dataset_id = prep.create_draft(name="pile")
    for index in range(5):
        stage(prep, dataset_id, f"{index}.png", png_bytes())

    summary, plan = prep.run(dataset_id)

    assert plan.needs_input is not None
    assert plan.labels == []
    assert summary.readiness.trainable is False
    # Auto-apply must not have run: there is nothing defensible to apply.
    assert summary.prep is not None and summary.prep.state == "planned"


def test_applying_an_incomplete_plan_is_refused(prep):
    dataset_id = prep.create_draft(name="pile")
    stage(prep, dataset_id, "a.png", png_bytes())
    _summary, plan = prep.run(dataset_id)

    with pytest.raises(PrepApplyError):
        prep.apply(dataset_id, plan)


def test_a_task_the_project_forbids_is_not_proposed(prep):
    dataset_id = prep.create_draft(name="scans")
    for label in ("normal", "kista"):
        stage(prep, dataset_id, f"{label}/a.png", png_bytes())

    _summary, plan = prep.run(dataset_id, allowed_task_types=["llm_finetune"])
    assert plan.task_type is None
    assert plan.needs_input is not None


# --- reversibility ------------------------------------------------------------


def test_undo_restores_the_pre_agent_manifest(prep):
    dataset_id = prep.create_draft(name="scans")
    for label in ("normal", "kista"):
        for index in range(6):
            stage(prep, dataset_id, f"{label}/{index}.png", png_bytes())

    before = prep.datasets.summary(dataset_id)
    assert before.labels == []

    prep.run(dataset_id)
    assert prep.datasets.summary(dataset_id).labels == ["kista", "normal"]

    restored = prep.undo(dataset_id)
    assert restored.labels == []
    assert restored.task_type == before.task_type


def test_undo_without_a_prior_run_is_refused(prep):
    dataset_id = prep.create_draft(name="x")
    with pytest.raises(PrepApplyError):
        prep.undo(dataset_id)


def test_raw_files_survive_apply_so_a_replan_needs_no_reupload(prep):
    """This retention is what makes auto-apply safe rather than one-way."""
    dataset_id = prep.create_draft(name="scans")
    for label in ("normal", "kista"):
        for index in range(6):
            stage(prep, dataset_id, f"{label}/{index}.png", png_bytes())

    prep.run(dataset_id)
    root = prep.datasets._location(dataset_id).root
    assert list(staging_root(root).rglob("*.png"))

    # And can be reclaimed deliberately.
    prep.discard_staged(dataset_id)
    assert not list(staging_root(root).rglob("*.png"))


def test_the_staging_directory_never_appears_in_the_catalog(prep):
    dataset_id = prep.create_draft(name="scans")
    stage(prep, dataset_id, "normal/a.png", png_bytes())
    ids = {dataset.id for dataset in prep.datasets.list_datasets()}
    assert dataset_id in ids
    assert not any(id_.startswith("_staging") for id_ in ids)


# --- review-gate variant ------------------------------------------------------


def test_auto_apply_false_stops_at_planned(prep):
    dataset_id = prep.create_draft(name="scans")
    for label in ("normal", "kista"):
        for index in range(6):
            stage(prep, dataset_id, f"{label}/{index}.png", png_bytes())

    summary, plan = prep.run(dataset_id, auto_apply=False)
    assert plan.task_type == "classification"
    assert summary.prep is not None and summary.prep.state == "planned"
    assert summary.readiness.trainable is False

    # And the same plan applies cleanly afterwards.
    applied = prep.apply(dataset_id, plan)
    assert applied.readiness.trainable is True


def test_an_edited_plan_is_honoured(prep):
    """The user tweaks; the agent does not overrule them."""
    dataset_id = prep.create_draft(name="scans")
    for label in ("normal", "kista"):
        for index in range(10):
            stage(prep, dataset_id, f"{label}/{index}.png", png_bytes())

    _summary, plan = prep.run(dataset_id, auto_apply=False)
    edited = DatasetPrepPlan.model_validate(plan.model_dump())
    edited.split.train, edited.split.valid, edited.split.test = 0.5, 0.5, 0.0

    applied = prep.apply(dataset_id, edited)
    assert applied.splits["test"].item_count == 0
    assert applied.splits["valid"].item_count > 0


def test_an_empty_sidecar_is_not_written_over_yolo_labels(prep):
    """Regression: an empty sidecar masks the YOLO label file behind it.

    `format_io._annotations` returns the JSON sidecar whenever one *exists*,
    checking YOLO only when it does not. Writing `{"annotations": []}` for every
    image therefore hid real geometry: on the repo's dental dataset, 747
    annotated images counted as zero and the dataset read as untrainable.
    """
    dataset_id = prep.create_draft(name="det")
    stage(prep, dataset_id, "data.yaml", b"names: ['lesion']\n")
    for split in ("train", "valid"):
        for index in range(3):
            stage(prep, dataset_id, f"{split}/images/{index}.png", png_bytes())
            stage(prep, dataset_id, f"{split}/labels/{index}.txt", b"0 0.5 0.5 0.2 0.2\n")

    summary, _plan = prep.run(dataset_id)
    root = prep.datasets._location(dataset_id).root

    train_images = sorted((root / "train" / "images").glob("*.png"))
    assert train_images
    for image_path in train_images:
        sidecar = root / "train" / "annotations" / f"{image_path.stem}.json"
        assert not sidecar.exists(), "an empty sidecar would hide the YOLO label"

    assert summary.splits["train"].annotation_count > 0
    assert summary.readiness.trainable is True


def test_classification_still_gets_a_sidecar_because_folders_are_its_only_source(prep):
    """The counterpart: with no YOLO file to fall back to, the sidecar is required."""
    dataset_id = prep.create_draft(name="cls")
    for label in ("normal", "kista"):
        for index in range(6):
            stage(prep, dataset_id, f"{label}/{index}.png", png_bytes())

    prep.run(dataset_id)
    root = prep.datasets._location(dataset_id).root
    images = sorted((root / "train" / "images").glob("*.png"))
    assert images
    for image_path in images:
        assert (root / "train" / "annotations" / f"{image_path.stem}.json").is_file()


def test_splitting_thousands_of_rows_does_not_go_quadratic(prep):
    """Regression: `_item_path` re-listed the whole directory on every move.

    `_move_item` calls it once per item, so splitting was O(n^2) in directory
    listings — importing a 5,500-row CSV meant 5,500 full scans and took long
    enough to look hung. A direct path lookup makes it O(n).

    The assertion is on wall time deliberately: the defect was not a wrong
    answer, it was an answer that arrived too late to be useful.
    """
    import time

    rows = ["text,label"] + [
        f"a review body number {i} with some words in it,{'pos' if i % 2 else 'neg'}"
        for i in range(3000)
    ]
    dataset_id = prep.create_draft(name="big")
    stage(prep, dataset_id, "reviews.csv", "\n".join(rows).encode())

    started = time.monotonic()
    summary, _plan = prep.run(dataset_id)
    elapsed = time.monotonic() - started

    assert summary.readiness.trainable is True
    assert sum(split.item_count for split in summary.splits.values()) == 3000
    # Quadratic behaviour took minutes here; linear takes a couple of seconds.
    assert elapsed < 60, f"splitting 3000 rows took {elapsed:.0f}s"


def test_every_row_is_imported_by_default(prep):
    """No cap unless one is configured.

    There used to be a hard 20,000-row ceiling, with a warning explaining the
    truncation *after* the import. A dataset that quietly does not match the file
    the user chose is the worse failure, so the default is now every row and the
    ceiling is opt-in (`PREP_MAX_INGEST_ROWS`).
    """
    over = 2_500
    rows = ["text,label"] + [
        f"body number {i} with enough words to read as prose,{'pos' if i % 2 else 'neg'}"
        for i in range(over)
    ]
    dataset_id = prep.create_draft(name="huge")
    stage(prep, dataset_id, "big.csv", "\n".join(rows).encode())

    summary, plan = prep.run(dataset_id)
    total = sum(split.item_count for split in summary.splits.values())
    assert total == over
    assert not any("were imported" in warning for warning in plan.warnings)


def test_a_configured_row_cap_is_reported_rather_than_silently_truncating(prep):
    prep.settings.prep_max_ingest_rows = 100
    try:
        rows = ["text,label"] + [
            f"body number {i} with enough words to read as prose,{'pos' if i % 2 else 'neg'}"
            for i in range(250)
        ]
        dataset_id = prep.create_draft(name="capped")
        stage(prep, dataset_id, "big.csv", "\n".join(rows).encode())

        summary, plan = prep.run(dataset_id)
        total = sum(split.item_count for split in summary.splits.values())
        assert total == 100
        assert any("PREP_MAX_INGEST_ROWS" in warning for warning in plan.warnings)
    finally:
        prep.settings.prep_max_ingest_rows = 0


def test_the_apply_stage_counts_the_items_it_writes(prep):
    """A bar needs a denominator, and `processed`/`total` is where it comes from."""
    ticks: list[tuple[int, int]] = []
    original = prep.ticker

    def spy(dataset_id, state, step, **kwargs):
        inner = original(dataset_id, state, step, **kwargs)

        def tick(processed, total, detail, **kwargs):
            ticks.append((processed, total))
            return inner(processed, total, detail, **kwargs)

        return tick

    prep.ticker = spy  # type: ignore[method-assign]

    dataset_id = prep.create_draft(name="student exam")
    stage(prep, dataset_id, "student_exam_performance.csv", student_exam_csv())
    prep.run(dataset_id)

    assert ticks, "the apply stage reported no progress at all"
    assert all(total > 0 for _processed, total in ticks)
    assert ticks[-1][0] == ticks[-1][1], "the last tick must land on 100%"


def test_the_bar_never_travels_backwards_across_a_run(prep):
    """Apply is two counted passes — write the items, then place them into
    splits — and each counts to its own 100%. The counts have to stay honest, so
    it is the *bar* that is banded; if it were not, it would reach 100%, reset,
    and climb again."""
    fractions: list[float] = []
    original = prep._set_state

    def record(dataset_id, state, plan=None, **kwargs):
        if kwargs.get("progress") is not None:
            fractions.append(kwargs["progress"])
        return original(dataset_id, state, plan, **kwargs)

    prep._set_state = record  # type: ignore[method-assign]

    dataset_id = prep.create_draft(name="student exam")
    stage(prep, dataset_id, "student_exam_performance.csv", student_exam_csv())
    prep.run(dataset_id)

    assert fractions == sorted(fractions), f"progress went backwards: {fractions}"
    assert fractions[-1] == 1.0


# --- a feature table becomes trainable ----------------------------------------


def test_a_feature_table_no_rule_maps_becomes_trainable(prep):
    """The reported failure, end to end: a student-exam CSV that detection cannot
    map used to apply as image classification and produce an empty dataset."""
    dataset_id = prep.create_draft(name="student exam")
    stage(prep, dataset_id, "student_exam_performance.csv", student_exam_csv())

    summary, plan = prep.run(dataset_id)

    assert plan.task_type == "text_classification"
    assert plan.transform is not None, "the sandbox should have written a transform"
    assert len(plan.labels) >= 2
    assert summary.readiness.trainable is True
    assert summary.splits["train"].item_count > 0
    assert summary.splits["valid"].item_count > 0


def test_a_model_naming_an_image_task_for_a_table_does_not_empty_the_dataset(prep, monkeypatch):
    """The exact model reply from the report: `classification`, with every
    mapping role filled from real column names."""
    from app.services.datasets.prep import service as service_module
    from app.services.providers import LlmUsage

    class FakeClient:
        def chat_json(self, messages, *, model, temperature=0.0, **kwargs):
            return (
                {
                    "task_type": "classification",
                    "confidence": 0.7,
                    "labels": ["pass_status", "performance_level"],
                    "field_mapping": {
                        "text": "student_id", "label": "pass_status",
                        "question": "exam_score", "answer": "attendance_pct",
                    },
                    "rationale": "These features can be used for classification.",
                },
                LlmUsage(prompt_tokens=100, completion_tokens=30, cost_usd=0.0),
            )

    monkeypatch.setattr(
        service_module.DatasetPrepService, "_llm_config", lambda self: ("key", "m")
    )
    monkeypatch.setattr(
        "app.services.providers.OpenRouterClient", lambda **kwargs: FakeClient()
    )

    dataset_id = prep.create_draft(name="student exam")
    stage(prep, dataset_id, "student_exam_performance.csv", student_exam_csv())

    summary, plan = prep.run(dataset_id)

    assert any("table data" in warning for warning in plan.warnings), (
        f"the fake client was not consulted: {plan.warnings}"
    )
    assert plan.task_type != "classification"
    assert summary.readiness.trainable is True
    assert summary.splits["train"].item_count > 0


def test_a_previous_runs_transform_output_is_not_ingested_by_the_next_plan(prep):
    """`_derived/` replaces the raw rows in apply, so output left by an earlier
    run would silently become this run's dataset."""
    from app.services.datasets.prep.staging import derived_root

    dataset_id = prep.create_draft(name="scans")
    stage(prep, dataset_id, "normal/a.png", png_bytes())
    stage(prep, dataset_id, "kista/b.png", png_bytes())

    derived = derived_root(prep.datasets._location(dataset_id).root)
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "prepared.jsonl").write_text(
        '{"text": "left over from a previous upload", "label": "stale"}\n',
        encoding="utf-8",
    )

    _, plan = prep.run(dataset_id)

    assert plan.task_type == "classification"
    assert "stale" not in plan.labels
    assert not (derived / "prepared.jsonl").exists()


# --- the run happens off the request path -------------------------------------


def test_start_returns_before_the_run_finishes(prep_async):
    """Applying a large upload is tens of thousands of file operations. Holding
    the request open for it is what reset the dev proxy's socket."""
    dataset_id = prep_async.create_draft(name="student exam")
    stage(prep_async, dataset_id, "student_exam_performance.csv", student_exam_csv())

    queued = prep_async.start(dataset_id)

    # The state is written before returning: a client that polls immediately
    # must not see `draft` and conclude nothing happened.
    assert queued.prep is not None and queued.prep.state == "planning"
    assert queued.readiness.next_action == "wait"

    assert wait_for_prep(prep_async, dataset_id) == "ready"
    assert prep_async.datasets.summary(dataset_id).readiness.trainable is True


def test_a_second_run_on_the_same_dataset_is_refused(prep_async, monkeypatch):
    """Two runs would interleave one's `_reset_items` with the other's writes."""
    started = threading.Event()
    release = threading.Event()

    def blocking_run(self, dataset_id, **kwargs):
        started.set()
        release.wait(timeout=10)
        return self.datasets.summary(dataset_id), None

    monkeypatch.setattr(DatasetPrepService, "run", blocking_run)

    dataset_id = prep_async.create_draft(name="busy")
    stage(prep_async, dataset_id, "a.csv", student_exam_csv(20))
    prep_async.start(dataset_id)
    assert started.wait(timeout=10)

    try:
        with pytest.raises(PrepBusyError):
            prep_async.start(dataset_id)
    finally:
        release.set()


def test_a_background_failure_is_recorded_where_the_studio_can_show_it(prep_async, monkeypatch):
    """Nothing awaits the worker, so a lost exception is a dataset stuck on
    "preparing" forever."""

    def exploding_run(self, dataset_id, **kwargs):
        raise RuntimeError("the transform sandbox died")

    monkeypatch.setattr(DatasetPrepService, "run", exploding_run)

    dataset_id = prep_async.create_draft(name="doomed")
    stage(prep_async, dataset_id, "a.csv", student_exam_csv(20))
    prep_async.start(dataset_id)

    assert wait_for_prep(prep_async, dataset_id) == "failed"
    status = prep_async.prep_status(dataset_id)
    assert status.error is not None and "sandbox died" in status.error


def test_a_new_run_clears_the_previous_failure(prep_async):
    dataset_id = prep_async.create_draft(name="retry")
    stage(prep_async, dataset_id, "student_exam_performance.csv", student_exam_csv())
    prep_async._record_failure(dataset_id, "an older problem")
    assert prep_async.prep_status(dataset_id).error == "an older problem"

    prep_async.start(dataset_id)

    assert wait_for_prep(prep_async, dataset_id) == "ready"
    assert prep_async.prep_status(dataset_id).error is None


def test_without_an_executor_start_runs_inline(prep):
    """The desktop build and the tests want the agent to just finish."""
    dataset_id = prep.create_draft(name="inline")
    stage(prep, dataset_id, "student_exam_performance.csv", student_exam_csv())

    summary = prep.start(dataset_id)

    assert summary.prep is not None and summary.prep.state == "ready"
    assert summary.readiness.trainable is True



# --- progress reporting -------------------------------------------------------


def test_a_run_names_each_stage_as_it_reaches_it(prep):
    """The readout used to be the word "working" for the whole run.

    What is asserted here is not the exact wording — that will change — but that
    every stage transition is recorded with a step, a sentence, and a fraction,
    and that the sentence carries a real count. A number that moves is the
    difference between a progress readout and a hung one.
    """
    seen: list[tuple[str, str, float | None]] = []
    original = prep._set_state

    def record(dataset_id, state, plan=None, **kwargs):
        step = kwargs.get("step")
        if step is not None:
            seen.append((step, kwargs.get("detail") or "", kwargs.get("progress")))
        return original(dataset_id, state, plan, **kwargs)

    prep._set_state = record  # type: ignore[method-assign]

    dataset_id = prep.create_draft(name="student exam")
    stage(prep, dataset_id, "student_exam_performance.csv", student_exam_csv())
    prep.run(dataset_id)

    steps = [step for step, _, _ in seen]
    assert steps[0] == "detecting"
    assert "planning" in steps
    assert "applying" in steps
    assert steps[-1] == "done"

    # Every stage said something, and the fractions only ever move forward.
    assert all(detail for _, detail, _ in seen)
    fractions = [progress for _, _, progress in seen if progress is not None]
    assert fractions == sorted(fractions)

    planning = next(detail for step, detail, _ in seen if step == "planning")
    assert "1 file" in planning, planning


def test_the_finished_status_says_what_was_produced(prep):
    dataset_id = prep.create_draft(name="student exam")
    stage(prep, dataset_id, "student_exam_performance.csv", student_exam_csv())
    prep.run(dataset_id)

    status = prep.prep_status(dataset_id)
    assert status.state == "ready"
    assert status.step == "done"
    assert status.progress == 1.0
    # The count comes from the splits that were actually written, so it is a
    # report rather than an echo of the plan.
    assert "items as" in status.detail


def test_a_failed_run_records_the_reason_as_its_detail(prep, monkeypatch):
    monkeypatch.setattr(
        "app.services.datasets.prep.service.detect",
        lambda root: (_ for _ in ()).throw(RuntimeError("disk went away")),
    )
    dataset_id = prep.create_draft(name="broken")
    stage(prep, dataset_id, "a.csv", b"a,b\n1,2\n")

    with pytest.raises(RuntimeError):
        prep.run(dataset_id)

    status = prep.prep_status(dataset_id)
    assert status.state == "failed"
    assert status.step == "idle"
    assert status.detail


def test_a_prep_left_mid_flight_by_a_restart_is_reconciled(settings, tmp_path):
    """A prep state is a string on a manifest with nothing running behind it.

    An interrupted run therefore reads `planning` forever — and the studio polls
    the dataset catalog every four seconds for as long as any dataset does,
    which is the most expensive read in the app. Found on a real workspace: one
    dataset had been `planning` since a crash weeks earlier.
    """
    from app.core.storage import Storage
    from app.services.datasets import DatasetService
    from app.services.datasets.prep.service import DatasetPrepService

    storage = Storage(settings)
    storage.ensure()
    datasets = DatasetService(settings, storage)
    prep = DatasetPrepService(settings, datasets)

    dataset_id = prep.create_draft(name="interrupted")
    prep._set_state(dataset_id, "planning")
    assert (datasets._location(dataset_id).metadata or {})["prep"]["state"] == "planning"

    assert prep.reconcile_stale_preps() == 1

    after = (datasets._location(dataset_id).metadata or {})["prep"]
    assert after["state"] == "failed"
    assert "restarted" in after["error"]
    # Idempotent: a second startup has nothing left to do.
    assert prep.reconcile_stale_preps() == 0
