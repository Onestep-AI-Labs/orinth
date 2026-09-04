"""Readiness rules, one case per row of the phase-21 rule table.

These are pure-function tests on purpose: `readiness_for` must never touch the
filesystem, because `DatasetService.summary` calls it for every dataset on every
catalog list. A test here that needed a `tmp_path` would mean the contract had
been broken.
"""

import pytest

from app.schemas import DatasetSplitSummary
from app.services.datasets.prep.readiness import (
    MIN_LLM_TRAIN_RECORDS,
    readiness_for,
)


def splits(**counts: tuple[int, int]) -> dict[str, DatasetSplitSummary]:
    """Build a split map from ``split=(item_count, annotation_count)`` pairs."""
    return {
        name: DatasetSplitSummary(
            split=name,  # type: ignore[arg-type]
            item_count=items,
            annotation_count=annotations,
        )
        for name, (items, annotations) in counts.items()
    }


def check(readiness, check_id: str):
    return next(entry for entry in readiness.checks if entry.id == check_id)


# --- the happy paths, one per task family ------------------------------------


def test_classification_with_labels_and_annotations_is_ready():
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["normal", "kista"],
        splits=splits(train=(8, 8), valid=(2, 2), test=(1, 1), unassigned=(0, 0)),
    )
    assert result.state == "ready"
    assert result.trainable is True
    assert result.summary == "Ready to train."
    assert result.next_action == "none"


def test_detection_is_ready_with_a_single_label():
    """Detection is single-class-viable; only classifiers need two."""
    result = readiness_for(
        task_type="object_detection",
        format="yolo",
        labels=["lesion"],
        splits=splits(train=(20, 40), valid=(5, 9), unassigned=(0, 0)),
    )
    assert result.trainable is True


def test_llm_finetune_is_ready_at_the_record_floor():
    result = readiness_for(
        task_type="llm_finetune",
        format="instruction_jsonl",
        labels=[],
        splits=splits(
            train=(MIN_LLM_TRAIN_RECORDS, MIN_LLM_TRAIN_RECORDS),
            valid=(3, 3),
            unassigned=(0, 0),
        ),
    )
    assert result.trainable is True


def test_language_modeling_needs_only_a_train_split():
    result = readiness_for(
        task_type="language_modeling",
        format="text_folder",
        labels=[],
        splits=splits(train=(50, 0), unassigned=(0, 0)),
    )
    assert result.trainable is True


# --- blocking rules -----------------------------------------------------------


def test_empty_dataset_asks_for_data_not_labels():
    """"Upload data" beats "add labels" when there is nothing there at all."""
    result = readiness_for(
        task_type="classification", format="image_folder", labels=[], splits=splits()
    )
    assert result.state == "needs_input"
    assert result.next_action == "upload"
    assert "empty" in result.summary


def test_classification_requires_two_labels():
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["normal"],
        splits=splits(train=(8, 8), valid=(2, 2)),
    )
    assert result.trainable is False
    assert result.next_action == "label"
    assert "at least 2 class labels" in result.summary


def test_empty_train_split_is_fixable_by_the_agent():
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["a", "b"],
        splits=splits(train=(0, 0), valid=(0, 0), unassigned=(30, 30)),
    )
    assert result.state == "needs_prep"
    assert result.next_action == "run_prep"
    assert check(result, "train_non_empty").passed is False


def test_unannotated_images_block_rather_than_training_on_nothing():
    """`keras_common.load_split` skips unannotated images, so this must be loud."""
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["a", "b"],
        splits=splits(train=(40, 0), valid=(10, 0), unassigned=(0, 0)),
    )
    assert result.trainable is False
    assert result.next_action == "label"
    assert check(result, "train_annotated").passed is False


def test_empty_valid_blocks_image_tasks():
    """`training_root` raises FileNotFoundError without `valid/images`."""
    result = readiness_for(
        task_type="segmentation",
        format="yolo",
        labels=["lesion"],
        splits=splits(train=(20, 20), valid=(0, 0), unassigned=(0, 0)),
    )
    assert result.trainable is False
    assert check(result, "valid_non_empty").severity == "blocking"


def test_empty_valid_only_warns_for_nlp():
    result = readiness_for(
        task_type="text_classification",
        format="text_folder",
        labels=["pos", "neg"],
        splits=splits(train=(40, 40), valid=(0, 0), unassigned=(0, 0)),
    )
    assert result.trainable is True
    assert check(result, "valid_non_empty").severity == "advisory"


@pytest.mark.parametrize("count", [1, MIN_LLM_TRAIN_RECORDS - 1])
def test_llm_under_the_record_floor_is_not_ready(count: int):
    """`llm_sft.require_min_train_records` raises below 10; catch it here first."""
    result = readiness_for(
        task_type="llm_finetune",
        format="instruction_jsonl",
        labels=[],
        splits=splits(train=(count, count), unassigned=(0, 0)),
    )
    assert result.trainable is False
    assert str(MIN_LLM_TRAIN_RECORDS) in result.summary


def test_an_empty_llm_dataset_says_upload_not_record_count():
    """With nothing uploaded, "add 10 records" is worse advice than "upload data"."""
    result = readiness_for(
        task_type="llm_finetune",
        format="instruction_jsonl",
        labels=[],
        splits=splits(train=(0, 0), unassigned=(0, 0)),
    )
    assert result.trainable is False
    assert result.next_action == "upload"
    assert "empty" in result.summary


def test_llm_counts_the_inbox_because_the_runner_falls_back_to_it():
    """`resolve_train_valid_records` trains off `unassigned` when train is empty."""
    result = readiness_for(
        task_type="llm_finetune",
        format="instruction_jsonl",
        labels=[],
        splits=splits(train=(0, 0), unassigned=(40, 40)),
    )
    assert result.trainable is True
    assert check(result, "train_non_empty").severity == "advisory"


def test_non_llm_format_cannot_be_fine_tuned():
    result = readiness_for(
        task_type="llm_finetune",
        format="csv",
        labels=[],
        splits=splits(train=(40, 40), unassigned=(0, 0)),
    )
    assert result.trainable is False
    assert check(result, "llm_format").passed is False


# --- prep lifecycle -----------------------------------------------------------


def test_an_apply_in_flight_blocks_every_other_verdict():
    """Apply rewrites the splits, so no verdict computed now survives it."""
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["a", "b"],
        splits=splits(train=(8, 8), valid=(2, 2), unassigned=(0, 0)),
        metadata={"prep": {"state": "applying"}},
    )
    assert result.state == "blocked"
    assert result.next_action == "wait"
    assert result.busy is True


@pytest.mark.parametrize("state", ["detecting", "planning"])
def test_re_analysing_a_ready_dataset_keeps_it_ready(state: str):
    """The bug this contract was rewritten for.

    Detect and plan only read `_staging/`. A user who asks Orinth to look at an
    already-trainable dataset again used to watch it drop to "Preparing…" and
    then, if the run stopped short of applying, to "Needs prep" — neither of
    which was true of a single file on disk. Analysis now annotates the verdict
    through `busy` instead of replacing it.
    """
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["a", "b"],
        splits=splits(train=(8, 8), valid=(2, 2), unassigned=(0, 0)),
        metadata={"prep": {"state": state}},
    )
    assert result.state == "ready"
    assert result.trainable is True
    assert result.busy is True


@pytest.mark.parametrize("state", ["detecting", "planning"])
def test_analysing_an_empty_dataset_still_blocks(state: str):
    """Nothing in the splits means there is no prior verdict to preserve, so the
    run in flight is the only honest thing to report."""
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=[],
        splits=splits(),
        metadata={"prep": {"state": state}},
    )
    assert result.state == "blocked"
    assert result.next_action == "wait"


@pytest.mark.parametrize("state", ["draft", "planned", "failed"])
def test_staged_but_unapplied_prep_asks_to_run_prep(state: str):
    """Raw files staged, splits still empty: the agent has work left to do."""
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["a", "b"],
        splits=splits(unassigned=(30, 30)),
        metadata={"prep": {"state": state}},
    )
    assert result.state == "needs_prep"
    assert result.next_action == "run_prep"
    assert "Run Prep" in result.summary


@pytest.mark.parametrize("state", ["draft", "planned", "failed"])
def test_a_stalled_run_over_trainable_data_is_only_advisory(state: str):
    """Same states, but the splits are populated and annotated.

    A run that ended somewhere other than `ready` is not evidence about files
    that are already on disk and already load. It stays in the check list so the
    user can see it happened; it no longer decides the verdict.
    """
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["a", "b"],
        splits=splits(train=(8, 8), valid=(2, 2), unassigned=(0, 0)),
        metadata={"prep": {"state": state}},
    )
    assert result.state == "ready"
    assert result.trainable is True
    assert check(result, "prep_applied").passed is False
    assert check(result, "prep_applied").severity == "advisory"


def test_applied_prep_does_not_block():
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["a", "b"],
        splits=splits(train=(8, 8), valid=(2, 2), unassigned=(0, 0)),
        metadata={"prep": {"state": "ready"}},
    )
    assert result.trainable is True


# --- tolerance ----------------------------------------------------------------


def test_legacy_manifest_without_prep_metadata_still_computes():
    """Every dataset predating phase 21, plus reference and shared samples."""
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["a", "b"],
        splits=splits(train=(8, 8), valid=(2, 2), unassigned=(0, 0)),
        metadata={},
    )
    assert result.trainable is True


def test_no_metadata_at_all_is_accepted():
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["a", "b"],
        splits=splits(train=(8, 8), valid=(2, 2), unassigned=(0, 0)),
        metadata=None,
    )
    assert result.trainable is True


def test_missing_splits_are_treated_as_empty_not_an_error():
    result = readiness_for(
        task_type="classification", format="image_folder", labels=["a", "b"], splits={}
    )
    assert result.trainable is False


def test_undrained_inbox_warns_without_blocking():
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["a", "b"],
        splits=splits(train=(8, 8), valid=(2, 2), unassigned=(5, 5)),
    )
    assert result.trainable is True
    assert result.state == "ready"
    inbox = check(result, "inbox_drained")
    assert inbox.passed is False and inbox.severity == "advisory"
    # A ready dataset with a caveat says the caveat, not "Ready to train."
    assert "5 items" in result.summary


def test_every_check_carries_a_detail_when_it_fails():
    """The training page renders `summary`; an empty one would say nothing."""
    result = readiness_for(
        task_type="classification", format="image_folder", labels=[], splits=splits()
    )
    for entry in result.checks:
        if not entry.passed:
            assert entry.detail, f"check {entry.id} failed with no explanation"
    assert result.summary


# --- integration: readiness over a real dataset on disk -----------------------
#
# The unit tests above prove the rules. These prove the rules are actually wired
# into `DatasetService.summary`, on a dataset built through the real service.


def test_readiness_rides_on_a_real_dataset_summary(settings, storage):
    """A freshly created, empty dataset is not trainable and says so."""
    from app.schemas import DatasetCreate
    from app.services.datasets import DatasetService

    service = DatasetService(settings, storage)
    dataset = service.create_dataset(
        DatasetCreate(
            name="Chest scans",
            task_type="classification",
            format="image_folder",
            labels=["normal", "kista"],
        )
    )

    assert dataset.readiness is not None
    assert dataset.readiness.trainable is False
    assert dataset.readiness.next_action == "upload"
    assert dataset.readiness.summary
    # No prep agent has touched it, so there is no prep block on the manifest.
    assert dataset.prep is None


def test_readiness_follows_a_dataset_through_upload_and_split(settings, storage, tmp_path):
    """Upload then split, and readiness should move from blocked to ready."""
    import anyio
    from io import BytesIO

    from fastapi import UploadFile
    from PIL import Image

    from app.schemas import (
        DatasetCreate,
        DatasetProcessRequest,
        DatasetSplitConfig,
    )
    from app.services.datasets import DatasetService

    service = DatasetService(settings, storage)
    dataset = service.create_dataset(
        DatasetCreate(
            name="Chest scans",
            task_type="classification",
            format="image_folder",
            labels=["normal", "kista"],
        )
    )

    def upload(name: str) -> UploadFile:
        buffer = BytesIO()
        Image.new("RGB", (64, 64), "white").save(buffer, format="JPEG")
        buffer.seek(0)
        return UploadFile(file=buffer, filename=name)

    for index in range(12):
        anyio.run(
            service.upload_image,
            dataset.id,
            "unassigned",
            upload(f"scan-{index}.jpg"),
            index % 2,
        )

    # Items are in the inbox, nothing is split yet: the agent can fix this.
    staged = service.summary(dataset.id)
    assert staged.readiness.trainable is False
    assert staged.readiness.next_action == "run_prep"

    service.process_dataset(
        dataset.id,
        DatasetProcessRequest(
            split=DatasetSplitConfig(train=0.7, valid=0.2, test=0.1, stratify=False)
        ),
    )

    prepared = service.summary(dataset.id)
    assert prepared.readiness.trainable is True
    assert prepared.readiness.state == "ready"


def test_a_dataset_blocked_on_a_person_says_what_it_needs():
    """"Run Prep" is wrong when prep already ran and stopped on purpose."""
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=[],
        splits=splits(unassigned=(30, 0)),
        metadata={
            "prep": {
                "state": "planned",
                "plan": {"needs_input": "These images carry no labels."},
            }
        },
    )
    assert result.state == "needs_input"
    assert result.summary == "These images carry no labels."
    assert result.next_action == "label"


def test_an_unapplied_draft_still_says_run_prep():
    result = readiness_for(
        task_type="classification",
        format="image_folder",
        labels=["a", "b"],
        splits=splits(unassigned=(30, 30)),
        metadata={"prep": {"state": "draft"}},
    )
    assert result.next_action == "run_prep"
    assert "Run Prep" in result.summary
