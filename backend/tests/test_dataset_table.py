"""The grid projection: one table shape across every modality.

These test `table.py` as the pure function it is — summaries in, columns and
rows out — rather than through the route, because what is worth pinning down is
which columns a task gets and which of them claim to be editable. The route
adds only paging, which `list_items_page` already owns and already tests.
"""

import json
from pathlib import Path

from app.core.config import Settings
from app.core.storage import Storage
from app.schemas import DatasetCreate, DatasetItemSummary
from app.services.datasets import DatasetService, table


def item(**overrides) -> DatasetItemSummary:
    base = {
        "id": "a.png",
        "dataset_id": "ds-1",
        "split": "train",
        "filename": "a.png",
        "annotation_count": 0,
        "classes": [],
    }
    base.update(overrides)
    return DatasetItemSummary(**base)


def keys(task_type: str, format: str = "image_folder") -> list[str]:
    return [column.key for column in table.columns_for(task_type=task_type, format=format)]


def editable(task_type: str, format: str = "image_folder") -> set[str]:
    return {
        column.key
        for column in table.columns_for(task_type=task_type, format=format)
        if column.editable
    }


# --- columns ------------------------------------------------------------------


def test_classification_gets_an_editable_label_column():
    assert keys("classification") == ["preview", "split", "filename", "label"]
    # Split and label are the only two cells with a write route behind them.
    assert editable("classification") == {"split", "label"}


def test_detection_reports_regions_rather_than_one_label():
    """A detection image carries many boxes, so a single `label` cell would be a
    lie about which one it is. The count plus the class names is the honest cell,
    and neither is editable here — that is the annotation editor's job."""
    assert keys("object_detection", "yolo") == ["preview", "split", "filename", "classes", "regions"]
    assert editable("object_detection", "yolo") == {"split"}


def test_text_classification_gets_text_and_an_editable_label():
    assert keys("text_classification", "text_folder") == ["split", "text", "label"]
    assert editable("text_classification", "text_folder") == {"split", "label"}


def test_summarization_gets_a_read_only_target():
    """Its target is a generated summary, not a member of a closed label set, so
    a picker would be wrong and free text here would bypass validation."""
    assert keys("summarization", "text_folder") == ["split", "text", "target"]
    assert editable("summarization", "text_folder") == {"split"}


def test_instruction_and_chat_records_differ_only_in_column_labels():
    instruction = table.columns_for(task_type="llm_finetune", format="instruction_jsonl")
    chat = table.columns_for(task_type="llm_finetune", format="chat_jsonl")
    assert [column.key for column in instruction] == [column.key for column in chat]
    assert [column.label for column in instruction] != [column.label for column in chat]


# --- rows ---------------------------------------------------------------------


def test_image_row_carries_the_thumbnail_url_and_the_label():
    row = table.row_for(
        item(image_url="/api/datasets/ds-1/items/train/a.png/image", label="kista"),
        task_type="classification",
        format="image_folder",
    )
    assert row.cells["preview"].endswith("/image")
    assert row.cells["label"] == "kista"
    assert row.split == "train"


def test_detection_row_joins_class_names_and_counts_regions():
    row = table.row_for(
        item(classes=["granuloma", "kista"], annotation_count=3),
        task_type="object_detection",
        format="yolo",
    )
    assert row.cells["classes"] == "granuloma, kista"
    assert row.cells["regions"] == 3


def test_record_row_uses_the_previews_the_item_read_already_produced():
    """`text_preview` / `output_preview` exist so the records table never opens a
    record file. The grid must ride the same fields, not re-read anything."""
    row = table.row_for(
        item(
            id="r1.json",
            media_type="record",
            text_preview="Summarize this",
            output_preview="A summary",
            token_estimate=12,
        ),
        task_type="llm_finetune",
        format="instruction_jsonl",
    )
    assert row.cells == {"prompt": "Summarize this", "response": "A summary", "tokens": 12}


def test_summarization_row_falls_back_to_the_class_list_for_its_target():
    """`label` is only populated for classification, so reading it alone would
    leave the target column blank on every summarization dataset."""
    row = table.row_for(
        item(id="t1.txt", media_type="text", text_preview="Long text", classes=["a summary"]),
        task_type="summarization",
        format="text_folder",
    )
    assert row.cells["target"] == "a summary"


def test_page_carries_paging_through_unchanged():
    page = table.page_for(
        [item(), item(id="b.png", filename="b.png")],
        task_type="classification",
        format="image_folder",
        total=120,
        limit=50,
        offset=50,
    )
    assert (page.total, page.limit, page.offset) == (120, 50, 50)
    assert len(page.rows) == 2


# --- through the service ------------------------------------------------------


def test_the_grid_reads_the_same_task_type_the_catalog_shows(
    tmp_path: Path, settings: Settings
):
    """A manifest on disk outlives the schema.

    `tabular`/`table` is a real legacy pair still present in workspaces, and
    `DatasetSummary` maps it to `text_classification` through `TASK_TYPE_ALIASES`.
    Reading `location.task_type` raw gave the grid a *different* task type from
    the badge on the card that opened it — a `target` column where the rest of
    the app said `label`. Both sides normalize.
    """
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)

    dataset = service.create_dataset(
        DatasetCreate(name="Legacy", task_type="text_classification", format="text_folder")
    )
    manifest_path = storage.datasets / dataset.id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_type"] = "tabular"
    manifest["format"] = "table"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = service.summary(dataset.id)
    page = service.table_page(dataset.id, "all")

    assert summary.task_type == "text_classification"
    assert [column.key for column in page.columns] == ["split", "text", "label"]


# --- record columns come from the records --------------------------------------


def test_a_field_the_derived_columns_do_not_show_gets_its_own_column():
    """`input` was invisible on every instruction dataset that used it: the grid
    had exactly two record columns whatever the records held."""
    columns = table.columns_for(
        task_type="llm_finetune",
        format="instruction_jsonl",
        record_fields=["instruction", "input", "output"],
    )
    assert [column.key for column in columns] == [
        "split",
        "prompt",
        "response",
        "field:input",
        "tokens",
    ]
    assert [column.label for column in columns][3] == "Input"


def test_an_unknown_field_keeps_the_name_the_record_gives_it():
    columns = table.columns_for(
        task_type="llm_finetune", format="instruction_jsonl", record_fields=["sentence1"]
    )
    assert [column.label for column in columns if column.key == "field:sentence1"] == ["sentence1"]


def test_a_chat_record_keeps_its_readable_turns_rather_than_a_json_blob():
    """`messages` is already shown as first-user / last-assistant turns. Adding a
    column for the raw list would replace two readable cells with one blob."""
    columns = table.columns_for(
        task_type="llm_finetune", format="chat_jsonl", record_fields=["messages"]
    )
    assert [column.key for column in columns] == ["split", "prompt", "response", "tokens"]


def test_the_page_takes_the_union_of_the_fields_its_records_carry():
    """A record dataset is not required to be uniform, so reading only the first
    row's keys would hide a column that appears on row two."""
    page = table.page_for(
        [item(id="r1.json", media_type="record"), item(id="r2.json", media_type="record")],
        task_type="llm_finetune",
        format="instruction_jsonl",
        total=2,
        limit=50,
        offset=0,
        records={
            "r1.json": {"instruction": "a", "output": "b"},
            "r2.json": {"instruction": "a", "output": "b", "source": "hub"},
        },
    )
    assert [column.key for column in page.columns] == [
        "split",
        "prompt",
        "response",
        "field:source",
        "tokens",
    ]
    # Only the fields that got a column are sent; emitting the rest would double
    # a fifty-row page over the wire to render nothing.
    assert "field:instruction" not in page.rows[0].cells
    assert page.rows[1].cells["field:source"] == "hub"
