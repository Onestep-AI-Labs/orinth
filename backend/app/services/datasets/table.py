"""One grid shape for every modality, projected from the paged item read.

The studio grew a different browser per task type — a thumbnail wall for images,
a preview list for text, a bespoke table for LLM records — and none of them let
you see a dataset the way Kaggle or the Hugging Face viewer does: rows, columns,
sortable, editable in place. Three components meant three places to fix a bug and
three answers to "what is actually in this data".

**Why this is a projection and not a second storage format.** The obvious
alternative is to materialize a table — a Parquet or SQLite index per split,
rebuilt on write. It would buy fast server-side sort and filter across the whole
dataset. It was not built, for two reasons:

1. The training runners read the on-disk layout directly. A materialized index is
   a second source of truth that can disagree with it, and the failure mode is
   silent: the grid shows a row the trainer skipped.
2. Paging does not need it. ``list_items_page`` already decides the page from
   the file paths alone and reads only that window — 50 items, not 15,000 — so a
   page costs the same here as it does in the image browser today.

What a materialized index *would* buy is whole-dataset sort and filter, and that
is exactly what is deferred: filtering already forces a full scan in
``list_items_page`` and is bounded by the same cost here. When a dataset is large
enough for that to hurt, the index belongs beside the splits as a derived
artifact with an explicit rebuild — not smuggled in under a browse endpoint.

So this module is pure: summaries in, columns and rows out. It reads no files of
its own.
"""

import json
from typing import Any

from app.schemas import (
    DatasetItemSummary,
    DatasetTableColumn,
    DatasetTablePage,
    DatasetTableRow,
)
from app.services.datasets.constants import IMAGE_TASK_TYPES, LLM_TASK_TYPES

#: Tasks whose supervision is a single class name, so the label column is a
#: closed set and can be edited with a picker rather than free text.
_SINGLE_LABEL_TASKS = {"classification", "text_classification"}


def columns_for(
    *, task_type: str, format: str, record_fields: list[str] | None = None
) -> list[DatasetTableColumn]:
    """The column set for one dataset, decided by task rather than by format.

    Format decides how a record is *stored*; task decides what the columns
    *mean*. `instruction_jsonl` and `chat_jsonl` are both `llm_finetune` and are
    distinguished below, but only because their record shapes differ — not
    because the format field is what is being switched on.

    `record_fields` is the exception, and it exists because the rule above was
    not enough. A record dataset showed exactly two columns — `Instruction` and
    `Output` — whatever the records held, so any other field the records carried
    was simply not in the table: `input` was invisible on every instruction
    dataset that used it, and an imported column that survived normalization had
    nowhere to appear. The named columns stay, because "first user turn" is a
    better cell than a JSON blob of the whole conversation; what is added is a
    column for every field those two do not already show.
    """
    split = DatasetTableColumn(key="split", label="Split", kind="split", editable=True)

    if task_type in LLM_TASK_TYPES:
        chat = format == "chat_jsonl"
        derived = (
            [
                DatasetTableColumn(key="prompt", label="First user turn", kind="text"),
                DatasetTableColumn(key="response", label="Last assistant turn", kind="text"),
            ]
            if chat
            else [
                DatasetTableColumn(key="prompt", label="Instruction", kind="text"),
                DatasetTableColumn(key="response", label="Output", kind="text"),
            ]
        )
        covered = _COVERED_FIELDS["chat_jsonl" if chat else "instruction_jsonl"]
        extra = [
            _record_column(field) for field in (record_fields or []) if field not in covered
        ]
        return [
            split,
            *derived,
            *extra,
            DatasetTableColumn(key="tokens", label="Tokens", kind="number"),
        ]

    if task_type in IMAGE_TASK_TYPES:
        columns = [
            DatasetTableColumn(key="preview", label="", kind="image"),
            split,
            DatasetTableColumn(key="filename", label="File", kind="text"),
        ]
        if task_type in _SINGLE_LABEL_TASKS:
            columns.append(
                DatasetTableColumn(key="label", label="Label", kind="label", editable=True)
            )
        else:
            # Detection and segmentation carry many boxes per image, so the
            # honest cell is a count plus the class names — not one label
            # pretending to be the answer. Editing them is the annotation
            # editor's job, which is why this is not editable here.
            columns.append(DatasetTableColumn(key="classes", label="Classes", kind="text"))
            columns.append(DatasetTableColumn(key="regions", label="Regions", kind="number"))
        return columns

    # Text tasks: summarization and QA carry their target in the annotation, and
    # classification carries a class. `target` covers all three so the grid does
    # not need a fourth branch for a column that is one string either way.
    columns = [
        split,
        DatasetTableColumn(key="text", label="Text", kind="text"),
    ]
    if task_type in _SINGLE_LABEL_TASKS:
        columns.append(DatasetTableColumn(key="label", label="Label", kind="label", editable=True))
    else:
        columns.append(DatasetTableColumn(key="target", label="Target", kind="text"))
    return columns


#: Fields the derived `prompt`/`response` columns already show, per format.
#: Anything outside these gets a column of its own, which is how `input` and an
#: import's leftover columns become visible instead of being dropped.
_COVERED_FIELDS = {
    "chat_jsonl": {"messages", "conversations"},
    "instruction_jsonl": {"instruction", "output"},
}

#: Record keys that get a friendlier heading than their raw name. Everything
#: else is shown as the record spells it — an imported column called
#: `sentence1` is called `sentence1` here, because that is what it is called in
#: the file the user will train on.
_FIELD_LABELS = {
    "instruction": "Instruction",
    "input": "Input",
    "output": "Output",
    "messages": "Messages",
    "conversations": "Conversations",
    "question": "Question",
    "context": "Context",
    "answer": "Answer",
    "text": "Text",
    "label": "Label",
}


def _record_column(field: str) -> DatasetTableColumn:
    return DatasetTableColumn(
        key=f"field:{field}",
        label=_FIELD_LABELS.get(field, field),
        # Structured fields (`messages`, a nested answer struct) render as JSON
        # rather than as `[object Object]`; the grid's `json` cell is the one
        # that says so honestly.
        kind="json" if field in {"messages", "conversations"} else "text",
    )


def _cell_text(value: Any) -> Any:
    """One record field, flattened to something a single-line cell can show."""
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool):
        return value
    rendered = json.dumps(value, ensure_ascii=False)
    return rendered if len(rendered) <= 400 else rendered[:400] + "…"


def row_for(
    item: DatasetItemSummary,
    *,
    task_type: str,
    format: str,
    record: dict[str, Any] | None = None,
) -> DatasetTableRow:
    """Project one item summary into the cells its columns expect."""
    cells: dict[str, Any] = {}

    if task_type in LLM_TASK_TYPES:
        cells["prompt"] = item.text_preview or ""
        cells["response"] = item.output_preview or ""
        cells["tokens"] = item.token_estimate
        covered = _COVERED_FIELDS[
            "chat_jsonl" if format == "chat_jsonl" else "instruction_jsonl"
        ]
        # Only the fields that got a column. Emitting the rest would double a
        # 50-row page over the wire to render nothing.
        for key, value in (record or {}).items():
            if key not in covered:
                cells[f"field:{key}"] = _cell_text(value)
    elif task_type in IMAGE_TASK_TYPES:
        cells["preview"] = item.image_url
        cells["filename"] = item.filename
        if task_type in _SINGLE_LABEL_TASKS:
            cells["label"] = item.label
        else:
            cells["classes"] = ", ".join(item.classes)
            cells["regions"] = item.annotation_count
    else:
        cells["text"] = item.text_preview or ""
        if task_type in _SINGLE_LABEL_TASKS:
            cells["label"] = item.label
        else:
            # Summarization and QA store the target as the annotation's class
            # name; `label` is only populated for classification, so read the
            # class list instead of leaving the column blank.
            cells["target"] = item.label or ", ".join(item.classes)

    return DatasetTableRow(id=item.id, split=item.split, cells=cells)


def page_for(
    items: list[DatasetItemSummary],
    *,
    task_type: str,
    format: str,
    total: int,
    limit: int,
    offset: int,
    records: dict[str, dict[str, Any]] | None = None,
) -> DatasetTablePage:
    """Columns and rows for one page.

    `records` maps item id to the record on disk, supplied by the caller for
    record datasets. The column set is the union of the keys on this page, in
    first-seen order: a record dataset is not required to be uniform, and taking
    only the first row's keys would hide a column that appears on row two.
    """
    fields: list[str] = []
    for item in items:
        for key in (records or {}).get(item.id, {}):
            if key not in fields:
                fields.append(key)

    return DatasetTablePage(
        columns=columns_for(task_type=task_type, format=format, record_fields=fields),
        rows=[
            row_for(
                item,
                task_type=task_type,
                format=format,
                record=(records or {}).get(item.id),
            )
            for item in items
        ],
        total=total,
        limit=limit,
        offset=offset,
    )
