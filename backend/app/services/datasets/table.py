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


def columns_for(*, task_type: str, format: str) -> list[DatasetTableColumn]:
    """The column set for one dataset, decided by task rather than by format.

    Format decides how a record is *stored*; task decides what the columns
    *mean*. `instruction_jsonl` and `chat_jsonl` are both `llm_finetune` and are
    distinguished below, but only because their record shapes differ — not
    because the format field is what is being switched on.
    """
    split = DatasetTableColumn(key="split", label="Split", kind="split", editable=True)

    if task_type in LLM_TASK_TYPES:
        if format == "chat_jsonl":
            return [
                split,
                DatasetTableColumn(key="prompt", label="First user turn", kind="text"),
                DatasetTableColumn(key="response", label="Last assistant turn", kind="text"),
                DatasetTableColumn(key="tokens", label="Tokens", kind="number"),
            ]
        return [
            split,
            DatasetTableColumn(key="prompt", label="Instruction", kind="text"),
            DatasetTableColumn(key="response", label="Output", kind="text"),
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


def row_for(
    item: DatasetItemSummary, *, task_type: str, format: str
) -> DatasetTableRow:
    """Project one item summary into the cells its columns expect."""
    cells: dict[str, Any] = {}

    if task_type in LLM_TASK_TYPES:
        cells["prompt"] = item.text_preview or ""
        cells["response"] = item.output_preview or ""
        cells["tokens"] = item.token_estimate
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

    del format
    return DatasetTableRow(id=item.id, split=item.split, cells=cells)


def page_for(
    items: list[DatasetItemSummary],
    *,
    task_type: str,
    format: str,
    total: int,
    limit: int,
    offset: int,
) -> DatasetTablePage:
    return DatasetTablePage(
        columns=columns_for(task_type=task_type, format=format),
        rows=[row_for(item, task_type=task_type, format=format) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )
