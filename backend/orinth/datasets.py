"""Reading and writing the platform's datasets from a kernel.

`load()` always returns a **polars DataFrame**, whatever the modality. One
return type means `df.head()` works on every dataset in the platform; what
changes per task is the *columns*, not the type. polars rather than pandas
because polars is already a base dependency (phase 21 added it to read Parquet
during detection), so the core return type costs nothing new — and adding pandas
would put two dataframe idioms in one workspace. `df.to_pandas()` still works
where pandas happens to be installed, and `df.to_dicts()` is the escape that
needs nothing.

**Image rows carry a path, never pixels.** A 40,000-image dataset is a
40,000-row frame of a few megabytes; pixels come per item through `images()`.
That, plus the row cap and `records()` streaming, is the whole answer to "a
dataset that will not fit in memory".
"""

import builtins
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from orinth import _workspace
from orinth.errors import DatasetBusyError, DatasetTooLargeError
from orinth.types import DatasetRef, Readiness

Split = str
_SPLITS = ("unassigned", "train", "valid", "test")


def _service():
    """The platform's own `DatasetService`, over the kernel's settings.

    Imported lazily: this pulls numpy and PIL, which is fine in a kernel and
    would not be at module import in a notebook that only wants `orinth.runs`.
    """
    from app.services.datasets import DatasetService

    return DatasetService(_workspace.settings(), _workspace.storage())


def _ref(summary) -> DatasetRef:
    readiness = summary.readiness
    return DatasetRef(
        id=summary.id,
        project_id=summary.project_id,
        name=summary.name,
        task_type=summary.task_type,
        format=summary.format,
        labels=builtins.list(summary.labels),
        splits={name: split.item_count for name, split in summary.splits.items()},
        path=Path(summary.path),
        readiness=Readiness(
            state=readiness.state,
            trainable=readiness.trainable,
            summary=readiness.summary,
            busy=getattr(readiness, "busy", False),
            checks=[check.model_dump() for check in readiness.checks],
        ),
    )


# --- read ---------------------------------------------------------------------


def list(  # noqa: A001 - `orinth.datasets.list()` is the name a user reaches for
    *, project_id: str | None = None, task_type: str | None = None
) -> builtins.list[DatasetRef]:
    """Every dataset in the workspace, optionally narrowed.

    Shadowing the builtin is deliberate: `orinth.datasets.list()` is the name a
    user reaches for, and the builtin is still `builtins.list` one attribute
    away. The same trade the `logging` and `csv` modules make.
    """
    service = _service()
    summaries = service.list_datasets(project_id or _workspace.default_project_id())
    refs = [_ref(summary) for summary in summaries]
    if task_type:
        refs = [ref for ref in refs if ref.task_type == task_type]
    return refs


def get(dataset_id: str) -> DatasetRef:
    return _ref(_service().summary(dataset_id))


def readiness(dataset_id: str) -> Readiness:
    return get(dataset_id).readiness


def _guard_busy(dataset_id: str) -> None:
    """Refuse to read a dataset apply is rewriting.

    Only `applying` drops and repopulates the split directories. Detect and plan
    read `_staging/` and leave everything else alone, which is the same
    distinction `readiness._PREP_REWRITING` draws — a notebook that refused
    during analysis would be wrong in the same way the readiness badge was.
    """
    service = _service()
    location = service._location(dataset_id)
    prep = (location.metadata or {}).get("prep") or {}
    if isinstance(prep, dict) and prep.get("state") == "applying":
        raise DatasetBusyError(dataset_id)


def _splits_for(value: "Split | Sequence[Split]") -> builtins.list[str]:
    if isinstance(value, str):
        return [value]
    return [str(entry) for entry in value]


def load(
    dataset_id: str,
    split: "Split | Sequence[Split]" = "train",
    *,
    limit: int | None = None,
    columns: Sequence[str] | None = None,
):
    """One split (or several) as a polars DataFrame.

    Raises `DatasetTooLargeError` past `NOTEBOOK_LOAD_MAX_ROWS` rather than
    truncating — a frame that looks complete and is not is the worse failure.
    """
    import polars

    _guard_busy(dataset_id)
    service = _service()
    summary = service.summary(dataset_id)
    wanted = _splits_for(split)

    total = sum(
        summary.splits[name].item_count for name in wanted if name in summary.splits
    )
    cap = _workspace.settings().notebook_load_max_rows
    if limit is None and total > cap:
        raise DatasetTooLargeError(total, cap)

    rows: builtins.list[dict[str, Any]] = []
    remaining = limit
    for name in wanted:
        for row in _rows_for_split(service, summary, name):
            rows.append(row)
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break
        if remaining is not None and remaining <= 0:
            break

    frame = polars.DataFrame(rows, infer_schema_length=None) if rows else polars.DataFrame()
    if columns and rows:
        keep = [name for name in columns if name in frame.columns]
        frame = frame.select(keep)
    return frame


def _rows_for_split(service, summary, split: str) -> Iterator[dict[str, Any]]:
    """Project one split into flat records, per the task's column contract."""
    from app.services.datasets.constants import IMAGE_TASK_TYPES, LLM_TASK_TYPES

    task = summary.task_type
    if task in LLM_TASK_TYPES:
        yield from _record_rows(service, summary, split)
        return
    if task in IMAGE_TASK_TYPES:
        yield from _image_rows(service, summary, split)
        return
    yield from _text_rows(service, summary, split)


def _image_rows(service, summary, split: str) -> Iterator[dict[str, Any]]:
    location = service._location(summary.id)
    for path in service._item_paths(location, split):
        # A directory listing is a snapshot; an item can vanish between listing
        # and opening it. Phase 21 hit exactly this and fixed it the same way.
        if not path.is_file():
            continue
        detail = service._item_from_path_if_present(location, split, path, include_annotations=True)
        if detail is None:
            continue
        yield {
            "item_id": detail.id,
            "split": split,
            "path": str(path.resolve()),
            "width": detail.width,
            "height": detail.height,
            "label": detail.label,
            "annotations": [
                {
                    "label": annotation.class_name,
                    "label_id": annotation.class_id,
                    "bbox": annotation.bbox,
                    "polygon": annotation.polygon,
                }
                for annotation in detail.annotations
            ],
        }


def _text_rows(service, summary, split: str) -> Iterator[dict[str, Any]]:
    location = service._location(summary.id)
    task = summary.task_type
    for path in service._item_paths(location, split):
        if not path.is_file():
            continue
        detail = service._item_from_path_if_present(location, split, path, include_annotations=True)
        if detail is None:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        row: dict[str, Any] = {"item_id": detail.id, "split": split, "text": text}
        target = detail.label or (detail.classes[0] if detail.classes else None)
        if task == "text_classification":
            row["label"] = target
            row["label_id"] = detail.class_id
        elif task == "summarization":
            row["summary"] = target
        elif task == "question_answering":
            # The annotation carries the answer; the question rides its own
            # field on the annotation where the writer put one.
            row["question"] = detail.annotations[0].text if detail.annotations else None
            row["answer"] = target
        yield row


def _record_rows(service, summary, split: str) -> Iterator[dict[str, Any]]:
    location = service._location(summary.id)
    for path in service._item_paths(location, split):
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        row: dict[str, Any] = {"item_id": path.name, "split": split}
        if summary.format == "chat_jsonl":
            row["messages"] = record.get("messages") or record.get("conversations") or []
        else:
            row["instruction"] = record.get("instruction")
            row["input"] = record.get("input")
            row["output"] = record.get("output")
        yield row


def records(dataset_id: str, split: Split = "train") -> Iterator[dict]:
    """Stream a split's records, one at a time, holding no file open.

    This is the shape any `datasets` / `trl` pipeline wants, and it is the
    answer to a split too large for `load()`.
    """
    _guard_busy(dataset_id)
    service = _service()
    summary = service.summary(dataset_id)
    yield from _rows_for_split(service, summary, split)


def paths(dataset_id: str, split: Split = "train") -> builtins.list[Path]:
    _guard_busy(dataset_id)
    service = _service()
    location = service._location(dataset_id)
    return [path.resolve() for path in service._item_paths(location, split) if path.is_file()]


def images(dataset_id: str, split: Split = "train") -> Iterator[tuple]:
    """`(item_id, PIL.Image)` per item, opened lazily.

    Pillow is imported here rather than at module scope so a notebook that only
    reads text never pays for it.
    """
    from PIL import Image

    for path in paths(dataset_id, split):
        with Image.open(path) as handle:
            # `load()` forces the decode before the context closes, so the
            # caller gets a usable image rather than one backed by a closed file.
            handle.load()
            yield path.name, handle.copy()


# --- write --------------------------------------------------------------------


def register(
    data,
    *,
    name: str,
    task_type: str,
    project_id: str | None = None,
    format: str | None = None,  # noqa: A002
    labels: Sequence[str] | None = None,
    columns: Mapping[str, str] | None = None,
    split: Mapping[str, float] | None = None,
    seed: int = 42,
    stratify: bool = True,
    wait: bool = True,
) -> DatasetRef:
    """Turn a dataframe or a list of dicts into a real Orinth dataset.

    Writes nothing itself. The rows are serialized to newline-delimited JSON,
    posted to `/api/datasets/ingest`, and then `/prep/apply` is handed a
    fully-specified plan. Routing through *apply* rather than through a new
    writer is the point: apply is the code phase 21 already validated, so a
    dataset built this way satisfies the readiness contract because it is built
    by the function readiness was written against. Going through detect+plan
    instead would ask the agent to re-derive what the caller just stated, which
    is how a mapping gets guessed wrong.

    Always creates a **new** dataset. Registering into an existing id would mean
    rebuilding a dataset out from under whatever is reading it, which is the
    failure phase 21 lived through and a notebook is the easiest place to
    trigger by accident.
    """
    from orinth import _register

    return _register.register(
        data,
        name=name,
        task_type=task_type,
        project_id=project_id or _workspace.default_project_id(),
        format=format,
        labels=labels,
        columns=columns,
        split=split,
        seed=seed,
        stratify=stratify,
        wait=wait,
    )
