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

from orinth import _http, _workspace
from orinth.errors import DatasetBusyError, DatasetTooLargeError, OrinthError
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

    **`limit=` is a head, not a sample.** Items come back in directory order,
    which for an image dataset means class order, so `limit=90` on a six-class
    set is ninety images of the first class. Fine for "what do the columns look
    like"; wrong for anything that reads the label. Take the frame and sample it
    yourself when the distribution matters.
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


def detect(dataset_id: str) -> dict:
    """What the deterministic scan concluded about a staged upload.

    The read half of `upload(..., auto_apply=False)`: modality, task type,
    format, candidate labels, and the signals behind each. Worth looking at
    before applying anything, because a wrong `task_type` here becomes a wrong
    dataset that has to be rebuilt.
    """
    return _http.get(f"/api/datasets/{dataset_id}/prep/detect") or {}


def upload(
    path: "str | Path",
    *,
    name: str | None = None,
    project_id: str | None = None,
    include: Sequence[str] | None = None,
    auto_apply: bool = True,
    wait: bool = True,
) -> DatasetRef:
    """Your own files — a folder, a single table, an archive — as a dataset.

    `register()` takes rows you already have in memory. This takes what is on
    disk, which is how data actually arrives: a folder of images with one
    subdirectory per class, a CSV someone exported, a YOLO export with its
    `data.yaml`.

    Nothing about the files is declared here, and that is deliberate. The upload
    is staged and handed to the **same prep agent a browser upload goes to**,
    which reads the directory layout and the file contents to decide the task,
    the format and the labels. A second detector living in the SDK would
    disagree with the first one the week after it was written.

    The directory structure is preserved (`relative_paths`), because for an
    image dataset the structure *is* the labelling.

    `auto_apply=False` stages and detects without committing, for when you want
    to look at `detect()` first.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise OrinthError(f"No such path: {root}")

    files = _files_under(root, include)
    if not files:
        raise OrinthError(f"Nothing to upload under {root} — no readable files matched.")

    project = project_id or _workspace.default_project_id()
    dataset_id = _stage(files, root, name=name or root.stem, project_id=project)

    if not auto_apply:
        return get(dataset_id)

    _http.post(f"/api/datasets/{dataset_id}/prep", json={"auto_apply": True})
    if wait:
        from orinth import _register

        _register._wait(dataset_id)
    return get(dataset_id)


#: Files a dataset never wants, and that a `**/*` walk always finds.
_SKIP_NAMES = {".DS_Store", "Thumbs.db", ".gitkeep"}
#: One request per batch. A folder of ten thousand images in a single multipart
#: body is a request nothing survives — not the server's memory, not the proxy's
#: body limit, and not the user's patience with no progress at all.
UPLOAD_BATCH = 200


def _files_under(root: "Path", include: Sequence[str] | None) -> builtins.list["Path"]:
    if root.is_file():
        return [root]
    patterns = builtins.list(include or ["**/*"])
    found: dict[str, Path] = {}
    for pattern in patterns:
        for candidate in sorted(root.glob(pattern)):
            if not candidate.is_file():
                continue
            if candidate.name in _SKIP_NAMES or candidate.name.startswith("._"):
                continue
            found[str(candidate)] = candidate
    return builtins.list(found.values())


def _stage(files: Sequence["Path"], root: "Path", *, name: str, project_id: str) -> str:
    """Create the draft with the first batch, then add the rest to it."""
    dataset_id = ""
    for start in range(0, len(files), UPLOAD_BATCH):
        batch = files[start : start + UPLOAD_BATCH]
        payload = []
        # `relative_paths` is a repeated form field, which httpx expresses as a
        # list value rather than repeated tuples. Sent in the same order as the
        # files: the layout is what detection reads, and without it every upload
        # looks like a flat pile.
        data: dict[str, Any] = {"relative_paths": []}
        for item in batch:
            relative = item.name if root.is_file() else str(item.relative_to(root))
            payload.append(("files", (item.name, item.read_bytes())))
            data["relative_paths"].append(relative)
        if not dataset_id:
            data["project_id"] = project_id
            data["name"] = name
            created = _http.post(
                "/api/datasets/ingest", files=payload, data=data, timeout=_http.UPLOAD_TIMEOUT
            )
            dataset_id = (created or {}).get("id", "")
            if not dataset_id:
                raise OrinthError("The API accepted the upload but returned no dataset id.")
        else:
            _http.post(
                f"/api/datasets/{dataset_id}/ingest",
                files=payload,
                data=data,
                timeout=_http.UPLOAD_TIMEOUT,
            )
    return dataset_id


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
