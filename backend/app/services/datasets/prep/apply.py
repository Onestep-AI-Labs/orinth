"""Turning a plan into a dataset the training runners can read.

Deliberately thin. Every write goes through a method `DatasetService` already
has — `_update_manifest`, `_create_layout`, `_write_annotation_json`,
`_write_yolo_label`, `_write_text_item`, `_write_record`, and above all
`process_dataset`, the existing splitter. Nothing here re-implements dataset
mechanics, because a second way to write a dataset is a second way to write it
subtly differently.

The one genuinely new thing is moving staged files into the layout the manifest
now declares. That is modality-specific and is the only place this module
touches disk directly.

Reversibility is what makes auto-apply defensible rather than reckless: the
pre-apply manifest is snapshotted to `prep_undo.json` before anything moves, and
`_staging/` is kept afterwards so re-planning needs no re-upload.
"""

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.schemas import (
    Box,
    DatasetAnnotation,
    DatasetPrepPlan,
    DatasetProcessRequest,
    DatasetSummary,
)
from app.services.datasets.constants import (
    IMAGE_SUFFIXES,
    IMAGE_TASK_TYPES,
    LLM_TASK_TYPES,
    NLP_TASK_TYPES,
    SPLITS,
    UNDO_FILENAME,
)
from app.services.datasets.prep.detect import TABLE_SUFFIXES, _read_rows
from app.services.datasets.prep.staging import (
    derived_files,
    derived_root,
    staged_files,
    staging_root,
)

if TYPE_CHECKING:  # pragma: no cover
    from app.services.datasets.service import DatasetService

#: Rows ingested from any one tabular or record source.
#:
#: The platform stores one file per item — plus an annotation sidecar and a label
#: file for text — so a 41,000-row CSV is 123,000 file creations and takes long
#: enough that an upload looks hung. The hub importer caps at 5,000 for the same
#: reason; this is more generous because the file is already local, but it is
#: still a bound. Hitting it is reported, never silent.
MAX_INGEST_ROWS = 20_000


class PrepApplyError(RuntimeError):
    """The plan cannot be applied as written."""


def apply_plan(
    service: "DatasetService", dataset_id: str, plan: DatasetPrepPlan
) -> DatasetSummary:
    """Rewrite the dataset to match `plan`, then split it.

    Raises `PrepApplyError` when the plan is incomplete — an unlabelled image
    folder, say, where only a human can supply the classes.
    """
    if plan.needs_input:
        raise PrepApplyError(plan.needs_input)
    if not plan.task_type or not plan.format:
        raise PrepApplyError(
            "Orinth could not work out what this data is. Choose a task type manually."
        )

    location = service._location(dataset_id)
    root = location.root

    _snapshot(root)

    task = str(plan.task_type)
    fmt = str(plan.format)
    labels = list(plan.labels)

    # Every item under `<split>/` was written by a previous apply from the same
    # staging directory, so it is rebuilt rather than added to. Without this a
    # second run — after an Undo, or after editing the plan — imports the upload
    # on top of itself and silently doubles the dataset. `_staging/` is what
    # makes that safe: a dataset the agent staged is one the agent can rebuild
    # from source, which is not true of a hand-built dataset and is why this is
    # guarded rather than unconditional.
    _reset_items(root)

    # Layout first: the media/annotation directories must exist before anything
    # is written into them, and `_create_layout` also emits `data.yaml` for YOLO.
    service._create_layout(root, fmt, task, labels)
    service._update_manifest(location, task_type=task, format=fmt, labels=labels)
    location = service._location(dataset_id)

    moved = _move_staged(service, location, plan)

    # Preprocessing rides the manifest; `process_dataset` persists it alongside
    # the split config in one write.
    service.process_dataset(
        dataset_id,
        DatasetProcessRequest(preprocess=plan.preprocess, split=plan.split),
    )

    location = service._location(dataset_id)
    metadata = dict(location.metadata or {})
    metadata["prep"] = {
        "state": "ready",
        "applied_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        "staged_files": len(staged_files(root)),
        "moved": moved,
        "plan": plan.model_dump(mode="json"),
    }
    service._update_manifest(location, metadata=metadata)
    return service.summary(dataset_id)


def undo(service: "DatasetService", dataset_id: str) -> DatasetSummary:
    """Restore the manifest as it stood before apply.

    A restore, not an inverse replay: reversing a split by recomputing which item
    went where would depend on the same RNG the split used, and would silently
    diverge the moment either changed.
    """
    location = service._location(dataset_id)
    path = location.root / UNDO_FILENAME
    if not path.is_file():
        raise PrepApplyError("There is nothing to undo for this dataset.")

    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise PrepApplyError("The saved pre-prep state could not be read.") from error

    manifest_path = location.root / "manifest.json"
    manifest_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    path.unlink(missing_ok=True)
    # The transform output belongs to the plan being undone. Leaving it would
    # make a later apply — one whose plan has no transform — silently read the
    # reshaped rows instead of the raw upload it was planned against.
    shutil.rmtree(derived_root(location.root), ignore_errors=True)
    return service.summary(dataset_id)


def _reset_items(root: Path) -> None:
    """Drop the split directories, but only for an agent-staged dataset."""
    if not staging_root(root).is_dir():
        return
    for split in SPLITS:
        shutil.rmtree(root / split, ignore_errors=True)


def _snapshot(root: Path) -> None:
    """Copy the manifest aside before the first write."""
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return
    target = root / UNDO_FILENAME
    # Only the *first* apply snapshots. A second apply over an agent-prepared
    # dataset must still undo to the state before the agent ever ran, not to the
    # previous agent run.
    if target.exists():
        return
    shutil.copy2(manifest, target)


# --- moving staged files into the layout --------------------------------------


def _move_staged(
    service: "DatasetService", location: Any, plan: DatasetPrepPlan
) -> dict[str, int]:
    """Place raw files where the runners expect them.

    Everything lands in `unassigned/`; `process_dataset` distributes from there.
    The exception is a dataset that already carries its own splits (a YOLO or
    COCO export), which is moved split-for-split so an author's intended
    train/test boundary is not reshuffled.
    """
    root = location.root
    staging = staging_root(root)
    if not staging.is_dir():
        return {}

    task = str(plan.task_type)
    if task in IMAGE_TASK_TYPES:
        return _move_images(service, location, staging, plan)
    if task in LLM_TASK_TYPES:
        return _move_records(service, location, staging, plan)
    if task in NLP_TASK_TYPES:
        return _move_text(service, location, staging, plan)
    return {}


#: Suffixes whose rows a transform replaces. Images and `.txt` files pass
#: through untouched even when a transform ran, because a transform reshapes
#: rows and has nothing to say about them.
_ROW_SUFFIXES = {".jsonl", ".json"} | TABLE_SUFFIXES


def _sources(root: Path, staging: Path) -> list[tuple[Path, Path]]:
    """(file, path-relative-to-staging) for everything apply should read.

    When `prep/transform.py` ran, its output in `_derived/` *replaces* the raw
    row files rather than joining them — importing both would ingest every row
    twice, once reshaped and once in the shape that could not be mapped. The
    raw files stay on disk regardless; they are the thing Undo restores to.
    """
    derived = derived_files(root)
    sources = [
        (path, path.relative_to(staging))
        for path in staged_files(root)
        if not (derived and path.suffix.lower() in _ROW_SUFFIXES)
    ]
    sources.extend((path, Path(path.name)) for path in derived)
    return sources


def _existing_splits(staging: Path) -> list[str]:
    """Training splits the upload already declares, if any."""
    return [
        split
        for split in SPLITS
        if split != "unassigned" and (staging / split).is_dir()
    ]


def _move_images(
    service: "DatasetService", location: Any, staging: Path, plan: DatasetPrepPlan
) -> dict[str, int]:
    labels = list(plan.labels)
    label_index = {label: index for index, label in enumerate(labels)}
    declared = _existing_splits(staging)
    moved: dict[str, int] = {}
    # COCO sidecars are read once, up front: they key annotations by file name for
    # the whole upload rather than per directory.
    coco = _coco_index(staging, label_index)

    for path in staged_files(location.root):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(staging)
        split = _split_for_path(relative, declared)
        label = _label_for_path(relative, label_index)

        target_dir = location.root / split / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = _unique(target_dir / path.name)
        shutil.copy2(path, target)

        annotations = coco.get(path.name) or _image_annotations(
            staging, relative, label, label_index
        )
        # Only write a sidecar when there is something in it. `_annotations`
        # short-circuits on an *existing* JSON file before it ever reaches the
        # YOLO fallback, so an empty `{"annotations": []}` masks a perfectly good
        # `labels/*.txt` — 747 annotated images read as zero.
        if annotations:
            service._write_annotation_json(
                location.root / split / "annotations" / f"{target.stem}.json", annotations
            )
        _copy_yolo_label(staging, relative, location.root / split / "labels" / f"{target.stem}.txt")
        moved[split] = moved.get(split, 0) + 1

    return moved


def _image_annotations(
    staging: Path, relative: Path, label: str | None, label_index: dict[str, int]
) -> list[DatasetAnnotation]:
    """Carry an existing sidecar across, or synthesize one from the folder name."""
    sidecar = staging / relative.parent.parent / "annotations" / f"{relative.stem}.json"
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            rows = data.get("annotations", []) if isinstance(data, dict) else []
            return [DatasetAnnotation.model_validate(row) for row in rows if isinstance(row, dict)]
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    if label is None:
        return []
    return [
        DatasetAnnotation(
            class_id=label_index.get(label, 0),
            class_name=label,
            kind="classification",
        )
    ]


def _copy_yolo_label(staging: Path, relative: Path, target: Path) -> None:
    """Preserve a YOLO label file so boxes and polygons survive the move."""
    source = staging / relative.parent.parent / "labels" / f"{relative.stem}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        shutil.copy2(source, target)
    else:
        target.write_text("", encoding="utf-8")


def _split_for_path(relative: Path, declared: list[str]) -> str:
    """Respect a split the upload already declares; otherwise use the inbox."""
    for part in relative.parts:
        if part in declared:
            return part
    return "unassigned"


def _label_for_path(relative: Path, label_index: dict[str, int]) -> str | None:
    """The nearest ancestor folder whose name is one of the plan's labels."""
    for part in reversed(relative.parts[:-1]):
        if part in label_index:
            return part
    return None


def _move_records(
    service: "DatasetService", location: Any, staging: Path, plan: DatasetPrepPlan
) -> dict[str, int]:
    """Flatten every staged JSONL/JSON/CSV into the record inbox."""
    written = 0
    truncated: dict[str, int] = {}
    for path, _ in _sources(location.root, staging):
        if path.suffix.lower() not in _ROW_SUFFIXES:
            continue
        rows = _read_rows(path, limit=None)
        if len(rows) > MAX_INGEST_ROWS:
            truncated[path.name] = len(rows)
            rows = rows[:MAX_INGEST_ROWS]
        for raw_row in rows:
            row = _as_trainable_record(raw_row, plan)
            if row is None:
                continue
            try:
                record = service._validate_record(location, row)
                service._write_record(location, "unassigned", record)
            except Exception:
                # One malformed row must not abort an import of thousands. The
                # count below is what the user sees, so a silent skip still shows
                # up as a smaller number than the file held.
                continue
            written += 1
    if written:
        service._regenerate_all_records_jsonl(location)
    _record_truncation(plan, truncated)
    return {"unassigned": written}


def _move_text(
    service: "DatasetService", location: Any, staging: Path, plan: DatasetPrepPlan
) -> dict[str, int]:
    """Write text items plus the annotation that carries their supervision."""
    mapping = plan.field_mapping
    declared = _existing_splits(staging)
    moved: dict[str, int] = {}
    truncated: dict[str, int] = {}

    for path, relative in _sources(location.root, staging):
        suffix = path.suffix.lower()
        split = _split_for_path(relative, declared)

        if suffix == ".txt":
            text = _read_text(path)
            if text is None:
                continue
            annotation = _text_annotation_from_sidecar(
                staging, relative, plan
            ) or _annotation_from_folder(relative, plan)
            service._write_text_item(location, split, path.name, text, annotation)
            moved[split] = moved.get(split, 0) + 1
            continue

        if suffix in {".csv", ".tsv", ".jsonl", ".json", ".parquet"}:
            rows = _read_rows(path, limit=None)
            if len(rows) > MAX_INGEST_ROWS:
                truncated[path.name] = len(rows)
                rows = rows[:MAX_INGEST_ROWS]
            for index, row in enumerate(rows):
                text = _row_text(row, mapping)
                if not text:
                    continue
                annotation = _row_annotation(row, plan)
                service._write_text_item(
                    location, split, f"{path.stem}-{index}.txt", text, annotation
                )
                moved[split] = moved.get(split, 0) + 1

    _record_truncation(plan, truncated)
    return moved


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _text_annotation_from_sidecar(
    staging: Path, relative: Path, plan: DatasetPrepPlan
) -> DatasetAnnotation | None:
    sidecar = staging / relative.parent.parent / "annotations" / f"{relative.stem}.json"
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        rows = data.get("annotations", []) if isinstance(data, dict) else []
    except (json.JSONDecodeError, OSError):
        return None
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            return DatasetAnnotation.model_validate(_normalize_kind(row))
        except ValueError:
            continue
    return None


def _normalize_kind(row: dict) -> dict:
    """Map the long on-disk `kind` spellings onto the schema Literal.

    Reference datasets carry `summarization`/`question_answering`, while
    `DatasetAnnotation.kind` accepts `summary`/`qa`. See `detect._KIND_TO_TASK`.
    """
    row = dict(row)
    kind = str(row.get("kind") or "")
    row["kind"] = {"summarization": "summary", "question_answering": "qa"}.get(kind, kind)
    return row


def _annotation_from_folder(relative: Path, plan: DatasetPrepPlan) -> DatasetAnnotation | None:
    label_index = {label: index for index, label in enumerate(plan.labels)}
    label = _label_for_path(relative, label_index)
    if label is None:
        return None
    return DatasetAnnotation(
        class_id=label_index[label], class_name=label, kind="classification"
    )


def _row_text(row: dict, mapping: Any) -> str:
    for key in (mapping.text, mapping.question, mapping.instruction, "text"):
        if key and row.get(key):
            return str(row[key])
    return ""


def _row_annotation(row: dict, plan: DatasetPrepPlan) -> DatasetAnnotation | None:
    """Build the annotation the task needs out of the mapped columns."""
    mapping = plan.field_mapping
    task = str(plan.task_type)

    if task == "text_classification" and mapping.label:
        value = row.get(mapping.label)
        if value is None or value == "":
            return None
        label = str(value)
        labels = plan.labels
        return DatasetAnnotation(
            class_id=labels.index(label) if label in labels else 0,
            class_name=label,
            kind="classification",
        )

    if task == "summarization" and mapping.summary:
        text = row.get(mapping.summary)
        if not text:
            return None
        return DatasetAnnotation(class_name="summary", kind="summary", text=str(text))

    if task == "question_answering" and mapping.question and mapping.answer:
        question, answer = row.get(mapping.question), row.get(mapping.answer)
        if not question or not answer:
            return None
        return DatasetAnnotation(
            class_name="answer",
            kind="qa",
            question=str(question),
            answer=str(answer),
        )

    return None


def _unique(path: Path) -> Path:
    """Avoid clobbering when two class folders hold the same filename."""
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise PrepApplyError(f"Too many files named {path.name}.")


def _coco_index(staging: Path, label_index: dict[str, int]) -> dict[str, list[DatasetAnnotation]]:
    """Annotations from every COCO sidecar in the upload, keyed by image name.

    COCO is an *input* format. The platform trains YOLO, whose runner reads
    `data.yaml` plus `labels/*.txt`, and its own viewers read the JSON sidecar —
    neither looks at `_annotations.coco.json`. Worse, `_image_paths` expects COCO
    images at `<split>/` rather than `<split>/images/`, so a COCO upload left as
    COCO lands somewhere the splitter never looks and the dataset reads as empty.

    Converting on the way in avoids all of that, and avoids having to rewrite
    image ids in a COCO file every time the data is re-split.
    """
    index: dict[str, list[DatasetAnnotation]] = {}
    for sidecar in sorted(staging.rglob("*.coco.json")) + sorted(staging.rglob("_annotations.json")):
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        categories = {
            int(row["id"]): str(row.get("name") or "")
            for row in data.get("categories", [])
            if isinstance(row, dict) and "id" in row
        }
        images = {
            int(row["id"]): str(row.get("file_name") or "")
            for row in data.get("images", [])
            if isinstance(row, dict) and "id" in row
        }
        for row in data.get("annotations", []):
            if not isinstance(row, dict):
                continue
            filename = images.get(_as_int(row.get("image_id")))
            if not filename:
                continue
            name = categories.get(_as_int(row.get("category_id")), "")
            annotation = _coco_annotation(row, name, label_index)
            if annotation is not None:
                index.setdefault(filename, []).append(annotation)
    return index


def _coco_annotation(
    row: dict, class_name: str, label_index: dict[str, int]
) -> DatasetAnnotation | None:
    """One COCO annotation as the platform's own shape.

    Segmentation polygons win over the bounding box when both are present: a mask
    carries strictly more information, and a dataset that ships masks is a
    segmentation dataset.
    """
    class_id = label_index.get(class_name, 0)
    segmentation = row.get("segmentation")
    if isinstance(segmentation, list) and segmentation:
        flat = segmentation[0] if isinstance(segmentation[0], list) else segmentation
        points = [
            [float(flat[index]), float(flat[index + 1])]
            for index in range(0, len(flat) - 1, 2)
            if isinstance(flat[index], (int, float))
        ]
        if len(points) >= 3:
            return DatasetAnnotation(
                class_id=class_id, class_name=class_name, kind="polygon", polygon=points
            )

    bbox = row.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x, y, width, height = (float(value) for value in bbox)
        return DatasetAnnotation(
            class_id=class_id,
            class_name=class_name,
            kind="box",
            bbox=Box(x=x, y=y, width=width, height=height),
        )
    return None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


#: `\n\nHuman:` / `\n\nAssistant:` transcripts, as used by hh-rlhf and its many
#: derivatives. The turn marker is the only structure these files carry.
_TRANSCRIPT_TURN = re.compile(r"(?:^|\n)\s*(Human|Assistant)\s*:[ \t]*", re.IGNORECASE)

_ROLE_BY_SPEAKER = {"human": "user", "assistant": "assistant"}


def _as_trainable_record(row: dict, plan: DatasetPrepPlan) -> dict | None:
    """Normalize a raw row into something `_validate_record` accepts.

    Preference datasets are the case that needs this. `{"chosen": ..., "rejected":
    ...}` is not fine-tuning data as it stands, but the accepted side is a full
    transcript of the exchange the model should produce — so it becomes the
    conversation and the rejected side is dropped. Without this the rows fail
    validation one by one and a 6,000-record file imports as zero.
    """
    if not isinstance(row, dict):
        return None
    if "messages" in row or "conversations" in row or "instruction" in row:
        return row

    transcript = row.get("chosen")
    if isinstance(transcript, str) and transcript.strip():
        messages = parse_transcript(transcript)
        return {"messages": messages} if messages else None
    return row


def parse_transcript(text: str) -> list[dict]:
    """Split a `Human:`/`Assistant:` transcript into role-tagged turns."""
    parts = _TRANSCRIPT_TURN.split(text.strip())
    # `split` with one capture group yields [pre, speaker, body, speaker, body…];
    # `pre` is whatever preceded the first marker and is discarded.
    messages: list[dict] = []
    for index in range(1, len(parts) - 1, 2):
        role = _ROLE_BY_SPEAKER.get(parts[index].lower())
        body = parts[index + 1].strip()
        if role and body:
            messages.append({"role": role, "content": body})
    # A conversation that does not end with the assistant has nothing to learn.
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    return messages if len(messages) >= 2 else []


def _record_truncation(plan: DatasetPrepPlan, truncated: dict[str, int]) -> None:
    """Say so when a source was larger than the ingest bound."""
    for name, total in truncated.items():
        plan.warnings.append(
            f"`{name}` holds {total:,} rows; the first {MAX_INGEST_ROWS:,} were imported. "
            "Split the file and upload the rest to add more."
        )
