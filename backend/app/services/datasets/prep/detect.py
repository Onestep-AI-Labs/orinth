"""What is this pile of files?

The platform used to answer that by asking the user: pick a domain, pick a task,
pick a record schema, type your class labels — all before a single byte had been
uploaded. Detection answers it from the files instead.

Everything here is deterministic. No model is consulted, nothing leaves the
machine, and the same directory always produces the same verdict. That matters
for two reasons: it works with no API key, and it gives the LLM stage in
``plan.py`` a factual floor it is not allowed to contradict — a YOLO root is a
YOLO root no matter what a model says about it.

The signal being read is **directory structure**, which is why ingest preserves
relative paths (see ``prep/ingest.py``). ``data.yaml`` beside ``train/labels/``
means detection; sibling ``normal/`` and ``kista/`` folders mean classification
with those two classes. The old upload path flattened every file into one
directory, which destroyed exactly this information.
"""

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.services.datasets.constants import IMAGE_SUFFIXES, SPLITS
from app.services.datasets.format_io import (
    is_yolo_root,
    labels_from_coco,
    labels_from_yolo_yaml,
)

#: Delimited/serialized row containers that carry their own column names.
TABLE_SUFFIXES = {".csv", ".tsv", ".parquet"}
RECORD_SUFFIXES = {".jsonl", ".json"}

#: Rows sampled for the schema summary handed to the planner. Kept small: it
#: rides in an LLM prompt, and twenty rows already show the shape.
SAMPLE_ROW_LIMIT = 20
#: Rows read for *profiling*, which is a different question from what to show a
#: model. Cardinality is the tell for a label column, and at twenty rows every
#: column has at most twenty distinct values — so free text is indistinguishable
#: from a closed class set. Reading more rows separates them; only the first
#: `SAMPLE_ROW_LIMIT` are ever put in a prompt.
PROFILE_ROW_LIMIT = 500
#: Cell values truncated to this before sampling, so one HTML blob in row 3
#: cannot blow the prompt budget.
SAMPLE_CELL_CHARS = 200
#: Above this share of distinct values a column is an id or a timestamp rather
#: than a class label.
_MAX_LABEL_VALUES = 20
#: Below this mean length a string column reads as a category, above it as prose.
_TEXT_FEATURE_CHARS = 40

# Column vocabularies. These are the same names `format_io._annotation_from_text_row`
# and `_text_from_upload_row` already sniff for. Detection deliberately reuses
# them rather than introducing a second vocabulary that could drift: if the
# importer and the agent disagreed about what a "label" column is called, a
# dataset would detect one way and import another.
TEXT_COLUMNS = ("text", "content", "source", "document", "context", "body")
LABEL_COLUMNS = ("label", "class", "class_name", "category", "sentiment", "target")
SUMMARY_COLUMNS = ("summary", "reference_summary", "highlights")
QUESTION_COLUMNS = ("question", "query")
ANSWER_COLUMNS = ("answer", "answers")


@dataclass
class Detection:
    """What the files say they are, plus the evidence for saying so.

    ``signals`` is not decoration. It is what the Overview screen shows the user
    so a decision can be checked against what they uploaded, rather than taken
    on trust — "312 files across 3 folders: normal, kista, granuloma" is
    verifiable in a way that "classification, 85% confident" is not.
    """

    modality: str = "unknown"
    task_type: str | None = None
    format: str | None = None
    confidence: float = 0.0
    candidate_labels: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    sample_rows: list[dict] = field(default_factory=list)
    file_counts: dict[str, int] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)
    #: role -> column name, decided here because this is where the full profile
    #: lives. `sample_rows` is capped for the prompt, so anything re-deriving a
    #: mapping from it downstream would be working from strictly less evidence.
    column_roles: dict[str, str] = field(default_factory=dict)
    # Set when the structure is recognized but something is missing that only a
    # human can supply — most often class labels for unlabelled images.
    needs_input: str | None = None


def detect(root: Path) -> Detection:
    """Classify a staged upload directory.

    Rules are evaluated most-specific first and the first match wins, so an
    explicit `data.yaml` beats a folder-name inference and a COCO sidecar beats
    a bare image tree.
    """
    if not root.exists() or not root.is_dir():
        return Detection(signals=["No files were uploaded."])

    root, descended = _descend_to_content(root)

    files = [path for path in root.rglob("*") if path.is_file()]
    if not files:
        return Detection(signals=["No files were uploaded."])

    counts = Counter(path.suffix.lower() for path in files)
    file_counts = {suffix or "(none)": count for suffix, count in counts.items()}

    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    tables = [path for path in files if path.suffix.lower() in TABLE_SUFFIXES]
    records = [path for path in files if path.suffix.lower() in RECORD_SUFFIXES]
    texts = [path for path in files if path.suffix.lower() == ".txt"]

    for rule in (
        _detect_manifest,
        _detect_orinth_annotations,
        _detect_yolo,
        _detect_coco,
        _detect_record_files,
        _detect_image_folders,
        _detect_text_folders,
        _detect_tables,
    ):
        found = rule(root, images=images, tables=tables, records=records, texts=texts)
        if found is not None:
            found.file_counts = file_counts
            if descended:
                found.signals.insert(0, f"Read the dataset from the `{descended}` folder inside the upload.")
            return found

    return Detection(
        file_counts=file_counts,
        signals=[f"{len(files)} files with no recognizable structure."],
        needs_input="Orinth could not tell what this data is. Pick a task type manually.",
    )


#: How many single-child wrapper folders to unwrap before giving up. Three covers
#: the realistic nesting (a zip of a folder of a dataset) without walking forever.
_MAX_DESCENT = 3

#: Folder names that are part of a dataset's layout rather than a wrapper around
#: it. A dataset holding only `train/` is content — descending into it would put
#: the split folder at the root and hide the very structure being looked for.
_STRUCTURAL_DIRS = set(SPLITS) | {
    "images",
    "texts",
    "annotations",
    "labels",
    "records",
}


def _descend_to_content(root: Path) -> tuple[Path, str | None]:
    """Step through wrapper folders that contain nothing but one subdirectory.

    People upload `my-dataset/` containing `dataset/` containing the actual
    splits, and an archive tool adds a wrapper of its own. Detecting on the
    wrapper sees one folder and concludes "no class grouping" — which is how
    `sample_data/vision` reads as 300 unlabelled images when the dataset one
    level down declares six classes in its own manifest.
    """
    current = root
    name: str | None = None
    for _ in range(_MAX_DESCENT):
        try:
            entries = [entry for entry in current.iterdir() if not entry.name.startswith(".")]
        except OSError:
            break
        # `__MACOSX` rides along in archives made on macOS and is never content.
        directories = [
            entry for entry in entries if entry.is_dir() and entry.name != "__MACOSX"
        ]
        files = [entry for entry in entries if entry.is_file()]
        if len(directories) != 1 or files:
            break
        only = directories[0]
        if only.name.lower() in _STRUCTURAL_DIRS:
            break
        current = only
        name = current.name
    return current, name


# --- rules 0-1: data Orinth itself wrote --------------------------------------
#
# These run before every structural guess, for the same reason
# `format_io._annotations` prefers a JSON sidecar over YOLO and COCO: a
# declaration beats an inference. Without them, re-uploading a dataset Orinth
# exported gets re-guessed — and guessed wrong, because a classification dataset
# stored with `<split>/images` and `<split>/labels` looks exactly like a YOLO
# detection root from the outside.


#: `kind` values seen on annotation sidecars, mapped to the task they imply.
#: Both spellings are real: `DatasetAnnotation.kind` is the short form, while the
#: reference datasets on disk carry the long task-type form.
_KIND_TO_TASK = {
    "classification": "classification",
    "box": "object_detection",
    "polygon": "segmentation",
    "summary": "summarization",
    "summarization": "summarization",
    "qa": "question_answering",
    "question_answering": "question_answering",
}

_KNOWN_TASKS = {
    "classification",
    "object_detection",
    "segmentation",
    "text_classification",
    "summarization",
    "question_answering",
    "llm_finetune",
    "language_modeling",
}


def _detect_manifest(root: Path, **_: object) -> Detection | None:
    """A dataset Orinth exported, re-uploaded whole. Trust its own manifest."""
    path = root / "manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None

    task = data.get("task_type")
    if task not in _KNOWN_TASKS:
        return None

    labels = [str(label) for label in data.get("labels", []) if label]
    return Detection(
        modality=_modality_for_task(str(task)),
        task_type=str(task),
        format=str(data.get("format") or "") or None,
        confidence=1.0,
        candidate_labels=labels,
        signals=[
            f"`manifest.json` declares this an Orinth `{task}` dataset.",
            f"It names {len(labels)} label(s): {', '.join(labels) or 'none'}."
            if labels
            else "It names no labels.",
        ],
    )


def _detect_orinth_annotations(root: Path, **_: object) -> Detection | None:
    """`<split>/annotations/*.json` sidecars, which state the task outright."""
    kinds: Counter[str] = Counter()
    class_names: set[str] = set()
    sampled = 0

    for split in SPLITS:
        annotation_dir = root / split / "annotations"
        if not annotation_dir.is_dir():
            continue
        for path in sorted(annotation_dir.glob("*.json")):
            rows = _annotation_rows(path)
            for row in rows:
                kind = str(row.get("kind") or "")
                if kind:
                    kinds[kind] += 1
                name = row.get("class_name")
                if name:
                    class_names.add(str(name))
            sampled += 1
            if sampled >= SAMPLE_ROW_LIMIT:
                break
        if sampled >= SAMPLE_ROW_LIMIT:
            break

    if not kinds:
        return None

    kind, _count = kinds.most_common(1)[0]
    task = _KIND_TO_TASK.get(kind)
    if task is None:
        return None

    has_images = any((root / split / "images").is_dir() for split in SPLITS)
    has_texts = any((root / split / "texts").is_dir() for split in SPLITS)

    # `classification` is the one ambiguous kind: it means image or text
    # classification depending on which media directory sits beside it.
    if task == "classification" and has_texts and not has_images:
        task = "text_classification"

    labels = sorted(class_names)
    if task == "summarization":
        labels = ["summary"]
    elif task == "question_answering":
        labels = ["answer"]

    return Detection(
        modality=_modality_for_task(task),
        task_type=task,
        format=_format_for_task(task, has_images=has_images),
        confidence=0.95,
        candidate_labels=labels,
        signals=[
            f"Annotation sidecars declare `kind: {kind}` ({kinds[kind]} annotations sampled).",
            f"Sidecars name {len(labels)} class(es): {', '.join(labels)}."
            if labels
            else "Sidecars name no classes.",
        ],
    )


def _annotation_rows(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, dict):
        rows = data.get("annotations", [])
        return [row for row in rows if isinstance(row, dict)]
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _modality_for_task(task: str) -> str:
    if task in {"classification", "object_detection", "segmentation"}:
        return "image"
    if task == "llm_finetune":
        return "record"
    return "text"


def _format_for_task(task: str, *, has_images: bool) -> str:
    if task in {"object_detection", "segmentation"}:
        return "yolo"
    if task == "classification" or has_images:
        return "image_folder"
    return "text_folder"


# --- image rules --------------------------------------------------------------


def _detect_yolo(root: Path, *, images: list[Path], **_: object) -> Detection | None:
    """Split folders holding `images/`, the layout `training_root` expects."""
    if not images or not is_yolo_root(root):
        return None

    signals = ["Split folders containing `images/` (YOLO layout)."]
    labels = labels_from_yolo_yaml(root)
    has_yaml = (root / "data.yaml").exists()
    if has_yaml:
        signals.append(f"`data.yaml` names {len(labels)} class(es): {', '.join(labels) or 'none'}.")

    geometry, geometry_signal = _yolo_label_geometry(root)
    if geometry_signal:
        signals.append(geometry_signal)

    task = "segmentation" if geometry == "polygon" else "object_detection"
    # An explicit data.yaml is a declaration; inferring from label files alone is
    # a guess, and the confidence has to say which one happened.
    confidence = 0.95 if has_yaml else 0.55
    if geometry == "polygon":
        confidence = min(confidence, 0.90)

    return Detection(
        modality="image",
        task_type=task,
        format="yolo",
        confidence=confidence,
        candidate_labels=labels,
        signals=signals,
        needs_input=None
        if labels
        else "This YOLO dataset has no `data.yaml`, so its class names are unknown.",
    )


def _yolo_label_geometry(root: Path) -> tuple[str | None, str | None]:
    """Read one label file to tell boxes from polygons.

    A YOLO detection line is `class cx cy w h` — five tokens. A segmentation line
    is `class` followed by an even number of polygon coordinates, so it is odd
    overall and longer than five. That single distinction is the whole difference
    between `object_detection` and `segmentation` here.
    """
    for split in SPLITS:
        label_dir = root / split / "labels"
        if not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            try:
                text = label_path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                tokens = line.split()
                if len(tokens) == 5:
                    return "box", "Label lines carry 5 values (bounding boxes)."
                if len(tokens) > 5 and len(tokens) % 2 == 1:
                    return (
                        "polygon",
                        f"Label lines carry {len(tokens)} values (polygon segmentation).",
                    )
    return None, None


def _detect_coco(root: Path, *, images: list[Path], **_: object) -> Detection | None:
    """A COCO sidecar, which says whether it holds masks or only boxes."""
    sidecars = sorted(root.rglob("_annotations.coco.json"))
    if not sidecars or not images:
        return None

    sidecar = sidecars[0]
    labels = labels_from_coco(sidecar.parent)
    has_masks = _coco_has_segmentation(sidecar)
    task = "segmentation" if has_masks else "object_detection"

    return Detection(
        modality="image",
        task_type=task,
        format="coco",
        confidence=0.90,
        candidate_labels=labels,
        signals=[
            f"COCO sidecar `{sidecar.name}` with {len(labels)} categor{'y' if len(labels) == 1 else 'ies'}.",
            "Annotations carry segmentation masks."
            if has_masks
            else "Annotations carry bounding boxes only.",
        ],
        needs_input=None if labels else "The COCO sidecar declares no categories.",
    )


def _coco_has_segmentation(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(data, dict):
        return False
    return any(
        isinstance(annotation, dict) and annotation.get("segmentation")
        for annotation in data.get("annotations", [])
    )


def _detect_image_folders(root: Path, *, images: list[Path], **_: object) -> Detection | None:
    """Images grouped by folder, where the folder name is the class."""
    if not images:
        return None

    by_parent: dict[Path, int] = Counter(path.parent for path in images)
    # A class folder holds images directly. Folders named `images` are part of a
    # split layout, not class names, and would otherwise become a "class".
    class_dirs = sorted(
        parent for parent in by_parent if parent != root and parent.name not in {"images", "texts"}
    )

    if len(class_dirs) >= 2:
        labels = sorted({parent.name for parent in class_dirs})
        return Detection(
            modality="image",
            task_type="classification",
            format="image_folder",
            confidence=0.85,
            candidate_labels=labels,
            signals=[
                f"{len(images)} images across {len(labels)} folders: {', '.join(labels)}.",
                "Folder names read as class labels.",
            ],
        )

    # Flat images with nothing to learn from. This is the case that used to train
    # silently on zero items, because `keras_common.load_split` drops every image
    # with no annotation and reports success on what is left.
    return Detection(
        modality="image",
        task_type="classification",
        format="image_folder",
        confidence=0.40,
        candidate_labels=[],
        signals=[f"{len(images)} images with no folder grouping and no annotations."],
        needs_input=(
            "These images carry no labels. Add class labels, or sort them into one "
            "folder per class and upload again."
        ),
    )


# --- record and text rules ----------------------------------------------------


def detect_record_format(columns: list[str]) -> tuple[str | None, dict[str, str]]:
    """Name the instruction/chat/QA shape a set of column names describes.

    Moved here from `DatasetHubService._detect_format` (phase 21) so hub import
    and the prep agent share one answer. `dataset_hub.py` now delegates to it.
    """
    cols = set(columns)
    if {"instruction", "output"} <= cols:
        mapping = {"instruction": "instruction", "output": "output"}
        if "input" in cols:
            mapping["input"] = "input"
        return "alpaca", mapping
    if "messages" in cols:
        return "messages", {"messages": "messages"}
    if "conversations" in cols:
        return "sharegpt", {"conversations": "conversations"}
    if {"question", "answer"} <= cols:
        mapping = {"question": "question", "answer": "answer"}
        if "context" in cols:
            mapping["context"] = "context"
        return "qa", mapping
    return None, {}


def _detect_record_files(root: Path, *, records: list[Path], **_: object) -> Detection | None:
    """JSONL/JSON rows whose keys name an instruction, chat, or QA shape."""
    if not records:
        return None

    rows: list[dict] = []
    for path in sorted(records):
        rows.extend(_read_rows(path, limit=PROFILE_ROW_LIMIT))
        if len(rows) >= PROFILE_ROW_LIMIT:
            break
    if not rows:
        return None

    columns = _columns_of(rows)
    shape, _mapping = detect_record_format(columns)
    if shape is None:
        return _detect_preference_rows(sorted(records)[0], rows, columns)

    if shape == "alpaca":
        return Detection(
            modality="record",
            task_type="llm_finetune",
            format="instruction_jsonl",
            confidence=0.90,
            columns=columns,
            sample_rows=_sample(rows),
            signals=[f"Records carry instruction/output keys ({len(rows)}+ rows sampled)."],
        )
    if shape in {"messages", "sharegpt"}:
        return Detection(
            modality="record",
            task_type="llm_finetune",
            format="chat_jsonl",
            confidence=0.90,
            columns=columns,
            sample_rows=_sample(rows),
            signals=[f"Records carry a `{'messages' if shape == 'messages' else 'conversations'}` turn list."],
        )
    # shape == "qa": extractive question answering, not LLM fine-tuning.
    return Detection(
        modality="text",
        task_type="question_answering",
        format="text_folder",
        confidence=0.85,
        columns=columns,
        sample_rows=_sample(rows),
        candidate_labels=["answer"],
        signals=["Rows carry question/answer columns."],
    )


def _detect_preference_rows(path: Path, rows: list[dict], columns: list[str]) -> Detection | None:
    """`chosen`/`rejected` preference pairs, as shipped by hh-rlhf and its kin.

    A preference set is not supervised fine-tuning data, but the accepted side of
    each pair is: `chosen` is a complete transcript of the conversation the model
    should have produced. Training on it and discarding `rejected` is the
    standard way these become SFT data, so the agent says so rather than
    reporting the file as unrecognized.
    """
    if "chosen" not in columns:
        return None
    return Detection(
        modality="record",
        task_type="llm_finetune",
        format="chat_jsonl",
        confidence=0.80,
        columns=columns,
        sample_rows=_sample(rows),
        signals=[
            f"`{path.name}` carries `chosen`"
            + ("/`rejected` preference pairs." if "rejected" in columns else " transcripts."),
            "The accepted side becomes the training conversation; the rejected side is dropped.",
        ],
    )


def _detect_text_folders(root: Path, *, texts: list[Path], **_: object) -> Detection | None:
    """`.txt` files grouped by folder, where the folder name is the class."""
    if not texts:
        return None

    class_dirs = sorted(
        {path.parent for path in texts if path.parent != root and path.parent.name != "texts"}
    )
    if len(class_dirs) >= 2:
        labels = sorted({parent.name for parent in class_dirs})
        return Detection(
            modality="text",
            task_type="text_classification",
            format="text_folder",
            confidence=0.80,
            candidate_labels=labels,
            signals=[
                f"{len(texts)} text files across {len(labels)} folders: {', '.join(labels)}.",
            ],
        )

    # An unstructured corpus is next-token training data, not a labelled set.
    return Detection(
        modality="text",
        task_type="language_modeling",
        format="text_folder",
        confidence=0.60,
        signals=[f"{len(texts)} text files with no class folders; reads as a plain corpus."],
    )


def _detect_tables(root: Path, *, tables: list[Path], **_: object) -> Detection | None:
    """A delimited or Parquet file, classified by what its columns turn out to hold."""
    if not tables:
        return None
    path = sorted(tables)[0]
    return _classify_rows(path, _read_rows(path, limit=PROFILE_ROW_LIMIT))


def _roles(**pairs: str | None) -> dict[str, str]:
    """Drop the roles nothing filled, so an empty value never reads as a column."""
    return {role: column for role, column in pairs.items() if column}


def _classify_rows(path: Path, rows: list[dict]) -> Detection | None:
    """Name the task a table supports, from its columns and their contents.

    Column *names* are tried first, then their *shape*. Real datasets rarely use
    the tidy names: Kaggle's SMS spam set ships `v1` and `v2`, and no vocabulary
    will ever cover that. What does cover it is that one column averages four
    characters across two distinct values and the other averages eighty.
    """
    columns = _columns_of(rows)
    if not columns:
        return Detection(
            modality="table",
            confidence=0.30,
            signals=[f"`{path.name}` could not be read as rows."],
            needs_input="Orinth could not read this file. Convert it to CSV or JSONL.",
        )

    sample = _sample(rows)
    stats = profile_columns(rows, columns)

    shape, _mapping = detect_record_format(columns)
    if shape == "alpaca":
        return Detection(
            modality="record", task_type="llm_finetune", format="instruction_jsonl",
            confidence=0.85, columns=columns, sample_rows=sample,
            signals=[f"`{path.name}` carries instruction/output columns."],
        )
    if shape in {"messages", "sharegpt"}:
        return Detection(
            modality="record", task_type="llm_finetune", format="chat_jsonl",
            confidence=0.85, columns=columns, sample_rows=sample,
            signals=[f"`{path.name}` carries a turn list."],
        )

    question = _match_column(columns, QUESTION_COLUMNS)
    answer = _match_column(columns, ANSWER_COLUMNS)
    if question and answer:
        return Detection(
            modality="text", task_type="question_answering", format="text_folder",
            confidence=0.85, columns=columns, sample_rows=sample,
            candidate_labels=["answer"],
            column_roles=_roles(question=question, answer=answer,
                                text=infer_text_column(columns, stats, exclude={question, answer})),
            signals=[f"`{path.name}` carries `{question}` and `{answer}` columns."],
        )

    summary = _match_column(columns, SUMMARY_COLUMNS)
    text = infer_text_column(columns, stats, exclude={summary} if summary else set())
    if summary and text:
        return Detection(
            modality="text", task_type="summarization", format="text_folder",
            confidence=0.85, columns=columns, sample_rows=sample,
            candidate_labels=["summary"],
            column_roles=_roles(text=text, summary=summary),
            signals=[f"`{path.name}` carries `{text}` and `{summary}` columns."],
        )

    text = infer_text_column(columns, stats, exclude=set())
    label = infer_label_column(columns, stats, exclude={text} if text else set())
    if text and label:
        values = _distinct_values(rows, label)
        named = _first_present([label], LABEL_COLUMNS) is not None
        return Detection(
            modality="text", task_type="text_classification", format="text_folder",
            confidence=0.80 if named else 0.65,
            columns=columns, sample_rows=sample,
            candidate_labels=sorted(values),
            column_roles=_roles(text=text, label=label),
            signals=[
                f"`{path.name}`: `{text}` holds the text "
                f"(~{stats[text].mean_chars:.0f} characters per row).",
                f"`{label}` holds {len(values)} distinct values: "
                f"{', '.join(sorted(values)[:6])}.",
            ],
        )

    return Detection(
        modality="table", confidence=0.30, columns=columns, sample_rows=sample,
        signals=[
            f"`{path.name}` has {len(columns)} columns "
            f"({', '.join(columns[:6])}) with no recognizable task mapping.",
        ],
        needs_input=(
            "Orinth could not map these columns to a task it can train. Pick a task "
            "and the columns that feed it."
        ),
    )


# --- helpers ------------------------------------------------------------------


def _read_rows(path: Path, limit: int | None = SAMPLE_ROW_LIMIT) -> list[dict]:
    """Best-effort row read. Detection never fails on a malformed file.

    `limit` defaults to the detection sample. Apply passes `None` for every row —
    keeping the default here silently ingested 20 rows of a 52,000-row Parquet.
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _read_parquet_rows(path, limit)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if suffix == ".jsonl":
        rows = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
            if limit and len(rows) >= limit:
                break
        return rows
    if suffix == ".json":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
        if isinstance(parsed, dict):
            # A wrapper object around the real rows, e.g. {"data": [...]}.
            for value in parsed.values():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    return value
            return [parsed]
        return []
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        reader = csv.DictReader(raw.splitlines(), delimiter=delimiter)
        rows = []
        for row in reader:
            rows.append(dict(row))
            if limit and len(rows) >= limit:
                break
        return rows
    return []


def _columns_of(rows: list[dict]) -> list[str]:
    """Union of keys across sampled rows, first-seen order preserved."""
    seen: dict[str, None] = {}
    for row in rows[:PROFILE_ROW_LIMIT]:
        for key in row:
            seen.setdefault(str(key), None)
    return list(seen)


def _sample(rows: list[dict]) -> list[dict]:
    """A prompt-safe slice: few rows, short cells."""
    return [
        {str(key): _truncate(value) for key, value in row.items()}
        for row in rows[:SAMPLE_ROW_LIMIT]
    ]


def _truncate(value: object) -> object:
    if isinstance(value, str) and len(value) > SAMPLE_CELL_CHARS:
        return value[:SAMPLE_CELL_CHARS] + "…"
    return value


def _first_present(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _distinct_values(rows: list[dict], column: str) -> set[str]:
    values: set[str] = set()
    for row in rows:
        value = row.get(column)
        if value is None or value == "":
            continue
        values.add(str(value))
        if len(values) > _MAX_LABEL_VALUES:
            break
    return values


def _read_parquet_rows(path: Path, limit: int | None = SAMPLE_ROW_LIMIT) -> list[dict]:
    """A sample of a Parquet file, without loading the whole thing.

    Parquet is the single most common shape on the Hugging Face Hub — most
    datasets there ship no other format — so treating it as unreadable would
    make the agent useless against the place users actually get data.

    Imported lazily: `detect` is on the import path of `DatasetService`, and the
    catalog should not pay for a dataframe library it may never use.
    """
    try:
        import polars as pl
    except ImportError:
        return []
    try:
        frame = pl.read_parquet(path, n_rows=limit) if limit else pl.read_parquet(path)
    except Exception:
        # Detection runs over files the user just dropped; an unreadable one
        # means "no rows here", not a crashed upload.
        return []
    rows: list[dict] = []
    for row in frame.to_dicts():
        rows.append({str(key): _stringify(value) for key, value in row.items()})
    return rows


def _stringify(value: object) -> object:
    """Flatten nested Parquet/Arrow values so column sniffing sees plain text."""
    if isinstance(value, dict):
        # SQuAD stores answers as {"text": [...], "answer_start": [...]}.
        for key in ("text", "value", "content"):
            inner = value.get(key)
            if isinstance(inner, list) and inner:
                return str(inner[0])
            if isinstance(inner, str) and inner:
                return inner
        return json.dumps(value, ensure_ascii=False, default=str)[:SAMPLE_CELL_CHARS]
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (str, int, float)):
            return str(value[0])
        return json.dumps(list(value), ensure_ascii=False, default=str)[:SAMPLE_CELL_CHARS]
    return value


@dataclass(frozen=True)
class ColumnStats:
    """What a column looks like, independent of what it is called."""

    name: str
    distinct: int
    mean_chars: float
    filled: int


def profile_columns(rows: list[dict], columns: list[str]) -> dict[str, ColumnStats]:
    stats: dict[str, ColumnStats] = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        present = [str(value) for value in values if value not in (None, "")]
        distinct = len({value for value in present})
        mean = sum(len(value) for value in present) / len(present) if present else 0.0
        stats[column] = ColumnStats(
            name=column, distinct=distinct, mean_chars=mean, filled=len(present)
        )
    return stats


def _match_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Exact name first, then a word-boundary substring match.

    Real column names are rarely the bare word. Kaggle's airline dataset calls
    its label `airline_sentiment` and its text `text`; the COVID one calls them
    `Sentiment` and `OriginalTweet`. Matching only exact names finds the second
    but not the first, which is worse than matching neither because it produces a
    half-mapping that looks deliberate.
    """
    exact = _first_present(columns, candidates)
    if exact:
        return exact
    for candidate in candidates:
        for column in columns:
            lowered = column.lower()
            parts = set(re.split(r"[^a-z0-9]+", lowered))
            if candidate in parts:
                return column
    for candidate in candidates:
        for column in columns:
            if candidate in column.lower():
                return column
    return None


def infer_text_column(
    columns: list[str], stats: dict[str, ColumnStats], exclude: set[str]
) -> str | None:
    """The free-text column, by name or — failing that — by being the longest.

    `v2` in Kaggle's SMS spam set carries the message and `v1` the class; neither
    name says so. Length does: one averages ~80 characters, the other ~4.
    """
    named = _match_column([c for c in columns if c not in exclude], TEXT_COLUMNS)
    if named:
        return named
    ranked = sorted(
        (stat for name, stat in stats.items() if name not in exclude),
        key=lambda stat: stat.mean_chars,
        reverse=True,
    )
    for stat in ranked:
        if stat.mean_chars >= _TEXT_FEATURE_CHARS and stat.filled:
            return stat.name
    return None


def _looks_like_classes(stat: ColumnStats) -> bool:
    """Whether a column reads as a closed set of classes.

    The decisive property is *repetition*. A column whose every value is distinct
    is an identifier or a free-text field however short it is, and a column named
    `target` holding a different sentence per row is a summarization target, not
    a taxonomy. Requiring distinct values to be at most half the rows seen keeps
    both out without a hardcoded list of column names to distrust.
    """
    if not stat.filled or stat.distinct < 2:
        return False
    if stat.mean_chars >= _TEXT_FEATURE_CHARS:
        return False
    ceiling = max(2, min(_MAX_LABEL_VALUES, stat.filled // 2))
    return stat.distinct <= ceiling


def infer_label_column(
    columns: list[str], stats: dict[str, ColumnStats], exclude: set[str]
) -> str | None:
    """The class column, by name or by looking like a small closed set."""
    named = _match_column([c for c in columns if c not in exclude], LABEL_COLUMNS)
    if named and _looks_like_classes(stats[named]):
        return named
    candidates = [
        stat
        for name, stat in stats.items()
        if name not in exclude and _looks_like_classes(stat)
    ]
    if not candidates:
        return None
    # Fewest classes wins: a 2-value column is far more likely the target than a
    # 19-value one, which is usually a category attribute riding alongside.
    return min(candidates, key=lambda stat: (stat.distinct, -stat.mean_chars)).name
