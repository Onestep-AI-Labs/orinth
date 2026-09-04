"""Holding raw uploads with their directory structure intact.

The existing upload path (`items._save_uploaded_image`) flattens everything into
`<split>/images/<stem>-<hex><suffix>`. That is correct for adding a file to a
dataset whose type is already known, and fatal for detection: the layout is the
evidence. `data.yaml` beside `train/labels/` means detection; sibling `normal/`
and `kista/` folders mean classification with those classes. Flatten it and the
only remaining option is to ask the user — which is the problem phase 21 exists
to remove.

So ingest writes bytes verbatim under `_staging/`, preserving each file's
relative path, and nothing reads them except `detect`. The convention mirrors the
tabular branch's `source/` directory: never mutated, kept after apply so a
re-plan needs no re-upload.

`_locations()` globs `*/manifest.json`, so a staging directory is invisible to
the catalog until a manifest is written beside it.
"""

from pathlib import Path

STAGING_DIRNAME = "_staging"
#: Where a sandboxed transform writes its output (see `prep/transform.py`).
#: A sibling of `_staging/` rather than a child, so `staged_files()` keeps
#: meaning "what the user uploaded" and apply can choose between the two
#: without filtering. Both are invisible to the catalog for the same reason:
#: `_locations()` globs `*/manifest.json`.
DERIVED_DIRNAME = "_derived"

#: Path segments that can escape the staging root or confuse the walk.
_REJECTED_SEGMENTS = {"", ".", ".."}


def staging_root(dataset_root: Path) -> Path:
    return dataset_root / STAGING_DIRNAME


def safe_relative_path(raw: str | None, fallback: str) -> Path | None:
    """Sanitize a client-supplied relative path, one segment at a time.

    Returns `None` when nothing usable survives. The browser supplies these from
    `File.webkitRelativePath` or a directory-picker walk, which makes them
    attacker-controlled in the same way any upload filename is: a `..` segment or
    a leading `/` would write outside the dataset.
    """
    candidate = (raw or "").strip().replace("\\", "/")
    if not candidate:
        candidate = (fallback or "").strip().replace("\\", "/")

    segments: list[str] = []
    for segment in candidate.split("/"):
        segment = segment.strip()
        if segment in _REJECTED_SEGMENTS:
            continue
        if segment.startswith("~"):
            continue
        # Windows drive letters and NUL bytes.
        segment = segment.replace("\x00", "")
        if ":" in segment:
            segment = segment.split(":")[-1]
        if not segment or segment in _REJECTED_SEGMENTS:
            continue
        segments.append(segment[:120])

    if not segments:
        return None
    return Path(*segments)


def resolve_within(root: Path, relative: Path) -> Path | None:
    """Final guard: the destination must resolve inside the staging root.

    `safe_relative_path` already strips traversal segment by segment; this
    catches anything it did not anticipate, including a symlinked parent.
    """
    try:
        destination = (root / relative).resolve()
        root_resolved = root.resolve()
    except OSError:
        return None
    if destination == root_resolved or root_resolved not in destination.parents:
        return None
    return destination


def staged_files(dataset_root: Path) -> list[Path]:
    root = staging_root(dataset_root)
    if not root.is_dir():
        return []
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def staged_count(dataset_root: Path) -> int:
    return len(staged_files(dataset_root))


def derived_root(dataset_root: Path) -> Path:
    return dataset_root / DERIVED_DIRNAME


def derived_files(dataset_root: Path) -> list[Path]:
    root = derived_root(dataset_root)
    if not root.is_dir():
        return []
    return [path for path in sorted(root.rglob("*")) if path.is_file()]
