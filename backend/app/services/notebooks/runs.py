"""Reading back what a kernel logged.

The *writing* side is `orinth.runs`, in the kernel. This is the API's reader,
and the split matters: the writer appends one complete line per point precisely
so this can tolerate reading a file that is still being written to, which is the
normal case — the notebook page charts a run while the cell producing it is
still going.

A truncated last line is therefore expected, not corruption, and is skipped
rather than raised on.
"""

import json
from pathlib import Path

from app.schemas import NotebookRun, NotebookRunSeries

RUN_FILE = "run.json"
METRICS_FILE = "metrics.jsonl"
ARTIFACTS_DIRNAME = "artifacts"


def list_runs(runs_dir: Path, notebook_id: str) -> list[NotebookRun]:
    if not runs_dir.is_dir():
        return []
    found: list[NotebookRun] = []
    for directory in sorted(runs_dir.iterdir(), reverse=True):
        run = _read_run(directory, notebook_id)
        if run is not None:
            found.append(run)
    return found


def get_run(runs_dir: Path, notebook_id: str, run_id: str) -> NotebookRunSeries | None:
    # `Path(run_id).name` strips any directory a caller put in the URL, so a
    # `..` cannot read outside the notebook's own runs.
    directory = runs_dir / Path(run_id).name
    run = _read_run(directory, notebook_id)
    if run is None:
        return None
    return NotebookRunSeries(run=run, points=_read_points(directory / METRICS_FILE))


def artifact_path(runs_dir: Path, run_id: str, name: str) -> Path | None:
    candidate = (runs_dir / Path(run_id).name / ARTIFACTS_DIRNAME / Path(name).name).resolve()
    if not candidate.is_file():
        return None
    if runs_dir.resolve() not in candidate.parents:
        return None
    return candidate


def _read_run(directory: Path, notebook_id: str) -> NotebookRun | None:
    if not directory.is_dir():
        return None
    try:
        payload = json.loads((directory / RUN_FILE).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return NotebookRun(
            id=payload.get("id") or directory.name,
            notebook_id=payload.get("notebook_id") or notebook_id,
            name=payload.get("name") or directory.name,
            params=payload.get("params") or {},
            status=payload.get("status") or "unknown",
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            dataset_id=payload.get("dataset_id"),
            tags=list(payload.get("tags") or []),
            metric_names=list(payload.get("metric_names") or []),
            artifacts=list(payload.get("artifacts") or []),
            text=dict(payload.get("text") or {}),
        )
    except ValueError:
        # A run written by a newer SDK could carry a shape this build cannot
        # validate. Dropping one run beats failing the whole list.
        return None


def _read_points(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    points: list[dict[str, float]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # The last line of a file being appended to right now. Expected.
            continue
        if isinstance(entry, dict):
            points.append(
                {key: float(value) for key, value in entry.items() if isinstance(value, (int, float))}
            )
    return points
