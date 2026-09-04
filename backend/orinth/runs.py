"""Lightweight experiment logging, written by the kernel and read by the API.

**Not `TrainingJob` rows, deliberately.** That model carries a model id, a
dataset id, hyperparameters, an artifacts contract, a promote path, and
cancel-by-PID — a notebook experiment has none of them. Worse,
`reconcile_stale_jobs` marks every queued or running job `failed` at startup
because background subprocesses do not survive a restart; a notebook run has no
process to reconcile, so every run would be flipped to `failed` on the next boot.
Runs stay a per-notebook artifact, charted on the notebook page and nowhere else.

Everything is **append-only, one complete line per write**. A kernel killed
mid-experiment leaves a readable partial series rather than a truncated JSON
blob, which is the difference between losing the last point and losing the run.
"""

import atexit
import builtins
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from orinth import _workspace
from orinth.errors import OrinthError
from orinth.types import RunRef

RUN_FILE = "run.json"
METRICS_FILE = "metrics.jsonl"
ARTIFACTS_DIRNAME = "artifacts"

#: The run `orinth.runs.log(...)` writes to when nobody started one explicitly.
_implicit: "Run | None" = None


def _now() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


def _runs_root() -> Path:
    directory = _workspace.notebook_dir()
    if directory is None:
        raise OrinthError(
            "orinth.runs only works inside a notebook kernel — "
            "ORINTH_NOTEBOOK_ID is not set in this process."
        )
    root = directory / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


class Run:
    """One experiment. Also a context manager, which is the intended form.

    `with orinth.runs.start("baseline") as run:` finishes the run on the way out
    whether the cell raised or not, and records *which* — an experiment that
    crashed is a real outcome and must not be indistinguishable from one nobody
    finished.
    """

    def __init__(self, run_id: str, directory: Path, payload: dict) -> None:
        self.id = run_id
        self.directory = directory
        self._payload = payload
        self._metrics = directory / METRICS_FILE
        self._finished = False

    # -- lifecycle --

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.finish(status="failed" if exc_type else "completed")
        return False

    def finish(self, status: str = "completed") -> None:
        if self._finished:
            return
        self._finished = True
        self._payload["status"] = status
        self._payload["finished_at"] = _now()
        self._write()

    # -- logging --

    def log(self, step: int | None = None, **metrics: float) -> None:
        """Append one point. Scalars only.

        Non-numeric values are refused rather than coerced: a metrics file with
        a string in it breaks every chart that reads it, and the caller almost
        certainly meant `log_text`.
        """
        point: dict[str, Any] = {"t": time.time()}
        point["step"] = int(step) if step is not None else self._next_step()
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OrinthError(
                    f"log() takes numbers; '{key}' is {type(value).__name__}. "
                    "Use log_text() for strings."
                )
            point[key] = float(value)
        with self._metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(point) + "\n")
        known = set(self._payload.get("metric_names") or [])
        known.update(key for key in metrics)
        self._payload["metric_names"] = sorted(known)
        self._write()

    def log_text(self, key: str, value: str) -> None:
        self._payload.setdefault("text", {})[key] = str(value)
        self._write()

    def log_artifact(self, source: "Path | bytes", *, name: str) -> Path:
        """Copy a file, or write bytes, into the run's artifacts directory."""
        artifacts = self.directory / ARTIFACTS_DIRNAME
        artifacts.mkdir(parents=True, exist_ok=True)
        # `Path(name).name` strips any directory the caller passed, so an
        # artifact cannot be written outside the run it belongs to.
        target = artifacts / Path(name).name
        if isinstance(source, bytes):
            target.write_bytes(source)
        else:
            shutil.copy2(Path(source), target)
        listed = set(self._payload.get("artifacts") or [])
        listed.add(target.name)
        self._payload["artifacts"] = sorted(listed)
        self._write()
        return target

    # -- internals --

    def _next_step(self) -> int:
        """Line count is the step when the caller does not supply one.

        Counting lines rather than holding a counter means a run reattached
        after a kernel restart continues where the file left off instead of
        overwriting from zero.
        """
        if not self._metrics.is_file():
            return 0
        with self._metrics.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _write(self) -> None:
        (self.directory / RUN_FILE).write_text(
            json.dumps(self._payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"<Run {self._payload.get('name')!r} {self._payload.get('status')}>"


def start(
    name: str,
    *,
    params: Mapping | None = None,
    dataset_id: str | None = None,
    tags: Sequence[str] = (),
) -> Run:
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    directory = _runs_root() / run_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": run_id,
        "notebook_id": _workspace.notebook_id(),
        "name": name,
        "params": dict(params or {}),
        "dataset_id": dataset_id,
        "tags": builtins.list(tags),
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "metric_names": [],
        "artifacts": [],
        "text": {},
    }
    run = Run(run_id, directory, payload)
    run._write()
    return run


def log(step: int | None = None, **metrics: float) -> None:
    """Log to the implicit run, starting one on first call.

    The shortest useful thing a user can type is one line, and requiring a
    `start()` before it would make the two-line version the minimum.
    """
    global _implicit
    if _implicit is None:
        _implicit = start("notebook")
    _implicit.log(step, **metrics)


def current() -> "Run | None":
    return _implicit


def list(*, notebook_id: str | None = None) -> builtins.list[RunRef]:  # noqa: A001
    root = (
        _workspace.storage().notebooks / notebook_id / "runs"
        if notebook_id
        else _runs_root()
    )
    if not root.is_dir():
        return []
    found: builtins.list[RunRef] = []
    for directory in sorted(root.iterdir(), reverse=True):
        payload = _read(directory / RUN_FILE)
        if payload is None:
            continue
        found.append(
            RunRef(
                id=payload.get("id", directory.name),
                notebook_id=payload.get("notebook_id") or (notebook_id or ""),
                name=payload.get("name", ""),
                status=payload.get("status", "unknown"),
                started_at=payload.get("started_at"),
                finished_at=payload.get("finished_at"),
            )
        )
    return found


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@atexit.register
def _finish_implicit() -> None:
    """Close the implicit run when the kernel shuts down.

    Best-effort by nature — a `kill -9` reaches no handler — which is exactly
    why the metrics file is append-only and every line complete: the series survives
    even when this does not run.
    """
    if _implicit is not None and not _implicit._finished:
        try:
            _implicit.finish()
        except Exception:  # noqa: BLE001 - interpreter teardown must not raise
            pass
