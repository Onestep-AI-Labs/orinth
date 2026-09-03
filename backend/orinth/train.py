"""Start and watch a training run, from a cell.

Over HTTP, not in-process. Training is a job: the API validates the request
against the project gate, writes a `TrainingJob` row, and hands it to the
executor that the Training page is already watching. A notebook that trained in
its own kernel instead would produce a model the platform never recorded, on a
process that dies with the tab — which is the opposite of what the Training page
exists to guarantee.

So a run started here is *the same run*. It appears on `/training`, streams the
same logs, registers the same model, and survives the notebook being closed.
The one thing this adds is that you can compute the inputs in Python first.
"""

import builtins
import time
from collections.abc import Mapping
from typing import Any

from orinth import _http, _workspace
from orinth.errors import OrinthError
from orinth.types import TrainingRun

#: How often `wait()` asks, and how long it is willing to. Training is measured
#: in minutes to hours, so a two-second poll is already generous; the ceiling is
#: there so a wedged job cannot hang a cell forever.
POLL_SECONDS = 2.0
POLL_TIMEOUT = 24 * 60 * 60.0


def _run(payload: dict) -> TrainingRun:
    progress = payload.get("progress") or {}
    return TrainingRun(
        id=payload.get("id", ""),
        project_id=payload.get("project_id", ""),
        task_type=payload.get("task_type", ""),
        model_family=payload.get("model_family", ""),
        status=payload.get("status", ""),
        percent=float(progress.get("percent") or 0.0),
        step=str(progress.get("current_step") or ""),
        metrics=dict(payload.get("metrics") or {}),
        model_id=payload.get("promoted_model_id"),
        error=payload.get("error"),
    )


def options(task_type: str | None = None) -> builtins.list[dict]:
    """What can be trained, and with what defaults.

    Returned as plain dicts rather than a frozen type: an option carries a
    per-family `defaults` and `advanced_parameters` payload whose shape is the
    model catalog's business, and freezing a view of it here would go stale the
    first time a family gains a knob.
    """
    query = {"task_type": task_type} if task_type else None
    return builtins.list(_http.get("/api/training/model-options", params=query) or [])


def compute() -> dict:
    """What this machine can train on, per framework.

    Per framework because they disagree — Apple silicon without
    `tensorflow-metal` gives torch a GPU and TensorFlow a CPU. Probed out of
    process by the backend and cached there, so asking is cheap.
    """
    return _http.get("/api/training/compute") or {}


def start(
    dataset_id: str,
    *,
    model_option_id: str,
    task_type: str,
    project_id: str | None = None,
    name: str | None = None,
    epochs: int = 50,
    image_size: int = 512,
    batch_size: int = 16,
    learning_rate: float = 0.002,
    optimizer: str = "AdamW",
    device: str = "",
    patience: int = 50,
    hyperparameters: Mapping[str, Any] | None = None,
    wait: bool = False,
) -> TrainingRun:
    """Queue a training job and return it immediately, or when it finishes.

    `wait=False` by default, which is the opposite of `datasets.register()`.
    Registering a dataset takes seconds and the id is useless until it is done;
    training takes minutes to hours, and blocking a cell on it costs you the
    kernel for the duration. Start it, keep working, and call `wait()` when you
    actually need the model.
    """
    payload = {
        "project_id": project_id or _workspace.default_project_id(),
        "task_type": task_type,
        "model_option_id": model_option_id,
        "model_name": name,
        "dataset_id": dataset_id,
        "epochs": int(epochs),
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "optimizer": optimizer,
        "device": device,
        "patience": int(patience),
        "hyperparameters": dict(hyperparameters or {}),
    }
    try:
        created = _http.post("/api/training/jobs", json=payload)
    except OrinthError as error:
        _http.raise_for_project(error, payload["project_id"], task_type)
        raise  # pragma: no cover - raise_for_project always raises
    run = _run(created or {})
    return wait_for(run.id) if wait else run


def get(job_id: str) -> TrainingRun:
    return _run(_http.get(f"/api/training/jobs/{job_id}") or {})


def list(  # noqa: A001 - matches `orinth.datasets.list`; the builtin is one attribute away
    *, project_id: str | None = None, status: str | None = None
) -> builtins.list[TrainingRun]:
    params: dict[str, str] = {"project_id": project_id or _workspace.default_project_id()}
    runs = [_run(entry) for entry in _http.get("/api/training/jobs", params=params) or []]
    return [run for run in runs if status is None or run.status == status]


def cancel(job_id: str) -> TrainingRun:
    return _run(_http.post(f"/api/training/jobs/{job_id}/cancel") or {})


def wait_for(job_id: str, *, on_progress=None) -> TrainingRun:
    """Block until the job reaches a terminal state, and hand back the last read.

    A failed run is *returned*, not raised. It is a real outcome with metrics and
    an error message attached, and a notebook comparing three runs should not
    lose the other two to an exception from the one that diverged.
    """
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        run = get(job_id)
        if on_progress is not None:
            on_progress(run)
        if run.done:
            return run
        time.sleep(POLL_SECONDS)
    raise OrinthError(
        f"Timed out waiting for training job '{job_id}'. "
        "It is still running — watch it on the Training page."
    )
