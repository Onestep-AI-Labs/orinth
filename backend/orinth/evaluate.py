"""Score a registered model against a split, through the platform's own testing job.

The difference from computing metrics by hand in a cell — which the
**Evaluate a model on a split** template also shows — is that this one is
*recorded*. It appears on `/testing`, it is comparable against every other
evaluation of the same dataset, and the per-image rows are on disk afterwards.

Do it by hand when you are inventing a metric. Do it here when you want the
number the platform will quote.
"""

import builtins
import time
from collections.abc import Sequence

from orinth import _http, _workspace
from orinth.errors import OrinthError
from orinth.types import Evaluation

POLL_SECONDS = 2.0
POLL_TIMEOUT = 2 * 60 * 60.0


def _evaluation(payload: dict) -> Evaluation:
    progress = payload.get("progress") or {}
    return Evaluation(
        id=payload.get("id", ""),
        model_id=payload.get("model_id", ""),
        dataset_key=payload.get("dataset_key", ""),
        status=payload.get("status", ""),
        percent=float(progress.get("percent") or 0.0),
        metrics=dict(payload.get("metrics") or {}),
        error=payload.get("error"),
    )


def datasets(*, project_id: str | None = None, task_type: str | None = None) -> builtins.list[dict]:
    """The splits that can be evaluated, keyed the way `start()` wants them.

    A `dataset_key` is not a dataset id: it names a dataset *and a split*, which
    is why this exists rather than letting the caller build one. Reading the
    keys from the API is also what keeps a notebook from evaluating against a
    split the platform cannot resolve.
    """
    params = {"project_id": project_id or _workspace.default_project_id()}
    if task_type:
        params["task_type"] = task_type
    return builtins.list(_http.get("/api/testing/datasets", params=params) or [])


def start(
    model_id: str,
    dataset_key: str,
    *,
    project_id: str | None = None,
    limit: int | None = None,
    wait: bool = True,
) -> Evaluation:
    """Queue one evaluation. Waits by default — this is minutes, not hours."""
    payload = {
        "project_id": project_id or _workspace.default_project_id(),
        "model_id": model_id,
        "dataset_key": dataset_key,
        "limit": limit,
    }
    created = _evaluation(_http.post("/api/testing/jobs", json=payload) or {})
    return wait_for(created.id) if wait else created


def compare(
    model_ids: Sequence[str],
    dataset_key: str,
    *,
    project_id: str | None = None,
    limit: int | None = None,
    wait: bool = True,
) -> builtins.list[Evaluation]:
    """One evaluation per model, over the same split, in one comparison.

    Batched through the API's own `/jobs/batch` rather than looped here, because
    that is what groups them under a `comparison_id` — a loop would produce N
    unrelated jobs the Testing page cannot line up side by side.
    """
    payload = {
        "project_id": project_id or _workspace.default_project_id(),
        "model_ids": builtins.list(model_ids),
        "dataset_key": dataset_key,
        "limit": limit,
    }
    created = [_evaluation(entry) for entry in _http.post("/api/testing/jobs/batch", json=payload) or []]
    if not wait:
        return created
    return [wait_for(job.id) for job in created]


def get(job_id: str) -> Evaluation:
    return _evaluation(_http.get(f"/api/testing/jobs/{job_id}") or {})


def list(  # noqa: A001 - matches `orinth.datasets.list`
    *, project_id: str | None = None, model_id: str | None = None
) -> builtins.list[Evaluation]:
    params = {"project_id": project_id or _workspace.default_project_id()}
    found = [_evaluation(entry) for entry in _http.get("/api/testing/jobs", params=params) or []]
    return [job for job in found if model_id is None or job.model_id == model_id]


def per_item(job_id: str) -> builtins.list[dict]:
    """The per-image rows behind an evaluation's headline metrics.

    This is where a bad number becomes actionable: the aggregate says 0.72, and
    these say *which* items it got wrong.
    """
    return builtins.list(_http.get(f"/api/testing/jobs/{job_id}/per-image") or [])


def wait_for(job_id: str, *, on_progress=None) -> Evaluation:
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        job = get(job_id)
        if on_progress is not None:
            on_progress(job)
        if job.done:
            return job
        time.sleep(POLL_SECONDS)
    raise OrinthError(
        f"Timed out waiting for evaluation '{job_id}'. Check the Testing page."
    )
