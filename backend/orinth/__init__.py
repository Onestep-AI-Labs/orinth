"""`import orinth` — the platform's data, one import away, inside a kernel.

This package exists so a notebook does not have to know where the workspace
lives. It runs in the kernel process, which is not the API process, so it is
free to import polars, PIL, or a predictor that pulls TensorFlow — the whole
reason the kernel is a separate process.

**Reads go direct; writes go over HTTP.** Reading a dataset means opening
thousands of files, and routing that through uvicorn would be slow and would put
the read on the thread pool serving the studio. Writing means creating a
dataset, and that has to run the project gate, the atomic manifest write, and the
readiness computation exactly once — which is what the API already does. So
`datasets.load()` reads `storage/` through `Settings`, and `datasets.register()`
posts to `/api/datasets/ingest` and `/prep/apply` like any other client.

**The dependency arrow points one way: `orinth` may import `app`; `app` must
never import `orinth`.** A backend that depended on the SDK would drag the
notebook stack into the API process, which is the thing the design is built to
avoid. `tests/test_orinth_sdk.py` asserts it.
"""

from orinth import datasets, models, projects, runs, settings
from orinth.errors import (
    DatasetBusyError,
    DatasetTooLargeError,
    OrinthError,
    ProjectTaskNotAllowed,
)
from orinth.types import DatasetRef, ModelRef, ProjectRef, Readiness

__version__ = "0.1.0"

__all__ = [
    "DatasetBusyError",
    "DatasetRef",
    "DatasetTooLargeError",
    "ModelRef",
    "OrinthError",
    "ProjectRef",
    "ProjectTaskNotAllowed",
    "Readiness",
    "__version__",
    "datasets",
    "models",
    "project",
    "projects",
    "runs",
    "settings",
]


def project():
    """The project this notebook belongs to, read from its manifest.

    A convenience over `projects.get(...)` because it is the single most common
    thing a cell needs and the notebook already knows the answer.
    """
    return projects.current()
