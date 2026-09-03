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

`train`, `evaluate`, and `inference` are that rule taken to its conclusion: each
starts *the platform's own job*, so a run begun in a cell appears on
`/training`, registers a model the catalog knows about, and outlives the tab.
Training inside the kernel instead would produce a model nothing recorded, on a
process that dies when the notebook is closed.

One module per thing the platform does, so a cell reads as the sentence the user
was already thinking: load a dataset, train on it, evaluate the model, predict
with it.

**The dependency arrow points one way: `orinth` may import `app`; `app` must
never import `orinth`.** A backend that depended on the SDK would drag the
notebook stack into the API process, which is the thing the design is built to
avoid. `tests/test_orinth_sdk.py` asserts it.
"""

from orinth import datasets, evaluate, inference, models, projects, runs, settings, train
from orinth.errors import (
    DatasetBusyError,
    DatasetTooLargeError,
    OrinthError,
    ProjectTaskNotAllowed,
)
from orinth.types import (
    DatasetRef,
    Evaluation,
    ModelRef,
    Prediction,
    ProjectRef,
    Readiness,
    TrainingRun,
)

__version__ = "0.1.0"

__all__ = [
    "DatasetBusyError",
    "DatasetRef",
    "DatasetTooLargeError",
    "Evaluation",
    "ModelRef",
    "OrinthError",
    "Prediction",
    "ProjectRef",
    "ProjectTaskNotAllowed",
    "Readiness",
    "TrainingRun",
    "__version__",
    "datasets",
    "evaluate",
    "inference",
    "models",
    "project",
    "projects",
    "runs",
    "settings",
    "train",
]


def project():
    """The project this notebook belongs to, read from its manifest.

    A convenience over `projects.get(...)` because it is the single most common
    thing a cell needs and the notebook already knows the answer.
    """
    return projects.current()
