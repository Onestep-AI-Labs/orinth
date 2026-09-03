"""Failures the SDK raises, each naming what to do instead.

Every one of these is a case where returning something plausible would be worse
than raising: half a training set, a frame read while the prep agent is
rewriting the directory under it, or a dataset created in a project that cannot
hold its task type.
"""


class OrinthError(RuntimeError):
    """Base for everything this package raises deliberately."""


class DatasetTooLargeError(OrinthError):
    """`load()` refused rather than truncating.

    Silently returning the first 200,000 rows of a larger dataset is the worse
    failure — the frame looks complete and every number computed from it is
    quietly wrong. `MAX_INGEST_ROWS` in the prep agent set this precedent.
    """

    def __init__(self, rows: int, cap: int) -> None:
        super().__init__(
            f"{rows:,} rows is over the {cap:,} row cap for load(). "
            "Pass limit=N for a sample, or stream with records() / images(), "
            "which never hold the whole split."
        )
        self.rows = rows
        self.cap = cap


class DatasetBusyError(OrinthError):
    """The prep agent is mid-apply, so the split directories are being rewritten.

    Only `applying` qualifies — detect and plan read `_staging/` and leave the
    dataset exactly as it was, which is the same distinction readiness draws.
    """

    def __init__(self, dataset_id: str) -> None:
        super().__init__(
            f"Orinth is rewriting '{dataset_id}' right now. "
            "Wait for prep to finish, then read it again."
        )
        self.dataset_id = dataset_id


class ProjectTaskNotAllowed(OrinthError):
    """The target project does not declare this task type.

    Surfaced from the API's own 409 rather than pre-checked here, so the SDK and
    the browser cannot disagree about what a project allows.
    """

    def __init__(self, project_id: str, task_type: str) -> None:
        super().__init__(
            f"Project '{project_id}' does not allow task '{task_type}'. "
            "Add it under project settings, or pass project_id= for one that does."
        )
        self.project_id = project_id
        self.task_type = task_type
