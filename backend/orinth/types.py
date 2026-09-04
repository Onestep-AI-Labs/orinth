"""Frozen handles the SDK hands back.

Dataclasses rather than the API's Pydantic models on purpose. A notebook user
tab-completes these, prints them, and puts them in a dict; a Pydantic model
carries validation machinery they will never use and a `.model_dump()` they have
to learn. These are plain, frozen, and `repr` readably.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Readiness:
    """Whether a dataset can be trained on — phase 21's verdict, verbatim."""

    state: str
    trainable: bool
    summary: str
    busy: bool = False
    checks: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        """`if orinth.datasets.readiness(id):` reads as intended."""
        return self.trainable


@dataclass(frozen=True)
class DatasetRef:
    id: str
    project_id: str
    name: str
    task_type: str
    format: str
    labels: list[str]
    #: split -> item_count
    splits: dict[str, int]
    #: The dataset root on disk. Present because a notebook legitimately wants
    #: to point another tool at it; nothing in the SDK requires the caller to
    #: use it.
    path: Path
    readiness: Readiness

    @property
    def total_items(self) -> int:
        return sum(self.splits.values())


@dataclass(frozen=True)
class ModelRef:
    id: str
    name: str
    family: str
    task_type: str
    source: str
    path: Path | None
    available: bool


@dataclass(frozen=True)
class ProjectRef:
    id: str
    name: str
    task_types: list[str]


@dataclass(frozen=True)
class RunRef:
    id: str
    notebook_id: str
    name: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None


@dataclass(frozen=True)
class TrainingRun:
    """One training job, as the platform's own Training page sees it.

    Carries `model_id` because that is the question a notebook asks next: a
    finished run has registered a model, and everything downstream —
    `evaluate.start()`, `inference.predict()`, `models.predictor()` — takes that
    id and not this one.
    """

    id: str
    project_id: str
    task_type: str
    model_family: str
    status: str
    percent: float
    step: str
    metrics: dict
    model_id: str | None
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.status in {"completed", "failed", "canceled"}

    def __bool__(self) -> bool:
        """`if run:` means it finished and produced something."""
        return self.status == "completed"


@dataclass(frozen=True)
class Evaluation:
    """One evaluation job. `metrics` is per task type — see the Testing page."""

    id: str
    model_id: str
    dataset_key: str
    status: str
    percent: float
    metrics: dict
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.status in {"completed", "failed", "canceled"}

    def __bool__(self) -> bool:
        return self.status == "completed"


@dataclass(frozen=True)
class Prediction:
    """One inference result, recorded by the platform exactly as the studio does.

    `label` is the image-level verdict, `detections` the raw boxes and polygons,
    `scores` the class probabilities where the model produces them. A text model
    fills `text` with its own result shape — label + scores, a summary, or an
    answer — and leaves the geometry empty.
    """

    id: str
    model_id: str
    label: str
    scores: dict[str, float]
    detections: list[dict]
    text: dict | None = None
    overlay_url: str | None = None
