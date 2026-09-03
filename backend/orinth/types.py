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
