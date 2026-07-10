from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetLocation:
    id: str
    project_id: str
    name: str
    task_type: str
    format: str
    source: str
    root: Path
    editable: bool
    labels: list[str]
    metadata: dict
