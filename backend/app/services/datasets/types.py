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
    #: Visible from every project, owned by none. Tracked starter datasets set
    #: this; it replaces the old ``id.startswith("sample_")`` convention, which
    #: was duplicated across the dataset and evaluation services and silently
    #: dropped a dataset out of both the moment it was renamed.
    shared: bool = False
