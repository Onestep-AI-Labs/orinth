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
    #: Manifest provenance (see phase 10). ``created`` for user-made datasets and
    #: legacy manifests without the field, ``imported_hf`` for HuggingFace Hub
    #: imports, ``recipe`` for document-generated datasets (phase 11). Groups the
    #: catalog without an id/name convention that would break on rename.
    origin: str = "created"
    #: For imports, the source reference (e.g. ``hub_id@revision``); ``None`` otherwise.
    origin_ref: str | None = None
