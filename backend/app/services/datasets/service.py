import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from app.core.config import Settings
from app.core.defaults import DEFAULT_LABELS, DEFAULT_PROJECT_ID, DEFAULT_TASK_TYPE
from app.core.storage import Storage
from app.schemas import (
    DatasetCreate,
    DatasetImportRequest,
    DatasetSplitConfig,
    DatasetSummary,
    DatasetUpdate,
)
from app.services.datasets.constants import (
    IMAGE_TASK_TYPES,
    NLP_TASK_TYPES,
    SPLITS,
    TASK_TYPE_ALIASES,
)
from app.services.datasets.format_io import FormatIoMixin
from app.services.datasets.items import ItemsMixin
from app.services.datasets.preprocess import PreprocessMixin
from app.services.datasets.types import DatasetLocation
from app.services.datasets.versioning import VersioningMixin


class DatasetService(ItemsMixin, VersioningMixin, PreprocessMixin, FormatIoMixin):
    """Facade over dataset lifecycle, item CRUD, versioning, preprocessing, and format IO.

    The behavior lives in mixins split by concern (see ``items.py``, ``versioning.py``,
    ``preprocess.py``, ``format_io.py``); this class keeps only the core dataset
    lifecycle/location/manifest logic shared by all of them.
    """

    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage

    def list_datasets(self, project_id: str | None = None) -> list[DatasetSummary]:
        """Datasets *visible* to a project: its own, plus the shared read-only samples.

        Use :meth:`count_owned_datasets` for ownership questions — the shared samples
        are visible everywhere and must never count against a project.
        """
        summaries = [self.summary(location.id) for location in self._locations()]
        if project_id:
            summaries = [
                dataset
                for dataset in summaries
                if dataset.project_id == project_id or dataset.id.startswith("sample_")
            ]
        return summaries

    def count_owned_datasets(self, project_id: str) -> int:
        """Datasets a project actually owns and could delete.

        Excludes the shared ``sample_*`` datasets and the read-only reference
        datasets, which are surfaced in every project but belong to none.
        """
        return sum(
            1
            for location in self._locations()
            if location.project_id == project_id and location.editable
        )

    def summary(self, dataset_id: str) -> DatasetSummary:
        location = self._location(dataset_id)
        return DatasetSummary(
            id=location.id,
            project_id=location.project_id,
            name=location.name,
            task_type=location.task_type,  # type: ignore[arg-type]
            format=location.format,  # type: ignore[arg-type]
            source=location.source,  # type: ignore[arg-type]
            editable=location.editable,
            path=str(location.root),
            labels=location.labels,
            classes=location.labels,
            splits={split: self._split_summary(location, split) for split in SPLITS},
            metadata=location.metadata,
        )

    def create_dataset(self, payload: DatasetCreate) -> DatasetSummary:
        task_type = self._normalize_task_type(payload.task_type)
        if task_type not in IMAGE_TASK_TYPES | NLP_TASK_TYPES:
            raise HTTPException(status_code=400, detail="Unsupported dataset task type")
        labels = self._normalize_labels(payload.labels, task_type)
        dataset_id = self._new_dataset_id(payload.name)
        root = self.storage.datasets / dataset_id
        format_name = self._default_format_for_task(task_type, payload.format)
        self._create_layout(root, format_name, task_type, labels)
        self._write_manifest(
            root,
            dataset_id=dataset_id,
            name=payload.name,
            format_name=format_name,
            project_id=payload.project_id,
            task_type=task_type,
            labels=labels,
            metadata={
                "created_from": "empty",
                "split_config": DatasetSplitConfig().model_dump(mode="json"),
            },
        )
        return self.summary(dataset_id)

    def update_dataset(self, dataset_id: str, payload: DatasetUpdate) -> DatasetSummary:
        location = self._editable_location(dataset_id)
        metadata = dict(location.metadata or {})
        updates: dict = {}
        if payload.name is not None:
            updates["name"] = payload.name
        if payload.metadata is not None:
            metadata.update(payload.metadata)
        if payload.preprocess is not None:
            metadata["preprocess"] = self._normalize_preprocess_config(payload.preprocess).model_dump(
                mode="json"
            )
        if payload.metadata is not None or payload.preprocess is not None:
            updates["metadata"] = metadata
        if updates:
            self._update_manifest(location, **updates)
        return self.summary(dataset_id)

    def import_dataset(self, payload: DatasetImportRequest) -> DatasetSummary:
        source = Path(payload.path).expanduser().resolve()
        if not source.exists():
            raise HTTPException(status_code=404, detail=f"Dataset path not found: {source}")
        task_type = self._normalize_task_type(payload.task_type)
        if task_type in NLP_TASK_TYPES:
            return self._import_nlp_dataset(source, payload, task_type)
        if not self._is_yolo_root(source):
            raise HTTPException(status_code=400, detail="Only YOLO dataset imports are supported")

        labels = self._normalize_labels(
            payload.labels or self._labels_from_yolo_yaml(source) or DEFAULT_LABELS,
            task_type,
        )
        dataset_id = self._new_dataset_id(payload.name or source.name)
        root = self.storage.datasets / dataset_id
        shutil.copytree(source, root)
        self._create_layout(root, payload.format, task_type, labels)
        self._write_manifest(
            root,
            dataset_id=dataset_id,
            name=payload.name or source.name,
            format_name=payload.format,
            project_id=payload.project_id,
            task_type=task_type,
            labels=labels,
            metadata={"imported_from": str(source)},
        )
        return self.summary(dataset_id)

    def clone_dataset(
        self, dataset_id: str, name: str | None = None, project_id: str | None = None
    ) -> DatasetSummary:
        source = self._location(dataset_id)
        if source.format not in {"yolo", "image_folder", "image_manifest", "text_folder", "jsonl", "csv"}:
            raise HTTPException(status_code=400, detail="Only supported datasets can be cloned")

        new_name = name or f"{source.name} Copy"
        new_id = self._new_dataset_id(new_name)
        root = self.storage.datasets / new_id
        shutil.copytree(source.root, root)
        self._create_layout(root, source.format, source.task_type, source.labels)
        self._write_manifest(
            root,
            dataset_id=new_id,
            name=new_name,
            format_name=source.format,
            project_id=project_id or source.project_id,
            task_type=source.task_type,
            labels=source.labels,
            metadata={"cloned_from": source.id, "source_path": str(source.root)},
        )
        return self.summary(new_id)

    def delete_dataset(self, dataset_id: str) -> None:
        location = self._location(dataset_id)
        if not location.editable:
            raise HTTPException(status_code=409, detail="Reference datasets are read-only")
        self.storage.delete_owned_path(location.root)

    def dataset_root(self, dataset_id: str) -> Path:
        return self._location(dataset_id).root

    def training_root(self, dataset_id: str) -> Path:
        location = self._location(dataset_id)
        if location.format != "yolo":
            raise ValueError("YOLO training requires a YOLO dataset")
        if not (location.root / "train" / "images").exists() or not (
            location.root / "valid" / "images"
        ).exists():
            raise FileNotFoundError("YOLO dataset must include train and valid image splits")
        self._write_data_yaml(location.root, location.labels)
        return location.root

    def yolo_split(self, dataset_id: str, split: str) -> Path:
        self._validate_split(split)
        if split == "unassigned":
            raise ValueError("Unassigned split is not available for YOLO evaluation")
        location = self._location(dataset_id)
        if location.format != "yolo":
            raise ValueError("YOLO split requested for a non-YOLO dataset")
        return location.root / split

    def split_root(self, dataset_id: str, split: str) -> Path:
        self._validate_split(split)
        return self._location(dataset_id).root / split

    def labels_for_dataset_key(self, dataset_key: str) -> list[str]:
        if dataset_key == "yolo_test":
            return self._labels_from_yolo_yaml(self.settings.datasets_path / "dental dataset_yolov11_format") or DEFAULT_LABELS
        if dataset_key == "coco_test":
            return self._labels_from_coco(self.settings.datasets_path / "dental dataset_coco_format" / "test") or DEFAULT_LABELS
        if dataset_key.startswith("dataset:"):
            _, dataset_id, _split = dataset_key.split(":", 2)
            return self._location(dataset_id).labels
        return DEFAULT_LABELS

    def task_for_dataset_key(self, dataset_key: str) -> str:
        if dataset_key in {"yolo_test", "coco_test"}:
            return "segmentation"
        if dataset_key.startswith("dataset:"):
            _, dataset_id, _split = dataset_key.split(":", 2)
            return self._location(dataset_id).task_type
        return DEFAULT_TASK_TYPE

    def _locations(self) -> list[DatasetLocation]:
        locations = [
            DatasetLocation(
                id="reference_yolo",
                project_id=DEFAULT_PROJECT_ID,
                name="YOLO Reference Dataset",
                task_type="segmentation",
                format="yolo",
                source="reference",
                root=self.settings.datasets_path / "dental dataset_yolov11_format",
                editable=False,
                labels=self._labels_from_yolo_yaml(
                    self.settings.datasets_path / "dental dataset_yolov11_format"
                )
                or DEFAULT_LABELS,
                metadata=self._reference_metadata("dental dataset_yolov11_format"),
            ),
            DatasetLocation(
                id="reference_coco",
                project_id=DEFAULT_PROJECT_ID,
                name="COCO Reference Dataset",
                task_type="segmentation",
                format="coco",
                source="reference",
                root=self.settings.datasets_path / "dental dataset_coco_format",
                editable=False,
                labels=self._labels_from_coco(
                    self.settings.datasets_path / "dental dataset_coco_format" / "test"
                )
                or DEFAULT_LABELS,
                metadata=self._reference_metadata("dental dataset_coco_format"),
            ),
        ]
        locations.extend(self._sample_nlp_locations())
        if self.storage.datasets.exists():
            for manifest_path in sorted(self.storage.datasets.glob("*/manifest.json")):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, KeyError):
                    continue
                task_type = self._normalize_task_type(manifest.get("task_type", DEFAULT_TASK_TYPE))
                labels = self._normalize_labels(
                    manifest.get("labels") or manifest.get("classes") or self._default_labels_for_task(task_type),
                    task_type,
                )
                locations.append(
                    DatasetLocation(
                        id=manifest["id"],
                        project_id=manifest.get("project_id", DEFAULT_PROJECT_ID),
                        name=manifest["name"],
                        task_type=task_type,
                        format=manifest.get("format", "yolo"),
                        source="editable",
                        root=manifest_path.parent,
                        editable=True,
                        labels=labels,
                        metadata=manifest.get("metadata", {}),
                    )
                )
        return locations

    def _sample_nlp_locations(self) -> list[DatasetLocation]:
        root = self.settings.repo_root / "sample_data" / "nlp"
        specs = [
            ("sample_text_classification", "Sample Text Classification", "text_classification", "text_classification", ["positive", "negative", "neutral"]),
            ("sample_summarization", "Sample Summarization", "summarization", "summarization", ["summary"]),
            ("sample_question_answering", "Sample Question Answering", "question_answering", "question_answering", ["answer"]),
        ]
        locations = []
        for dataset_id, name, dirname, task_type, labels in specs:
            dataset_root = root / dirname
            if not dataset_root.exists():
                continue
            locations.append(
                DatasetLocation(
                    id=dataset_id,
                    project_id=DEFAULT_PROJECT_ID,
                    name=name,
                    task_type=task_type,
                    format="text_folder",
                    source="reference",
                    root=dataset_root,
                    editable=False,
                    labels=labels,
                    metadata={"created_from": "tracked_sample"},
                )
            )
        return locations

    def _location(self, dataset_id: str) -> DatasetLocation:
        for location in self._locations():
            if location.id == dataset_id:
                return location
        raise HTTPException(status_code=404, detail="Dataset not found")

    def _editable_location(self, dataset_id: str) -> DatasetLocation:
        location = self._location(dataset_id)
        if not location.editable:
            raise HTTPException(status_code=409, detail="Reference datasets are read-only")
        return location

    def _reference_metadata(self, dirname: str) -> dict:
        root = self.settings.datasets_path / dirname
        yaml_path = root / "data.yaml"
        metadata = {}
        if yaml_path.exists():
            for line in yaml_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("workspace:"):
                    metadata["workspace"] = line.split(":", 1)[1].strip()
                if line.strip().startswith("project:"):
                    metadata["source_project"] = line.split(":", 1)[1].strip()
                if line.strip().startswith("version:"):
                    metadata["version"] = line.split(":", 1)[1].strip()
        return metadata

    def _create_layout(self, root: Path, format_name: str, task_type: str, labels: list[str]) -> None:
        for split in SPLITS:
            media_dir = "texts" if self._is_nlp_task(task_type) else "images"
            (root / split / media_dir).mkdir(parents=True, exist_ok=True)
            (root / split / "annotations").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
        if format_name == "yolo" and not self._is_nlp_task(task_type):
            self._write_data_yaml(root, labels)

    def _write_manifest(
        self,
        root: Path,
        *,
        dataset_id: str,
        name: str,
        format_name: str,
        project_id: str,
        task_type: str,
        labels: list[str],
        metadata: dict,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "id": dataset_id,
                    "project_id": project_id,
                    "name": name,
                    "task_type": task_type,
                    "format": format_name,
                    "labels": labels,
                    "metadata": metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _update_manifest(self, location: DatasetLocation, **updates) -> None:
        manifest_path = location.root / "manifest.json"
        manifest = {
            "id": location.id,
            "project_id": location.project_id,
            "name": location.name,
            "task_type": location.task_type,
            "format": location.format,
            "labels": location.labels,
            "metadata": location.metadata,
        }
        if manifest_path.exists():
            manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest.update(updates)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _is_nlp_task(self, task_type: str) -> bool:
        return self._normalize_task_type(task_type) in NLP_TASK_TYPES

    def _normalize_task_type(self, task_type: str) -> str:
        return TASK_TYPE_ALIASES.get(str(task_type), str(task_type))

    def _default_format_for_task(self, task_type: str, format_name: str) -> str:
        if task_type in NLP_TASK_TYPES and format_name in {"yolo", "image_folder", "image_manifest"}:
            return "text_folder"
        return format_name

    def _default_labels_for_task(self, task_type: str) -> list[str]:
        if task_type == "text_classification":
            return ["positive", "negative", "neutral"]
        if task_type == "summarization":
            return ["summary"]
        if task_type == "question_answering":
            return ["answer"]
        return DEFAULT_LABELS.copy()

    def _normalize_labels(self, labels: list[str], task_type: str = DEFAULT_TASK_TYPE) -> list[str]:
        normalized = []
        for label in labels or []:
            cleaned = self._clean_label(label)
            if cleaned not in normalized:
                normalized.append(cleaned)
        return normalized or self._default_labels_for_task(task_type)

    def _new_dataset_id(self, name: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
        slug = "-".join(part for part in slug.split("-") if part)[:48] or "dataset"
        return f"{slug}-{uuid4().hex[:8]}"

    def _clean_label(self, label: str) -> str:
        cleaned = " ".join(label.strip().split())
        if not cleaned:
            raise HTTPException(status_code=400, detail="Label cannot be empty")
        return cleaned[:80]

    def _validate_split(self, split: str) -> None:
        if split not in SPLITS:
            raise HTTPException(status_code=400, detail="Unknown dataset split")
