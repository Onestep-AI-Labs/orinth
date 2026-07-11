import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.defaults import DEFAULT_LABELS, DEFAULT_PROJECT_ID
from app.core.config import Settings
from app.core.storage import Storage
from app.ml.predictors.base import Predictor
from app.ml.predictors.keras_classification import KerasClassificationPredictor
from app.ml.predictors.unet_inception import UnetInceptionPredictor
from app.ml.predictors.yolo import YoloPredictor
from app.schemas import ModelInfo


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: str
    description: str
    paths: dict[str, Path]
    promoted: bool = False
    project_id: str = DEFAULT_PROJECT_ID
    task_type: str = "segmentation"
    labels: list[str] | None = None
    source: str = "reference"
    training_job_id: str | None = None
    created_at: datetime | None = None
    metrics: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return all(path.exists() for path in self.paths.values())

    def to_info(self) -> ModelInfo:
        return ModelInfo(
            id=self.id,
            name=self.name,
            family=self.family,  # type: ignore[arg-type]
            description=self.description,
            available=self.available,
            paths={key: str(path) for key, path in self.paths.items()},
            promoted=self.promoted,
            project_id=self.project_id,
            task_type=self.task_type,  # type: ignore[arg-type]
            labels=(self.labels or DEFAULT_LABELS).copy(),
            source=self.source,  # type: ignore[arg-type]
            training_job_id=self.training_job_id,
            created_at=self.created_at,
            metrics=self.metrics or {},
            artifacts=self.artifacts or {},
        )


class ModelRegistry:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage
        self._predictors: dict[str, Predictor] = {}

    def list_specs(self) -> list[ModelSpec]:
        models_dir = self.settings.models_path
        specs = [
            ModelSpec(
                id="yolo_11_best",
                name="YOLOv11 Best",
                family="yolo",
                description="Single-stage YOLOv11 segmentation model.",
                paths={"weights": models_dir / "yolo_11_best" / "weights" / "best.pt"},
                labels=DEFAULT_LABELS.copy(),
            ),
            ModelSpec(
                id="unet_inception",
                name="U-Net + Inception",
                family="unet_inception",
                description="Two-stage U-Net lesion segmentation plus Inception classification.",
                paths={
                    "unet": models_dir / "unet_inception" / "best_unet_model.keras",
                    "classifier": models_dir
                    / "unet_inception"
                    / "best_classifier_inception.keras",
                },
                labels=DEFAULT_LABELS.copy(),
            ),
        ]
        specs.extend(self._load_promoted_specs())
        return specs

    def get_spec(self, model_id: str) -> ModelSpec:
        for spec in self.list_specs():
            if spec.id == model_id:
                return spec
        raise KeyError(model_id)

    def list_models(
        self, project_id: str | None = None, task_type: str | None = None
    ) -> list[ModelInfo]:
        models = [spec.to_info() for spec in self.list_specs()]
        if project_id:
            models = [
                model
                for model in models
                if model.project_id in {project_id, DEFAULT_PROJECT_ID}
                or model.source == "reference"
            ]
        if task_type:
            models = [model for model in models if model.task_type == task_type]
        return models

    def get_predictor(self, model_id: str) -> Predictor:
        spec = self.get_spec(model_id)
        if not spec.available:
            missing = [str(path) for path in spec.paths.values() if not path.exists()]
            raise FileNotFoundError(f"Model assets are missing for {model_id}: {missing}")
        if model_id not in self._predictors:
            if spec.family == "yolo":
                self._predictors[model_id] = YoloPredictor(
                    spec.paths["weights"],
                    spec.labels or DEFAULT_LABELS,
                )
            elif spec.family == "unet_inception":
                self._predictors[model_id] = UnetInceptionPredictor(
                    spec.paths["unet"],
                    spec.paths["classifier"],
                )
            elif spec.family == "keras_classification":
                self._predictors[model_id] = KerasClassificationPredictor(
                    spec.paths["model"],
                    spec.labels or DEFAULT_LABELS,
                    image_size=int((spec.artifacts or {}).get("image_size") or 224),
                )
            else:
                raise ValueError(f"Unsupported model family: {spec.family}")
        return self._predictors[model_id]

    def register_model(
        self,
        *,
        model_id: str,
        name: str,
        family: str,
        task_type: str,
        paths: dict[str, Path],
        labels: list[str],
        project_id: str = DEFAULT_PROJECT_ID,
        description: str = "Trained model artifact.",
        training_job_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        source: str = "trained",
    ) -> ModelInfo:
        registry = self._read_registry()
        registry[model_id] = {
            "id": model_id,
            "name": name,
            "family": family,
            "description": description,
            "paths": {key: str(path.resolve()) for key, path in paths.items()},
            "promoted": True,
            "project_id": project_id,
            "task_type": task_type,
            "labels": labels,
            "source": source,
            "training_job_id": training_job_id,
            "created_at": datetime.utcnow().isoformat(),
            "metrics": metrics or {},
            "artifacts": artifacts or {},
        }
        self.storage.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.storage.registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        self._predictors.pop(model_id, None)
        return self.get_spec(model_id).to_info()

    def update_model(self, model_id: str, *, name: str) -> ModelInfo:
        registry = self._read_registry()
        if model_id not in registry:
            raise KeyError(model_id)
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Model name is required")
        registry[model_id]["name"] = clean_name
        self._write_registry(registry)
        return self.get_spec(model_id).to_info()

    def delete_model(self, model_id: str) -> bool:
        registry = self._read_registry()
        item = registry.pop(model_id, None)
        if item is None:
            raise KeyError(model_id)
        self._write_registry(registry)
        model_dir = self.storage.trained_models / model_id
        deleted_storage = self.storage.delete_owned_path(model_dir)
        if not deleted_storage:
            for raw_path in (item.get("paths") or {}).values():
                self.storage.delete_owned_path(raw_path)
        self._predictors.pop(model_id, None)
        return True

    def promote_yolo_model(
        self,
        model_id: str,
        weights_path: Path,
        name: str,
        *,
        labels: list[str] | None = None,
        project_id: str = DEFAULT_PROJECT_ID,
        training_job_id: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> ModelInfo:
        return self.register_model(
            model_id=model_id,
            name=name,
            family="yolo",
            task_type="segmentation",
            paths={"weights": weights_path},
            labels=labels or DEFAULT_LABELS,
            project_id=project_id,
            description="Promoted YOLO training artifact.",
            training_job_id=training_job_id,
            metrics=metrics,
            source="promoted",
        )

    def _load_promoted_specs(self) -> list[ModelSpec]:
        registry = self._read_registry()
        specs = []
        for item in registry.values():
            paths = {key: Path(value) for key, value in item.get("paths", {}).items()}
            created_at = None
            if item.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(item["created_at"])
                except ValueError:
                    created_at = None
            specs.append(
                ModelSpec(
                    id=item["id"],
                    name=item["name"],
                    family=item["family"],
                    description=item.get("description", "Promoted model"),
                    paths=paths,
                    promoted=bool(item.get("promoted", True)),
                    project_id=item.get("project_id", DEFAULT_PROJECT_ID),
                    task_type=item.get("task_type", "segmentation"),
                    labels=item.get("labels") or DEFAULT_LABELS.copy(),
                    source=item.get("source", "promoted"),
                    training_job_id=item.get("training_job_id"),
                    created_at=created_at,
                    metrics=item.get("metrics") or {},
                    artifacts=item.get("artifacts") or {},
                )
            )
        return specs

    def _read_registry(self) -> dict[str, Any]:
        if not self.storage.registry_file.exists():
            return {}
        return json.loads(self.storage.registry_file.read_text(encoding="utf-8"))

    def _write_registry(self, registry: dict[str, Any]) -> None:
        self.storage.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.storage.registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
