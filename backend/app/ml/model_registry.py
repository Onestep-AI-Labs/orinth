import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.storage import Storage
from app.ml.predictors.base import Predictor
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
            ),
        ]
        specs.extend(self._load_promoted_specs())
        return specs

    def get_spec(self, model_id: str) -> ModelSpec:
        for spec in self.list_specs():
            if spec.id == model_id:
                return spec
        raise KeyError(model_id)

    def list_models(self) -> list[ModelInfo]:
        return [spec.to_info() for spec in self.list_specs()]

    def get_predictor(self, model_id: str) -> Predictor:
        spec = self.get_spec(model_id)
        if not spec.available:
            missing = [str(path) for path in spec.paths.values() if not path.exists()]
            raise FileNotFoundError(f"Model assets are missing for {model_id}: {missing}")
        if model_id not in self._predictors:
            if spec.family == "yolo":
                self._predictors[model_id] = YoloPredictor(spec.paths["weights"])
            elif spec.family == "unet_inception":
                self._predictors[model_id] = UnetInceptionPredictor(
                    spec.paths["unet"],
                    spec.paths["classifier"],
                )
            else:
                raise ValueError(f"Unsupported model family: {spec.family}")
        return self._predictors[model_id]

    def promote_yolo_model(self, model_id: str, weights_path: Path, name: str) -> ModelInfo:
        registry = self._read_registry()
        registry[model_id] = {
            "id": model_id,
            "name": name,
            "family": "yolo",
            "description": "Promoted YOLO training artifact.",
            "paths": {"weights": str(weights_path.resolve())},
            "promoted": True,
        }
        self.storage.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.storage.registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        self._predictors.pop(model_id, None)
        return self.get_spec(model_id).to_info()

    def _load_promoted_specs(self) -> list[ModelSpec]:
        registry = self._read_registry()
        specs = []
        for item in registry.values():
            paths = {key: Path(value) for key, value in item.get("paths", {}).items()}
            specs.append(
                ModelSpec(
                    id=item["id"],
                    name=item["name"],
                    family=item["family"],
                    description=item.get("description", "Promoted model"),
                    paths=paths,
                    promoted=bool(item.get("promoted", True)),
                )
            )
        return specs

    def _read_registry(self) -> dict[str, Any]:
        if not self.storage.registry_file.exists():
            return {}
        return json.loads(self.storage.registry_file.read_text(encoding="utf-8"))
