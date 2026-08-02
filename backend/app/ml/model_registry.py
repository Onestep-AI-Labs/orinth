import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from app.core.config import Settings
from app.core.defaults import DEFAULT_LABELS, DEFAULT_PROJECT_ID
from app.core.storage import Storage
from app.ml.common.base import Predictor
from app.ml.nlp.baseline.predictors import (
    ExtractiveSummarizerPredictor,
    KeywordQAPredictor,
    TextClassificationPredictor,
)
from app.ml.nlp.huggingface.predictors import (
    HuggingFaceQAPredictor,
    HuggingFaceSummarizerPredictor,
    HuggingFaceTextClassificationPredictor,
)
from app.ml.nlp.keras_classifier import KerasTextClassificationPredictor
from app.ml.nlp.seq2seq.predictor import KerasSeq2SeqSummarizerPredictor
from app.ml.vision.keras_classification.predictor import KerasClassificationPredictor
from app.ml.vision.unet_inception.predictor import UnetInceptionPredictor
from app.ml.vision.yolo.predictor import YoloPredictor
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
    base_model_id: str | None = None
    format: str | None = None
    created_at: datetime | None = None
    metrics: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        return all(path.exists() for path in self.paths.values())

    def size_on_disk(self) -> int | None:
        total = 0
        found = False
        for path in self.paths.values():
            if not path.exists():
                continue
            found = True
            if path.is_dir():
                for nested in path.rglob("*"):
                    if nested.is_file():
                        total += nested.stat().st_size
            else:
                total += path.stat().st_size
        return total if found else None

    def to_info(self) -> ModelInfo:
        # LLM artifacts are label-free (a fine-tune has no class list); falling
        # back to the medical DEFAULT_LABELS here is what made an LLM adapter
        # card render "granuloma / kista". Keep their label list empty.
        is_llm = self.family in {"llm_hf", "llm_adapter", "llm_gguf"}
        display_labels = list(self.labels or ([] if is_llm else DEFAULT_LABELS))
        return ModelInfo(
            id=self.id,
            name=self.name,
            family=self.family,
            description=self.description,
            available=self.available,
            paths={key: str(path) for key, path in self.paths.items()},
            promoted=self.promoted,
            project_id=self.project_id,
            task_type=self.task_type,  # type: ignore[arg-type]
            labels=display_labels,
            source=self.source,  # type: ignore[arg-type]
            training_job_id=self.training_job_id,
            base_model_id=self.base_model_id,
            format=self.format,
            size_bytes=self.size_on_disk(),
            created_at=self.created_at,
            metrics=self.metrics or {},
            artifacts=self.artifacts or {},
        )


class ModelRegistry:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage
        self._predictors: dict[str, Predictor] = {}
        # Re-entrant because registry methods call each other (e.g.
        # register_model -> get_spec -> list_specs -> _load_promoted_specs)
        # and get_predictor uses it for double-checked cache access too.
        self._lock = threading.RLock()

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
            ModelSpec(
                id="keyword_text_classifier",
                name="Keyword Text Classifier",
                family="nlp_text_classification",
                description="Offline keyword and TF-IDF compatible baseline for text classification.",
                paths={},
                labels=["positive", "negative", "neutral"],
                task_type="text_classification",
            ),
            ModelSpec(
                id="extractive_summarizer",
                name="Extractive Summarizer",
                family="nlp_summarization",
                description="Offline extractive baseline for text summarization.",
                paths={},
                labels=["summary"],
                task_type="summarization",
            ),
            ModelSpec(
                id="keyword_qa",
                name="Keyword QA",
                family="nlp_qa",
                description="Offline keyword-overlap baseline for question answering.",
                paths={},
                labels=["answer"],
                task_type="question_answering",
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
                if model.project_id == project_id
                or (model.source == "reference" and model.family.startswith("nlp_"))
            ]
        if task_type:
            models = [model for model in models if model.task_type == task_type]
        return models

    def get_predictor(self, model_id: str) -> Predictor:
        spec = self.get_spec(model_id)
        if not spec.available:
            missing = [str(path) for path in spec.paths.values() if not path.exists()]
            raise FileNotFoundError(f"Model assets are missing for {model_id}: {missing}")

        with self._lock:
            cached = self._predictors.get(model_id)
        if cached is not None:
            return cached

        # Heavy model construction (loading weights, building TF/Torch
        # graphs, etc.) happens outside the lock so it never blocks other
        # registry operations or other predictor lookups. If construction
        # raises, nothing is written to the cache.
        predictor = self._construct_predictor(spec)

        with self._lock:
            # Double-checked: another thread may have constructed and
            # cached a predictor for this model_id while we were building
            # ours. Keep whichever was cached first and discard the rest.
            existing = self._predictors.get(model_id)
            if existing is not None:
                return existing
            self._predictors[model_id] = predictor
            return predictor

    def _construct_predictor(self, spec: ModelSpec) -> Predictor:
        if spec.family == "yolo":
            return YoloPredictor(
                spec.paths["weights"],
                spec.labels or DEFAULT_LABELS,
            )
        if spec.family == "unet_inception":
            return UnetInceptionPredictor(
                spec.paths["unet"],
                spec.paths["classifier"],
            )
        if spec.family == "keras_classification":
            return KerasClassificationPredictor(
                spec.paths["model"],
                spec.labels or DEFAULT_LABELS,
                image_size=int((spec.artifacts or {}).get("image_size") or 224),
            )
        if spec.family == "nlp_text_classification":
            return TextClassificationPredictor(
                spec.paths.get("model"),
                spec.labels or ["positive", "negative", "neutral"],
            )
        if spec.family == "nlp_summarization":
            return ExtractiveSummarizerPredictor(
                spec.paths.get("model"),
                spec.labels or ["summary"],
            )
        if spec.family == "nlp_qa":
            return KeywordQAPredictor(
                spec.paths.get("model"),
                spec.labels or ["answer"],
            )
        if spec.family in {"nlp_keras_cnn", "nlp_keras_lstm", "nlp_keras_bilstm"}:
            return KerasTextClassificationPredictor(
                spec.paths["model"],
                spec.labels or ["positive", "negative", "neutral"],
            )
        if spec.family == "nlp_keras_seq2seq":
            return KerasSeq2SeqSummarizerPredictor(
                spec.paths["model"],
                spec.labels or ["summary"],
            )
        if spec.family == "hf_bert_text_classification":
            return HuggingFaceTextClassificationPredictor(
                spec.paths["model"],
                spec.labels or ["positive", "negative", "neutral"],
            )
        if spec.family == "hf_bart_summarization":
            return HuggingFaceSummarizerPredictor(
                spec.paths["model"],
                spec.labels or ["summary"],
            )
        if spec.family == "hf_bert_question_answering":
            return HuggingFaceQAPredictor(
                spec.paths["model"],
                spec.labels or ["answer"],
            )
        if spec.family == "sklearn_pipeline":
            # Lazy import: keeps the sklearn/joblib dependency off the module
            # import path, consistent with the other predictor families.
            from app.ml.sklearn.predictor import SklearnPipelinePredictor

            return SklearnPipelinePredictor(
                spec.paths["model"],
                spec.labels or DEFAULT_LABELS,
                task_type=spec.task_type,
            )
        if spec.family == "llm_gguf":
            raise ValueError(
                "GGUF models run through the managed llama.cpp server, not the form-based "
                "predictor path. Start serving on the model detail page and use the chat surface."
            )
        if spec.family in {"llm_hf", "llm_adapter"}:
            raise ValueError(
                "This LLM artifact is not directly servable. Export it to GGUF from the "
                "model detail page, then serve the registered GGUF model."
            )
        if spec.family == "architecture_lm":
            raise ValueError(
                "From-scratch language models trained in the architecture studio are not "
                "served for inference yet. Their sample generations are on the training "
                "run's detail page."
            )
        raise ValueError(f"Unsupported model family: {spec.family}")

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
        base_model_id: str | None = None,
        format: str | None = None,
    ) -> ModelInfo:
        with self._lock:
            registry = self._read_registry()
            registry[model_id] = {
                "id": model_id,
                "name": name,
                "family": family,
                "description": description,
                "paths": {key: self._registry_path(path) for key, path in paths.items()},
                "promoted": True,
                "project_id": project_id,
                "task_type": task_type,
                "labels": labels,
                "source": source,
                "training_job_id": training_job_id,
                "base_model_id": base_model_id,
                "format": format,
                "created_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "metrics": metrics or {},
                "artifacts": artifacts or {},
            }
            self.storage.registry_file.parent.mkdir(parents=True, exist_ok=True)
            self._write_registry(registry)
            self._predictors.pop(model_id, None)
            return self.get_spec(model_id).to_info()

    def update_model(self, model_id: str, *, name: str) -> ModelInfo:
        with self._lock:
            registry = self._read_registry()
            if model_id not in registry:
                raise KeyError(model_id)
            clean_name = name.strip()
            if not clean_name:
                raise ValueError("Model name is required")
            item = dict(registry[model_id])
            item["name"] = clean_name
            # Uploaded model dirs are keyed by id (storage/uploaded_models/<id>),
            # not by slug-name, so a rename never moves files — only trained
            # model folders get the name-based relocation.
            if item.get("source") != "uploaded":
                item = self._move_owned_model_dir(item)
            registry[model_id] = item
            self._write_registry(registry)
            return self.get_spec(model_id).to_info()

    def delete_model(self, model_id: str) -> bool:
        with self._lock:
            registry = self._read_registry()
            item = registry.pop(model_id, None)
            if item is None:
                raise KeyError(model_id)
            self._write_registry(registry)
            if item.get("source") == "uploaded":
                # Uploaded weights live under storage/uploaded_models/<id>/;
                # remove the whole directory the upload created.
                self.storage.delete_owned_path(self.storage.uploaded_models / model_id)
                self._predictors.pop(model_id, None)
                return True
            model_dir = self._model_dir_for_registry_item(item)
            deleted_storage = self.storage.delete_owned_path(model_dir)
            if not deleted_storage:
                for raw_path in (item.get("paths") or {}).values():
                    self.storage.delete_owned_path(raw_path)
            self._predictors.pop(model_id, None)
            return True

    def model_download(self, model_id: str) -> tuple[Path, str]:
        spec = self.get_spec(model_id)
        paths = {key: path for key, path in spec.paths.items() if path.exists()}
        if len(paths) != len(spec.paths):
            missing = [str(path) for path in spec.paths.values() if not path.exists()]
            raise FileNotFoundError(f"Model assets are missing for {model_id}: {missing}")
        if len(paths) == 1 and next(iter(paths.values())).is_file():
            key, path = next(iter(paths.items()))
            suffix = path.suffix or ".bin"
            return path, f"{slugify_model_name(spec.name)}-{model_id}-{key}{suffix}"

        self.storage.model_downloads.mkdir(parents=True, exist_ok=True)
        archive_path = self.storage.model_downloads / f"{slugify_model_name(spec.name)}-{model_id}.zip"
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            metadata = {
                "id": spec.id,
                "name": spec.name,
                "family": spec.family,
                "task_type": spec.task_type,
                "labels": spec.labels or DEFAULT_LABELS,
                "source": spec.source,
                "training_job_id": spec.training_job_id,
            }
            archive.writestr("metadata.json", json.dumps(metadata, indent=2))
            for key, path in paths.items():
                if path.is_dir():
                    for nested in sorted(path.rglob("*")):
                        if nested.is_file():
                            archive.write(nested, arcname=f"{key}/{nested.relative_to(path).as_posix()}")
                else:
                    archive.write(path, arcname=f"{key}{path.suffix or '.bin'}")
        return archive_path, archive_path.name

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
            paths = {
                key: self._resolve_registry_path(value, item=item, path_key=key)
                for key, value in item.get("paths", {}).items()
            }
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
                    # LLM artifacts are label-free; only non-LLM families fall
                    # back to the medical DEFAULT_LABELS.
                    labels=item.get("labels")
                    or ([] if item["family"] in {"llm_hf", "llm_adapter", "llm_gguf"} else DEFAULT_LABELS.copy()),
                    source=item.get("source", "promoted"),
                    training_job_id=item.get("training_job_id"),
                    base_model_id=item.get("base_model_id"),
                    format=item.get("format"),
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
        target = self.storage.registry_file
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._json_safe(registry), indent=2, allow_nan=False)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".model_registry.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(payload)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _move_owned_model_dir(self, item: dict[str, Any]) -> dict[str, Any]:
        model_id = str(item["id"])
        current_dir = self._model_dir_for_registry_item(item)
        desired_dir = self.storage.trained_models / model_storage_dir_name(
            str(item["name"]), model_id
        )
        if current_dir is None or not current_dir.exists() or current_dir.resolve() == desired_dir.resolve():
            artifacts = dict(item.get("artifacts") or {})
            artifacts["model_dir"] = self._registry_path(desired_dir)
            item["artifacts"] = artifacts
            return item
        if desired_dir.exists():
            raise ValueError(f"Model storage folder already exists: {desired_dir.name}")
        desired_dir.parent.mkdir(parents=True, exist_ok=True)
        current_dir.rename(desired_dir)
        item["paths"] = self._rewrite_model_paths(item.get("paths") or {}, current_dir, desired_dir)
        artifacts = dict(item.get("artifacts") or {})
        artifacts["model_dir"] = self._registry_path(desired_dir)
        item["artifacts"] = artifacts
        metadata_path = desired_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(artifacts)
            metadata["name"] = item["name"]
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return item

    def _rewrite_model_paths(
        self, paths: dict[str, str], current_dir: Path, desired_dir: Path
    ) -> dict[str, str]:
        rewritten = {}
        for key, value in paths.items():
            owned_path = self.storage.owned_path(value)
            if owned_path is None:
                rewritten[key] = value
                continue
            try:
                relative = owned_path.resolve().relative_to(current_dir.resolve())
            except ValueError:
                rewritten[key] = value
                continue
            rewritten[key] = self._registry_path(desired_dir / relative)
        return rewritten

    def _model_dir_for_registry_item(self, item: dict[str, Any]) -> Path | None:
        artifacts = item.get("artifacts") or {}
        artifact_dir = artifacts.get("model_dir")
        owned_dir = self.storage.owned_path(artifact_dir)
        if owned_dir is not None and owned_dir.exists() and self._is_trained_model_dir(owned_dir):
            return owned_dir
        legacy_dir = self.storage.trained_models / str(item.get("id", ""))
        if legacy_dir.exists():
            return legacy_dir
        for raw_path in (item.get("paths") or {}).values():
            owned_path = self.storage.owned_path(raw_path)
            if owned_path is None or not owned_path.exists():
                owned_path = self._resolve_registry_path(raw_path, item=item)
                if not owned_path.exists():
                    continue
            try:
                relative = owned_path.resolve().relative_to(self.storage.trained_models.resolve())
            except ValueError:
                continue
            if relative.parts:
                return self.storage.trained_models / relative.parts[0]
        return None

    def _is_trained_model_dir(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.storage.trained_models.resolve())
        except ValueError:
            return False
        return True

    def _registry_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.storage.root.resolve()).as_posix()
        except ValueError:
            return str(resolved)

    def _resolve_registry_path(
        self,
        value: str | Path,
        *,
        item: dict[str, Any] | None = None,
        path_key: str | None = None,
    ) -> Path:
        raw_path = Path(value)
        path = raw_path if raw_path.is_absolute() else self.storage.root / raw_path
        path = path.resolve()
        if path.exists():
            return path

        moved_path = self._moved_storage_path(path)
        if moved_path is not None:
            return moved_path

        model_id = str((item or {}).get("id") or "")
        if model_id and path_key:
            suffix = path.name
            candidates = [
                self.storage.trained_models / model_id / suffix,
                self.storage.trained_models / model_id / path_key / suffix,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate.resolve()
        return path

    def _moved_storage_path(self, path: Path) -> Path | None:
        parts = path.parts
        if "storage" not in parts:
            return None
        storage_index = len(parts) - 1 - list(reversed(parts)).index("storage")
        relative_parts = parts[storage_index + 1 :]
        if not relative_parts:
            return None
        candidate = self.storage.root.joinpath(*relative_parts)
        return candidate.resolve() if candidate.exists() else None

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value


def model_storage_dir_name(name: str, model_id: str) -> str:
    slug = slugify_model_name(name)
    return f"{slug}-{model_id}"


def slugify_model_name(name: str) -> str:
    chars = []
    previous_dash = False
    for char in name.lower():
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    slug = "".join(chars).strip("-")
    return slug[:72].strip("-") or "model"
