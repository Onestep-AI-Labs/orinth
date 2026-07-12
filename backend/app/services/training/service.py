import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.database import SessionLocal
from app.core.defaults import DEFAULT_LABELS, DEFAULT_PROJECT_ID, DEFAULT_TASK_TYPE
from app.core.storage import Storage
from app.db.models import TrainingJob
from app.ml.model_registry import ModelRegistry, model_storage_dir_name
from app.ml.nlp.huggingface.catalog import huggingface_model_id
from app.ml.training_catalog import training_model_options
from app.schemas import (
    ModelAssetPrepareRequest,
    ModelAssetStatus,
    TrainingJobCreate,
    TrainingModelOption,
)
from app.services import job_runner
from app.services.datasets import DatasetService
from app.services.job_progress import append_log, make_progress
from app.services.training.artifacts import (
    clean_log_line,
    collect_training_curves,
    collect_training_metrics,
    find_best_model,
    parse_training_history,
)

ACTIVE_TRAINING_PROCESSES: dict[str, subprocess.Popen] = {}

KERAS_APPLICATION_OPTIONS: list[dict[str, Any]] = [
    {
        "id": "keras_mobilenet_v2",
        "name": "Keras MobileNetV2",
        "app_name": "MobileNetV2",
        "image_size": 224,
        "kwargs": {"alpha": 1.0},
        "description": "Lightweight ImageNet CNN suitable for fast transfer learning.",
    },
    {
        "id": "keras_efficientnet_b0",
        "name": "Keras EfficientNetB0",
        "app_name": "EfficientNetB0",
        "image_size": 224,
        "kwargs": {},
        "description": "Balanced EfficientNet baseline for image classification transfer learning.",
    },
    {
        "id": "keras_efficientnet_b1",
        "name": "Keras EfficientNetB1",
        "app_name": "EfficientNetB1",
        "image_size": 240,
        "kwargs": {},
        "description": "EfficientNet B1 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b2",
        "name": "Keras EfficientNetB2",
        "app_name": "EfficientNetB2",
        "image_size": 260,
        "kwargs": {},
        "description": "EfficientNet B2 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b3",
        "name": "Keras EfficientNetB3",
        "app_name": "EfficientNetB3",
        "image_size": 300,
        "kwargs": {},
        "description": "EfficientNet B3 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b4",
        "name": "Keras EfficientNetB4",
        "app_name": "EfficientNetB4",
        "image_size": 380,
        "kwargs": {},
        "description": "EfficientNet B4 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b5",
        "name": "Keras EfficientNetB5",
        "app_name": "EfficientNetB5",
        "image_size": 456,
        "kwargs": {},
        "description": "EfficientNet B5 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b6",
        "name": "Keras EfficientNetB6",
        "app_name": "EfficientNetB6",
        "image_size": 528,
        "kwargs": {},
        "description": "EfficientNet B6 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_b7",
        "name": "Keras EfficientNetB7",
        "app_name": "EfficientNetB7",
        "image_size": 600,
        "kwargs": {"name": "efficientnetb7"},
        "description": "Large EfficientNet B7 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_v2_b0",
        "name": "Keras EfficientNetV2B0",
        "app_name": "EfficientNetV2B0",
        "image_size": 224,
        "kwargs": {},
        "description": "EfficientNetV2 B0 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_v2_b3",
        "name": "Keras EfficientNetV2B3",
        "app_name": "EfficientNetV2B3",
        "image_size": 300,
        "kwargs": {},
        "description": "EfficientNetV2 B3 ImageNet backbone.",
    },
    {
        "id": "keras_efficientnet_v2_s",
        "name": "Keras EfficientNetV2S",
        "app_name": "EfficientNetV2S",
        "image_size": 384,
        "kwargs": {},
        "description": "EfficientNetV2 S ImageNet backbone.",
    },
    {
        "id": "keras_resnet50",
        "name": "Keras ResNet50",
        "app_name": "ResNet50",
        "image_size": 224,
        "kwargs": {},
        "description": "Classic ResNet50 ImageNet backbone.",
    },
    {
        "id": "keras_xception",
        "name": "Keras Xception",
        "app_name": "Xception",
        "image_size": 299,
        "kwargs": {},
        "description": "Xception ImageNet backbone.",
    },
    {
        "id": "keras_inception_v3",
        "name": "Keras InceptionV3",
        "app_name": "InceptionV3",
        "image_size": 299,
        "kwargs": {},
        "description": "InceptionV3 ImageNet backbone.",
    },
    {
        "id": "keras_densenet121",
        "name": "Keras DenseNet121",
        "app_name": "DenseNet121",
        "image_size": 224,
        "kwargs": {},
        "description": "DenseNet121 ImageNet backbone.",
    },
    {
        "id": "keras_convnext_tiny",
        "name": "Keras ConvNeXtTiny",
        "app_name": "ConvNeXtTiny",
        "image_size": 224,
        "kwargs": {},
        "description": "ConvNeXt Tiny ImageNet backbone.",
    },
]

KERAS_APPLICATIONS_BY_ID = {item["id"]: item for item in KERAS_APPLICATION_OPTIONS}

ULTRALYTICS_MODEL_OPTIONS: list[dict[str, Any]] = [
    {
        "id": "ultralytics_yolo11_detect",
        "name": "Ultralytics YOLO11 Detection",
        "family": "yolo",
        "task_types": ["object_detection"],
        "weights": "yolo11n.pt",
        "runnable": True,
        "needs_download": True,
        "description": "YOLO11 nano detection fine-tuning through the Ultralytics YOLO runner.",
    },
    {
        "id": "ultralytics_yolo11_segment",
        "name": "Ultralytics YOLO11 Segmentation",
        "family": "yolo",
        "task_types": ["segmentation"],
        "weights": "yolo11n-seg.pt",
        "runnable": True,
        "needs_download": True,
        "description": "YOLO11 nano instance segmentation fine-tuning through the Ultralytics YOLO runner.",
    },
    {
        "id": "ultralytics_yolo26_detect",
        "name": "Ultralytics YOLO26 Detection",
        "family": "yolo",
        "task_types": ["object_detection"],
        "weights": "yolo26n.pt",
        "runnable": True,
        "needs_download": True,
        "description": "YOLO26 nano detection fine-tuning through the Ultralytics YOLO runner.",
    },
    {
        "id": "ultralytics_yolo26_segment",
        "name": "Ultralytics YOLO26 Segmentation",
        "family": "yolo",
        "task_types": ["segmentation"],
        "weights": "yolo26n-seg.pt",
        "runnable": True,
        "needs_download": True,
        "description": "YOLO26 nano instance segmentation fine-tuning through the Ultralytics YOLO runner.",
    },
    {
        "id": "ultralytics_sam3",
        "name": "Ultralytics SAM3",
        "family": "sam",
        "task_types": ["segmentation"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "Promptable segmentation catalog option. Training is gated until the dataset workflow is validated.",
    },
    {
        "id": "ultralytics_mobilesam",
        "name": "Ultralytics MobileSAM",
        "family": "sam",
        "task_types": ["segmentation"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "Mobile promptable segmentation catalog option, gated pending validation.",
    },
    {
        "id": "ultralytics_fastsam",
        "name": "Ultralytics FastSAM",
        "family": "sam",
        "task_types": ["segmentation"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "Fast promptable segmentation catalog option, gated pending validation.",
    },
    {
        "id": "ultralytics_yolo_nas",
        "name": "Ultralytics YOLO-NAS",
        "family": "yolo_nas",
        "task_types": ["object_detection"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "YOLO-NAS detection catalog option, gated until runner support is validated.",
    },
    {
        "id": "ultralytics_rt_detr",
        "name": "Ultralytics RT-DETR",
        "family": "rt_detr",
        "task_types": ["object_detection"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "RT-DETR detection catalog option, gated until runner support is validated.",
    },
    {
        "id": "ultralytics_yolo_world",
        "name": "Ultralytics YOLO-World",
        "family": "yolo_world",
        "task_types": ["object_detection"],
        "weights": None,
        "runnable": False,
        "needs_download": True,
        "description": "Open-vocabulary detection catalog option, gated pending validation.",
    },
]

ULTRALYTICS_OPTIONS_BY_ID = {item["id"]: item for item in ULTRALYTICS_MODEL_OPTIONS}


class TrainingService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        registry: ModelRegistry,
        dataset_service: DatasetService | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.registry = registry
        self.dataset_service = dataset_service

    def create_job(self, db: Session, payload: TrainingJobCreate) -> TrainingJob:
        option = self.option_by_id(payload.model_option_id)
        if not option.runnable:
            raise ValueError(f"Training option is not runnable yet: {payload.model_option_id}")
        if payload.task_type not in option.task_types:
            supported = ", ".join(option.task_types)
            raise ValueError(
                f"Training option {payload.model_option_id} does not support {payload.task_type}. "
                f"Supported tasks: {supported}"
            )
        parameters = payload.model_dump()
        parameters["model_family"] = option.family
        job = TrainingJob(
            id=uuid4().hex,
            project_id=payload.project_id,
            model_family=option.family,
            status="queued",
            parameters=parameters,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    def list_jobs(
        self, db: Session, limit: int = 25, project_id: str | None = None
    ) -> list[TrainingJob]:
        return job_runner.list_jobs(db, TrainingJob, limit=limit, project_id=project_id)

    def model_options(self, task_type: str | None = None) -> list[TrainingModelOption]:
        return training_model_options(task_type)

    def option_by_id(self, option_id: str) -> TrainingModelOption:
        for option in self.model_options():
            if option.id == option_id:
                return option
        raise ValueError(f"Unknown training option: {option_id}")

    def prepare_model_asset(self, payload: ModelAssetPrepareRequest) -> ModelAssetStatus:
        option = self.option_by_id(payload.option_id)
        if option.source == "huggingface":
            if not option.runnable:
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="gated",
                    message="This Hugging Face model family is listed but not runnable until validated.",
                )
            asset_dir = self.storage.model_assets / "huggingface" / payload.option_id
            cache_dir = self.storage.model_assets / "huggingface_cache"
            asset_dir.mkdir(parents=True, exist_ok=True)
            if not payload.download:
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="missing",
                    path=str(asset_dir),
                    message="Download was not requested.",
                )
            try:
                from transformers import (  # noqa: PLC0415
                    AutoModel,
                    AutoModelForSeq2SeqLM,
                    AutoTokenizer,
                )

                model_id = huggingface_model_id(payload.option_id)
                token = self.settings.huggingface_token
                kwargs = {"cache_dir": str(cache_dir)}
                if token:
                    kwargs["token"] = token
                AutoTokenizer.from_pretrained(model_id, **kwargs)
                if option.family == "hf_bart_summarization":
                    AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)
                else:
                    AutoModel.from_pretrained(model_id, **kwargs)
                marker = asset_dir / "prepared.json"
                marker.write_text(
                    json.dumps(
                        {
                            "option_id": payload.option_id,
                            "model_id": model_id,
                            "token_configured": bool(token),
                            "prepared_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="ready",
                    path=str(asset_dir),
                    message=f"Hugging Face assets are prepared for {model_id}.",
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the user as asset status
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="failed",
                    path=str(asset_dir),
                    message=str(exc),
                )
        if option.id == "yolo_local":
            path = self.settings.models_path / "yolo_11_best" / "weights" / "best.pt"
            return ModelAssetStatus(
                option_id=payload.option_id,
                status="ready" if path.exists() else "missing",
                path=str(path),
                message=None if path.exists() else "Local YOLO weights are missing.",
            )
        if option.source == "local":
            return ModelAssetStatus(
                option_id=payload.option_id,
                status="ready",
                message="This offline baseline does not require downloaded assets.",
            )
        if option.source == "ultralytics":
            if not option.runnable:
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="gated",
                    message="This Ultralytics model family is listed but not runnable until validated.",
                )
            asset_dir = self.storage.model_assets / payload.option_id
            asset_dir.mkdir(parents=True, exist_ok=True)
            if not payload.download:
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="missing",
                    path=str(asset_dir),
                    message="Download was not requested.",
                )
            try:
                from ultralytics import YOLO  # noqa: PLC0415

                weights = ultralytics_initial_weights(payload.option_id, self.settings)
                YOLO(weights)
                marker = asset_dir / "prepared.json"
                marker.write_text(
                    f'{{"option_id": "{payload.option_id}", "weights": "{weights}", '
                    f'"prepared_at": "{datetime.now(UTC).replace(tzinfo=None).isoformat()}"}}\n',
                    encoding="utf-8",
                )
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="ready",
                    path=str(asset_dir),
                    message=f"Ultralytics weights are prepared: {weights}",
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the user as asset status
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="failed",
                    path=str(asset_dir),
                    message=str(exc),
                )
        if option.source == "keras_applications":
            asset_dir = self.storage.model_assets / payload.option_id
            asset_dir.mkdir(parents=True, exist_ok=True)
            if not payload.download:
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="missing",
                    path=str(asset_dir),
                    message="Download was not requested.",
                )
            try:
                os.environ.setdefault("KERAS_HOME", str(self.storage.model_assets / "keras_home"))
                spec = keras_application_spec(payload.option_id)
                app_name = spec["app_name"]
                import tensorflow as tf  # noqa: PLC0415

                app = getattr(tf.keras.applications, app_name)
                app(
                    weights="imagenet",
                    include_top=False,
                    pooling="avg",
                    input_shape=(int(spec["image_size"]), int(spec["image_size"]), 3),
                    **spec.get("kwargs", {}),
                )
                marker = asset_dir / "prepared.json"
                marker.write_text(
                    f'{{"option_id": "{payload.option_id}", "prepared_at": "{datetime.now(UTC).replace(tzinfo=None).isoformat()}"}}\n',
                    encoding="utf-8",
                )
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="ready",
                    path=str(asset_dir),
                    message="Keras application weights are prepared.",
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the user as asset status
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="failed",
                    path=str(asset_dir),
                    message=str(exc),
                )
        return ModelAssetStatus(
            option_id=payload.option_id,
            status="missing",
            message="No asset preparation path is registered for this option.",
        )

    def delete_jobs(self, db: Session, ids: list[str] | None = None, project_id: str | None = None) -> dict:
        def is_deletable(job: TrainingJob) -> bool:
            return job.status in {"completed", "failed", "canceled"}

        def on_delete(job: TrainingJob) -> None:
            run_dir = (job.artifacts or {}).get("run_dir")
            self.storage.delete_owned_path(run_dir or (self.storage.training_runs / job.id))

        return job_runner.delete_jobs(
            db,
            TrainingJob,
            ids=ids,
            project_id=project_id,
            is_deletable=is_deletable,
            on_delete=on_delete,
        )

    def run_job(self, job_id: str) -> None:
        job_runner.run_job_lifecycle(
            SessionLocal,
            TrainingJob,
            job_id,
            on_start=self._on_training_start,
            execute=self._execute_training,
            on_cleanup=self._cleanup_active_process,
        )

    def _cleanup_active_process(self, job_id: str) -> None:
        ACTIVE_TRAINING_PROCESSES.pop(job_id, None)

    def _on_training_start(self, db: Session, job: TrainingJob, started_at: datetime) -> None:
        artifacts = append_log(job.artifacts, "Training subprocess starting")
        artifacts["progress"] = make_progress(
            percent=1,
            current_step="Starting subprocess",
            started_at=started_at,
            logs=artifacts["logs"],
        )
        job.artifacts = artifacts

    def _execute_training(self, db: Session, job: TrainingJob, started_at: datetime) -> None:
        run_dir = self.storage.training_runs / job.id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        command = self._command_for_job(job, run_dir)

        artifacts = dict(job.artifacts or {})
        artifacts.update({"run_dir": str(run_dir), "log": str(log_path), "command": command})
        job.artifacts = artifacts
        db.commit()

        return_code = self._run_process(db, job.id, command, run_dir, log_path, started_at)
        refreshed_job = db.get(TrainingJob, job.id)
        if refreshed_job is None:
            return
        job = refreshed_job

        artifacts = dict(job.artifacts or {})
        artifacts.update({"run_dir": str(run_dir), "log": str(log_path)})
        best_model = find_best_model(run_dir)
        if best_model:
            artifacts["best_model"] = str(best_model)
        history = parse_training_history(run_dir / "results.csv")
        if history:
            artifacts["history"] = history
        metrics = collect_training_metrics(run_dir)
        if metrics:
            artifacts["metrics"] = metrics
        curves = collect_training_curves(run_dir)
        if curves:
            artifacts["curves"] = curves
        artifact_urls = self._artifact_urls(run_dir)
        if artifact_urls:
            artifacts["artifact_urls"] = artifact_urls
        finished_at = datetime.now(UTC).replace(tzinfo=None)

        if job.status == "canceled":
            artifacts = append_log(artifacts, "Training canceled")
            artifacts["progress"] = make_progress(
                percent=100,
                current_step="Canceled",
                started_at=started_at,
                finished_at=finished_at,
                logs=artifacts["logs"],
            )
            job.error = "Training canceled"
            job.artifacts = artifacts
            job.updated_at = finished_at
            db.commit()
            return

        job.status = "completed" if return_code == 0 else "failed"
        if return_code != 0:
            job.error = f"Training process exited with code {return_code}"
        elif best_model:
            try:
                promoted_model_id = self._register_training_model(job, run_dir, Path(best_model), metrics)
                job.promoted_model_id = promoted_model_id
                artifacts["promoted_model_id"] = promoted_model_id
            except Exception as exc:  # noqa: BLE001 - registration failure should be visible but not lose run
                artifacts = append_log(artifacts, f"Model registration failed: {exc}")
        artifacts = append_log(
            artifacts,
            "Training completed" if return_code == 0 else (job.error or "Training failed"),
        )
        artifacts["progress"] = make_progress(
            percent=100,
            processed=int(job.parameters.get("epochs", 0)),
            total=int(job.parameters.get("epochs", 0)),
            current_step="Completed" if return_code == 0 else "Failed",
            started_at=started_at,
            finished_at=finished_at,
            logs=artifacts["logs"],
        )
        job.artifacts = artifacts
        job.updated_at = finished_at
        db.commit()

    def promote(self, db: Session, job_id: str) -> str:
        job = db.get(TrainingJob, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.model_family not in {
            "yolo",
            "keras_classification",
            "nlp_text_classification",
            "nlp_summarization",
            "nlp_qa",
            "nlp_keras_cnn",
            "nlp_keras_lstm",
            "nlp_keras_bilstm",
            "nlp_keras_seq2seq",
            "hf_bert_text_classification",
            "hf_bert_question_answering",
            "hf_bart_summarization",
        }:
            raise ValueError("Only runnable training artifacts can be promoted")
        if job.promoted_model_id:
            return job.promoted_model_id
        best_model = job.artifacts.get("best_model") if job.artifacts else None
        if not best_model:
            raise FileNotFoundError("No best_model artifact found for this job")
        run_dir = Path((job.artifacts or {}).get("run_dir") or self.storage.training_runs / job.id)
        model_id = self._register_training_model(
            job,
            run_dir,
            Path(best_model),
            (job.artifacts or {}).get("metrics") or {},
        )
        job.promoted_model_id = model_id
        job.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        return model_id

    def cancel(self, db: Session, job_id: str) -> TrainingJob:
        job = db.get(TrainingJob, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status not in {"queued", "running"}:
            return job
        process = ACTIVE_TRAINING_PROCESSES.get(job_id)
        if process is not None and process.poll() is None:
            process.terminate()
        artifacts = append_log(job.artifacts, "Cancel requested")
        artifacts["progress"] = make_progress(
            percent=100,
            current_step="Canceled",
            finished_at=datetime.now(UTC).replace(tzinfo=None),
            logs=artifacts["logs"],
        )
        job.status = "canceled"
        job.error = "Training canceled"
        job.artifacts = artifacts
        job.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        db.refresh(job)
        return job

    def reconcile_stale_jobs(self) -> None:
        db = SessionLocal()
        try:
            jobs = db.scalars(
                select(TrainingJob).where(TrainingJob.status.in_(["queued", "running"]))
            ).all()
            for job in jobs:
                artifacts = append_log(job.artifacts, "Backend restarted before training completed")
                artifacts["progress"] = make_progress(
                    percent=100,
                    current_step="Failed",
                    finished_at=datetime.now(UTC).replace(tzinfo=None),
                    logs=artifacts["logs"],
                )
                job.status = "failed"
                job.error = "Backend restarted before training completed"
                job.artifacts = artifacts
                job.updated_at = datetime.now(UTC).replace(tzinfo=None)
            db.commit()
        finally:
            db.close()

    def _command_for_job(self, job: TrainingJob, run_dir: Path) -> list[str]:
        params = job.parameters
        if job.model_family in {
            "nlp_text_classification",
            "nlp_summarization",
            "nlp_qa",
            "nlp_keras_cnn",
            "nlp_keras_lstm",
            "nlp_keras_bilstm",
            "nlp_keras_seq2seq",
            "hf_bert_text_classification",
            "hf_bert_question_answering",
            "hf_bart_summarization",
        }:
            dataset_id = params.get("dataset_id")
            if not dataset_id or self.dataset_service is None:
                raise ValueError("NLP training requires a project text dataset")
            dataset_root = self.dataset_service.prepared_training_root(
                dataset_id, run_dir / "prepared_dataset"
            )
            option_id = str(params.get("model_option_id") or "")
            if not option_id:
                option_id = {
                    "text_classification": "nlp_tfidf_classifier",
                    "summarization": "nlp_extractive_summarizer",
                    "question_answering": "nlp_keyword_qa",
                }.get(str(params.get("task_type")), "nlp_tfidf_classifier")
            option = self.option_by_id(option_id)
            defaults = option.defaults or {}
            hyperparameters = params.get("hyperparameters") or {}
            max_length = hyperparameters.get("max_length") or defaults.get("max_length") or 160
            target_max_length = (
                hyperparameters.get("target_max_length")
                or defaults.get("target_max_length")
                or max(32, int(max_length) // 3)
            )
            vocab_size = hyperparameters.get("vocab_size") or defaults.get("vocab_size") or 12000
            batch_size = int(params.get("batch_size") or defaults.get("batch_size") or 8)
            if batch_size <= 0:
                batch_size = int(defaults.get("batch_size") or 8)
            return [
                sys.executable,
                "-m",
                "app.training.runners.nlp_train",
                "--run-dir",
                str(run_dir),
                "--dataset-root",
                str(dataset_root),
                "--task-type",
                str(params.get("task_type")),
                "--model-option-id",
                option_id,
                "--epochs",
                str(params.get("epochs", 1)),
                "--learning-rate",
                str(params.get("learning_rate", 1.0)),
                "--batch-size",
                str(batch_size),
                "--max-length",
                str(max_length),
                "--target-max-length",
                str(target_max_length),
                "--vocab-size",
                str(vocab_size),
                "--hf-model-id",
                str(defaults.get("model_id") or huggingface_model_id(option_id)),
                "--hf-cache-dir",
                str(self.storage.model_assets / "huggingface_cache"),
            ]
        if job.model_family == "yolo":
            dataset_id = params.get("dataset_id", "reference_yolo")
            if self.dataset_service is not None:
                dataset_root = self.dataset_service.prepared_training_root(
                    dataset_id, run_dir / "prepared_dataset"
                )
                labels = self.dataset_service.summary(dataset_id).labels
            else:
                dataset_root = self.settings.datasets_path / "dental dataset_yolov11_format"
                labels = DEFAULT_LABELS
            initial_weights = ultralytics_initial_weights(
                params.get("model_option_id", "yolo_local"), self.settings
            )
            command = [
                sys.executable,
                "-m",
                "app.training.runners.yolo_train",
                "--run-dir",
                str(run_dir),
                "--dataset-root",
                str(dataset_root),
                "--initial-weights",
                str(initial_weights),
                "--class-names",
                ",".join(labels),
                "--epochs",
                str(params["epochs"]),
                "--image-size",
                str(params["image_size"]),
                "--batch-size",
                str(params["batch_size"]),
                "--cache",
                str(params.get("cache", "disk")),
                "--workers",
                str(params.get("workers", 0)),
                "--patience",
                str(params.get("patience", 50)),
                "--optimizer",
                str(params.get("optimizer", "AdamW")),
                "--learning-rate",
                str(params.get("learning_rate", 0.002)),
            ]
            if params.get("device"):
                command.extend(["--device", str(params["device"])])
            return command
        if job.model_family == "keras_classification":
            dataset_id = params.get("dataset_id", "reference_yolo")
            dataset_root = self.settings.datasets_path / "dental dataset_yolov11_format"
            if self.dataset_service is not None:
                dataset_root = self.dataset_service.dataset_root(dataset_id)
                dataset = self.dataset_service.summary(dataset_id)
                preprocess = (dataset.metadata or {}).get("preprocess") or {}
                if isinstance(preprocess, dict) and preprocess.get("enabled"):
                    dataset_root = self.dataset_service.prepared_training_root(
                        dataset_id, run_dir / "prepared_dataset"
                    )
            base_model = params.get("base_model") or keras_application_name(
                params.get("model_option_id", "keras_efficientnet_b0")
            )
            app_kwargs = params.get("architecture", {}).get("application_kwargs")
            if app_kwargs is None:
                app_kwargs = keras_application_kwargs(params.get("model_option_id", "keras_efficientnet_b0"))
            return [
                sys.executable,
                "-m",
                "app.training.runners.keras_classification_train",
                "--run-dir",
                str(run_dir),
                "--dataset-root",
                str(dataset_root),
                "--base-model",
                str(base_model),
                "--application-kwargs",
                json.dumps(app_kwargs),
                "--epochs",
                str(params["epochs"]),
                "--image-size",
                str(params["image_size"]),
                "--batch-size",
                str(params["batch_size"]),
                "--optimizer",
                str(params.get("optimizer", "adam")),
                "--learning-rate",
                str(params.get("learning_rate", 0.001)),
            ]
        return [
            sys.executable,
            "-m",
            "app.training.runners.unet_inception_train",
            "--run-dir",
            str(run_dir),
            "--dataset-root",
            str(self.settings.datasets_path / "dental dataset_coco_format"),
            "--epochs",
            str(params["epochs"]),
        ]

    def _run_process(
        self,
        db: Session,
        job_id: str,
        command: list[str],
        run_dir: Path,
        log_path: Path,
        started_at: datetime,
    ) -> int:
        logs: list[str] = []
        with log_path.open("w", encoding="utf-8") as log_file:
            env = self._process_env()
            process = subprocess.Popen(
                command,
                cwd=self.settings.repo_root / "backend",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            ACTIVE_TRAINING_PROCESSES[job_id] = process
            self._set_pid(db, job_id, process.pid)
            last_update = 0.0
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                clean = clean_log_line(line)
                if clean:
                    logs.append(clean)
                now = monotonic()
                if now - last_update > 1.5:
                    last_update = now
                    self._update_training_progress(db, job_id, run_dir, logs, started_at)
            return process.wait()

    def _process_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("HF_HOME", str(self.storage.model_assets / "huggingface_home"))
        env.setdefault("TRANSFORMERS_CACHE", str(self.storage.model_assets / "huggingface_cache"))
        token = self.settings.huggingface_token
        if token:
            env["HUGGINGFACE_HUB_TOKEN"] = token
            env["HF_TOKEN"] = token
        return env

    def _set_pid(self, db: Session, job_id: str, pid: int) -> None:
        job = db.get(TrainingJob, job_id)
        if job is None:
            return
        artifacts = dict(job.artifacts or {})
        artifacts["pid"] = pid
        job.artifacts = artifacts
        job.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()

    def _update_training_progress(
        self,
        db: Session,
        job_id: str,
        run_dir: Path,
        logs: list[str],
        started_at: datetime,
    ) -> None:
        job = db.get(TrainingJob, job_id)
        if job is None:
            return
        epochs = int(job.parameters.get("epochs", 0))
        history = parse_training_history(run_dir / "results.csv")
        metrics = history[-1] if history else {}
        raw_epoch = int(metrics.get("epoch", 0)) if metrics else 0
        epoch = max(raw_epoch, len(history))
        step = logs[-1] if logs else "Training running"
        artifacts = dict(job.artifacts or {})
        artifacts["logs"] = logs[-100:]
        if history:
            artifacts["history"] = history
        if metrics:
            artifacts["metrics"] = metrics
        artifacts["progress"] = make_progress(
            percent=(epoch / epochs * 100) if epochs and epoch else min(99, 5 + len(logs)),
            processed=epoch,
            total=epochs or None,
            current_step=step[:180],
            started_at=started_at,
            logs=artifacts["logs"],
        )
        job.artifacts = artifacts
        job.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()

    def _register_training_model(
        self,
        job: TrainingJob,
        run_dir: Path,
        best_model: Path,
        metrics: dict,
    ) -> str:
        params = job.parameters or {}
        model_id = f"trained_{job.model_family}_{job.id[:8]}"
        display_name = (params.get("model_name") or "").strip()
        if not display_name:
            display_name = f"Trained {job.model_family.replace('_', ' ')} {job.id[:8]}"
        model_dir = self.storage.trained_models / model_storage_dir_name(display_name, model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        labels = DEFAULT_LABELS.copy()
        dataset_id = params.get("dataset_id")
        if self.dataset_service is not None and dataset_id:
            labels = self.dataset_service.summary(dataset_id).labels

        if job.model_family == "yolo":
            weights_dir = model_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            stable_model = weights_dir / "best.pt"
            shutil.copy2(best_model, stable_model)
            paths = {"weights": stable_model}
            family = "yolo"
        elif job.model_family == "keras_classification":
            stable_model = model_dir / "best_model.keras"
            shutil.copy2(best_model, stable_model)
            paths = {"model": stable_model}
            family = "keras_classification"
        elif job.model_family in {
            "nlp_text_classification",
            "nlp_summarization",
            "nlp_qa",
            "nlp_keras_cnn",
            "nlp_keras_lstm",
            "nlp_keras_bilstm",
            "nlp_keras_seq2seq",
            "hf_bert_text_classification",
            "hf_bert_question_answering",
            "hf_bart_summarization",
        }:
            stable_model = model_dir / best_model.name
            if best_model.is_dir():
                if stable_model.exists():
                    shutil.rmtree(stable_model)
                shutil.copytree(best_model, stable_model)
            else:
                shutil.copy2(best_model, stable_model)
                for sidecar in [
                    "last_model.keras",
                    "tokenizer.json",
                    "metrics.json",
                    "validation_predictions.json",
                ]:
                    source_sidecar = run_dir / sidecar
                    if source_sidecar.exists() and source_sidecar != best_model:
                        shutil.copy2(source_sidecar, model_dir / sidecar)
                runner_metadata_source = run_dir / "metadata.json"
                if runner_metadata_source.exists():
                    shutil.copy2(runner_metadata_source, model_dir / "runner_metadata.json")
            paths = {"model": stable_model}
            family = job.model_family
        else:
            raise ValueError(f"Unsupported trained model family: {job.model_family}")

        runner_metadata: dict[str, Any] = {}
        runner_metadata_path = run_dir / "metadata.json"
        if runner_metadata_path.exists():
            try:
                loaded_metadata = json.loads(runner_metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded_metadata = {}
            if isinstance(loaded_metadata, dict):
                runner_metadata = loaded_metadata

        metadata: dict[str, Any] = {
            **runner_metadata,
            "labels": labels,
            "task_type": params.get("task_type", DEFAULT_TASK_TYPE),
            "model_family": job.model_family,
            "training_job_id": job.id,
            "image_size": params.get("image_size"),
            "run_dir": str(run_dir),
        }
        metadata["name"] = display_name
        metadata["model_id"] = model_id
        metadata["model_dir"] = str(model_dir)
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        info = self.registry.register_model(
            model_id=model_id,
            name=display_name,
            family=family,
            task_type=params.get("task_type", DEFAULT_TASK_TYPE),
            paths=paths,
            labels=labels,
            project_id=job.project_id or DEFAULT_PROJECT_ID,
            description=f"Completed training artifact from job {job.id[:8]}.",
            training_job_id=job.id,
            metrics=metrics,
            artifacts=metadata,
            source="trained",
        )
        return info.id

    def _artifact_urls(self, run_dir: Path) -> dict[str, str]:
        urls = {}
        for path in sorted(run_dir.glob("*.png")):
            try:
                urls[path.stem] = self.storage.media_url(path)
            except ValueError:
                continue
        for path in sorted((run_dir / "weights").glob("*.png")):
            try:
                urls[path.stem] = self.storage.media_url(path)
            except ValueError:
                continue
        return urls


def ultralytics_initial_weights(option_id: str, settings: Settings) -> str:
    if option_id == "yolo_local":
        return str(settings.models_path / "yolo_11_best" / "weights" / "best.pt")
    option = ULTRALYTICS_OPTIONS_BY_ID.get(option_id)
    if option and option.get("weights"):
        return str(option["weights"])
    return str(settings.models_path / "yolo_11_best" / "weights" / "best.pt")


def keras_application_name(option_id: str) -> str:
    if option_id in KERAS_APPLICATIONS_BY_ID:
        return str(KERAS_APPLICATIONS_BY_ID[option_id]["app_name"])
    if option_id in {str(item["app_name"]) for item in KERAS_APPLICATION_OPTIONS}:
        return option_id
    return "EfficientNetB0"


def keras_application_kwargs(option_id: str) -> dict:
    if option_id in KERAS_APPLICATIONS_BY_ID:
        return dict(KERAS_APPLICATIONS_BY_ID[option_id].get("kwargs", {}))
    return {}


def keras_application_spec(option_id: str) -> dict:
    if option_id in KERAS_APPLICATIONS_BY_ID:
        return KERAS_APPLICATIONS_BY_ID[option_id]
    for item in KERAS_APPLICATION_OPTIONS:
        if item["app_name"] == option_id:
            return item
    return KERAS_APPLICATIONS_BY_ID["keras_efficientnet_b0"]
