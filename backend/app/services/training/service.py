import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Literal, TextIO, cast
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.database import SessionLocal
from app.core.defaults import DEFAULT_LABELS, DEFAULT_PROJECT_ID, DEFAULT_TASK_TYPE
from app.core.storage import Storage
from app.db.models import Architecture as ArchitectureRow
from app.db.models import TrainingJob
from app.ml.architecture.emit_keras import MODULE_FILENAME, EmitError, emit_module
from app.ml.architecture.graph import build_graph
from app.ml.architecture.lm_catalog import LM_FAMILY
from app.ml.architecture.shapes import infer_shapes
from app.ml.architecture.train_catalog import ARCHITECTURE_FAMILY, ARCHITECTURE_ID_KEY
from app.ml.llm.catalog import (
    LLM_HF_CUSTOM_ID,
    LLM_INSTALL_HINT,
    LLM_MODEL_OPTIONS,
    LLM_SFT_FAMILY,
    LOCAL_BASE_OPTION_PREFIX,
    llm_extras_available,
    llm_gated_license,
    llm_model_id,
    local_llm_base_options,
)
from app.ml.model_registry import ModelRegistry, model_storage_dir_name, slugify_model_name
from app.ml.nlp.huggingface.catalog import huggingface_model_id
from app.ml.training_catalog import training_model_options
from app.ml.vision.keras_classification.catalog import (
    KERAS_APPLICATION_OPTIONS,
    KERAS_APPLICATIONS_BY_ID,
)
from app.ml.vision.yolo.catalog import ULTRALYTICS_OPTIONS_BY_ID
from app.schemas import (
    ArchitectureGraph,
    LlmModelInfo,
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
    epoch_progress,
    find_best_model,
    iter_process_lines,
    llm_progress,
    parse_training_history,
)

ACTIVE_TRAINING_PROCESSES: dict[str, subprocess.Popen] = {}


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
        if option.family == LLM_SFT_FAMILY and not llm_extras_available():
            raise ValueError(f"LLM fine-tuning dependencies are missing. {LLM_INSTALL_HINT}")
        if option.id == LLM_HF_CUSTOM_ID and not (payload.base_model or "").strip():
            raise ValueError(
                "Enter a Hugging Face model id in the Base model field for custom fine-tuning "
                "(for example `Qwen/Qwen2.5-1.5B-Instruct`)."
            )
        if not option.runnable:
            raise ValueError(f"Training option is not runnable yet: {payload.model_option_id}")
        if payload.task_type not in option.task_types:
            supported = ", ".join(option.task_types)
            raise ValueError(
                f"Training option {payload.model_option_id} does not support {payload.task_type}. "
                f"Supported tasks: {supported}"
            )
        if option.family in {ARCHITECTURE_FAMILY, LM_FAMILY}:
            architecture_id = architecture_id_from(payload.model_dump())
            if not architecture_id:
                raise ValueError(
                    "Pick an architecture to train. Start from the studio's Train button, "
                    "or choose one in the Architecture field."
                )
            if db.get(ArchitectureRow, architecture_id) is None:
                raise ValueError(f"Architecture not found: {architecture_id}")
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
        self, db: Session, limit: int = 25, offset: int = 0, project_id: str | None = None
    ) -> list[TrainingJob]:
        return job_runner.list_jobs(db, TrainingJob, limit=limit, offset=offset, project_id=project_id)

    def model_options(self, task_type: str | None = None) -> list[TrainingModelOption]:
        options = training_model_options(task_type)
        # "Local base models": one dynamic option per registered llm_hf model
        # (phase 12 uploads and prior merges), resolved per request so a fresh
        # upload lists without a restart.
        if task_type in (None, "llm_finetune"):
            options.extend(local_llm_base_options(self.registry))
        return options

    def option_by_id(self, option_id: str) -> TrainingModelOption:
        for option in self.model_options():
            if option.id == option_id:
                return option
        raise ValueError(f"Unknown training option: {option_id}")

    def llm_model_details(self, model_ref: str) -> LlmModelInfo:
        """Accurate base-model details from the Hugging Face Hub API (phase 14).

        Resolves a catalog option id or local base id to its underlying model,
        then queries the Hub for real total size, file count, gating, and
        popularity — so the training form shows true numbers instead of the
        catalog's rough estimate. Local bases report their on-disk size.
        """
        ref = (model_ref or "").strip()
        if not ref:
            return LlmModelInfo(model_ref=ref, exists=False, error="No model id provided.")

        # Resolve a catalog/local option id to its actual model.
        if ref.startswith(LOCAL_BASE_OPTION_PREFIX):
            try:
                spec = self.registry.get_spec(ref.removeprefix(LOCAL_BASE_OPTION_PREFIX))
            except KeyError:
                return LlmModelInfo(model_ref=ref, exists=False, error="Local base model not found.")
            path = next(iter(spec.paths.values()), None)
            size = spec.size_on_disk()
            file_count = None
            if path is not None and path.is_dir():
                file_count = sum(1 for child in path.rglob("*") if child.is_file())
            return LlmModelInfo(
                model_ref=ref,
                exists=spec.available,
                size_bytes=size,
                file_count=file_count,
                library="local",
            )
        if ref in {item["id"] for item in LLM_MODEL_OPTIONS}:
            ref = llm_model_id(ref)

        try:
            from huggingface_hub import model_info  # noqa: PLC0415

            info = model_info(ref, files_metadata=True, token=self.settings.huggingface_token)
            sizes = [f.size for f in (info.siblings or []) if f.size]
            gated = bool(getattr(info, "gated", False))
            return LlmModelInfo(
                model_ref=ref,
                exists=True,
                gated=gated,
                size_bytes=sum(sizes) or None,
                file_count=len(info.siblings or []) or None,
                downloads=getattr(info, "downloads", None),
                likes=getattr(info, "likes", None),
                library=getattr(info, "library_name", None),
            )
        except Exception as exc:  # noqa: BLE001 — surfaced to the form, never a 500
            # Classify by exception type, not message text: a 404's own message
            # mentions "gated" generically, which would misreport a typo'd id.
            name = type(exc).__name__
            if name == "GatedRepoError":
                return LlmModelInfo(
                    model_ref=ref,
                    exists=True,
                    gated=True,
                    error="Access to this model is gated — accept its license on huggingface.co "
                    "with the account behind your Hugging Face token (Settings).",
                )
            if name in {"RepositoryNotFoundError", "EntryNotFoundError"}:
                return LlmModelInfo(
                    model_ref=ref,
                    exists=False,
                    error="Model id not found on the Hugging Face Hub.",
                )
            return LlmModelInfo(model_ref=ref, exists=False, error=str(exc))

    def prepare_model_asset(self, payload: ModelAssetPrepareRequest) -> ModelAssetStatus:
        option = self.option_by_id(payload.option_id)
        if option.family == LLM_SFT_FAMILY:
            return self._prepare_llm_base(payload, option)
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

    def _prepare_llm_base(self, payload: ModelAssetPrepareRequest, option) -> ModelAssetStatus:
        """Prepare an LLM base: local registry lookup or hub snapshot download.

        Downloads run 2–8 GB each, so a disk-space preflight fails before the
        download instead of at 90%; a Gemma 403 without an accepted license
        returns an actionable message naming the model page.
        """
        if option.id.startswith(LOCAL_BASE_OPTION_PREFIX):
            model_id = option.id.removeprefix(LOCAL_BASE_OPTION_PREFIX)
            try:
                spec = self.registry.get_spec(model_id)
            except KeyError:
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="missing",
                    message=f"Registered base model not found: {model_id}",
                )
            path = next(iter(spec.paths.values()), None)
            return ModelAssetStatus(
                option_id=payload.option_id,
                status="ready" if spec.available else "missing",
                path=str(path) if path else None,
                message=None if spec.available else "Local base model files are missing.",
            )

        # The custom option's hub id is typed by the user, so it rides the
        # request rather than resolving from a fixed catalog entry.
        if (option.defaults or {}).get("custom_hf"):
            model_id = (payload.model_ref or "").strip()
            if not model_id:
                return ModelAssetStatus(
                    option_id=payload.option_id,
                    status="missing",
                    message="Enter a Hugging Face model id to prepare it.",
                )
            asset_dir = self.storage.model_assets / "huggingface" / "custom" / slugify_model_name(model_id)
        else:
            model_id = llm_model_id(payload.option_id)
            asset_dir = self.storage.model_assets / "huggingface" / payload.option_id
        cache_dir = self.storage.model_assets / "huggingface_cache"
        asset_dir.mkdir(parents=True, exist_ok=True)
        if not payload.download:
            marker = asset_dir / "prepared.json"
            return ModelAssetStatus(
                option_id=payload.option_id,
                status="ready" if marker.exists() else "missing",
                path=str(asset_dir),
                message=None if marker.exists() else "Download was not requested.",
            )

        token = self.settings.huggingface_token
        approx_gb = float((option.defaults or {}).get("approx_download_gb") or 8)

        required_bytes = int(approx_gb * 1024**3)
        try:
            from huggingface_hub import model_info  # noqa: PLC0415

            info = model_info(model_id, files_metadata=True, token=token)
            sizes = [file.size for file in (info.siblings or []) if file.size]
            if sizes:
                required_bytes = sum(sizes)
        except Exception:  # noqa: BLE001 — metadata is a refinement; the catalog estimate stands
            pass
        free_bytes = shutil.disk_usage(self.storage.model_assets).free
        margin = 1024**3
        if free_bytes < required_bytes + margin:
            required_gb = required_bytes / 1024**3
            free_gb = free_bytes / 1024**3
            return ModelAssetStatus(
                option_id=payload.option_id,
                status="failed",
                path=str(asset_dir),
                message=(
                    f"Not enough disk space for {model_id}: needs ~{required_gb:.1f} GB "
                    f"(plus 1 GB margin), only {free_gb:.1f} GB free under storage/."
                ),
            )

        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)}
            if token:
                kwargs["token"] = token
            snapshot_download(model_id, **kwargs)
        except Exception as exc:  # noqa: BLE001 — surfaced to the user as asset status
            message = str(exc)
            status: Literal["ready", "missing", "gated", "failed"] = "failed"
            if "403" in message or "gated" in message.lower() or "restricted" in message.lower():
                status = "gated"
                message = (
                    f"Access to {model_id} is gated. Accept the license on "
                    f"https://huggingface.co/{model_id} with the account behind your "
                    "Hugging Face token (Settings), then retry."
                )
            return ModelAssetStatus(
                option_id=payload.option_id,
                status=status,
                path=str(asset_dir),
                message=message,
            )
        marker = asset_dir / "prepared.json"
        marker.write_text(
            json.dumps(
                {
                    "option_id": payload.option_id,
                    "model_id": model_id,
                    "token_configured": bool(token),
                    "gated_license": llm_gated_license(payload.option_id),
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
            message=f"Base model snapshot is prepared for {model_id}.",
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
        command = self._command_for_job(job, run_dir, db)

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
        sample_generations = self._read_sample_generations(run_dir)
        if sample_generations:
            artifacts["sample_generations"] = sample_generations
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

    def _read_sample_generations(self, run_dir: Path) -> list:
        """Runner-written qualitative outputs (`sample_generations.json`, phase 14)."""
        path = run_dir / "sample_generations.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

    def promote(self, db: Session, job_id: str) -> str:
        job = db.get(TrainingJob, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.model_family not in {
            "yolo",
            "keras_classification",
            "llm_sft",
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

    def _llm_model_ref(self, option_id: str, base_model: str | None = None) -> str:
        """Resolve the base a run trains against.

        Local options resolve to the registered model's filesystem path; the
        custom option uses the hub id the user typed (carried on ``base_model``);
        every other option resolves from its fixed catalog entry.
        """
        if option_id.startswith(LOCAL_BASE_OPTION_PREFIX):
            spec = self.registry.get_spec(option_id.removeprefix(LOCAL_BASE_OPTION_PREFIX))
            path = next(iter(spec.paths.values()), None)
            if path is None:
                raise ValueError(f"Registered base model has no files: {option_id}")
            return str(path)
        if option_id == LLM_HF_CUSTOM_ID:
            ref = (base_model or "").strip()
            if not ref:
                raise ValueError(
                    "Custom Hugging Face fine-tuning requires a model id in the Base model field."
                )
            return ref
        return llm_model_id(option_id)

    def _command_for_job(
        self, job: TrainingJob, run_dir: Path, db: Session | None = None
    ) -> list[str]:
        """Build the subprocess command for a job.

        `db` is optional because only the architecture family needs to read a
        row to build its command; every other family builds purely from the
        job's own parameters, and requiring a session would couple them to the
        database for no reason.
        """

        params = job.parameters
        if job.model_family == LLM_SFT_FAMILY:
            dataset_id = params.get("dataset_id")
            if not dataset_id or self.dataset_service is None:
                raise ValueError("LLM fine-tuning requires an llm_finetune project dataset")
            dataset_root = self.dataset_service.prepared_training_root(
                dataset_id, run_dir / "prepared_dataset"
            )
            option_id = str(params.get("model_option_id") or "")
            return [
                sys.executable,
                "-m",
                "app.training.runners.llm_sft",
                "--run-dir",
                str(run_dir),
                "--dataset-root",
                str(dataset_root),
                "--model-ref",
                self._llm_model_ref(option_id, params.get("base_model")),
                "--model-option-id",
                option_id,
                "--epochs",
                str(params.get("epochs", 3)),
                "--learning-rate",
                str(params.get("learning_rate", 0.0002)),
                "--batch-size",
                str(params.get("batch_size", 2)),
                "--hf-cache-dir",
                str(self.storage.model_assets / "huggingface_cache"),
                "--advanced",
                json.dumps(params.get("hyperparameters") or {}),
            ]
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
                "--advanced",
                json.dumps(hyperparameters),
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
                "--advanced",
                json.dumps(params.get("hyperparameters") or {}),
            ]
            if params.get("device"):
                command.extend(["--device", str(params["device"])])
            return command
        if job.model_family in {ARCHITECTURE_FAMILY, LM_FAMILY}:
            if db is None:
                raise ValueError("Training a visual architecture needs a database session")
            return self._architecture_command(job, run_dir, db)
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
                "--advanced",
                json.dumps(params.get("hyperparameters") or {}),
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

    def _architecture_command(
        self, job: TrainingJob, run_dir: Path, db: Session
    ) -> list[str]:
        """Emit the graph's Keras module into the run dir, then train it.

        The module is written per run rather than referenced, so a job is
        pinned to the graph as it stood when the run started — editing the
        architecture mid-run cannot change what is training.
        """

        params = job.parameters or {}
        architecture_id = architecture_id_from(params)
        row = db.get(ArchitectureRow, architecture_id) if architecture_id else None
        if row is None:
            raise ValueError(
                "This run has no architecture attached. Open the studio and start training from there."
            )

        hyperparameters = params.get("hyperparameters") or {}
        is_language_model = job.model_family == LM_FAMILY

        dataset_id = params.get("dataset_id", "reference_yolo")
        dataset_root = self.settings.datasets_path / "dental dataset_yolov11_format"
        labels = list(DEFAULT_LABELS)
        if self.dataset_service is not None:
            dataset_root = self.dataset_service.dataset_root(dataset_id)
            dataset = self.dataset_service.summary(dataset_id)
            labels = list(dataset.labels or labels)
            preprocess = (dataset.metadata or {}).get("preprocess") or {}
            if isinstance(preprocess, dict) and preprocess.get("enabled"):
                dataset_root = self.dataset_service.prepared_training_root(
                    dataset_id, run_dir / "prepared_dataset"
                )

        # The head's width comes from the vocabulary for a language model and
        # from the label set for a classifier. The runner passes the real value
        # to `build_model`; this only sizes the emitted default and lets shape
        # inference resolve before the subprocess starts.
        width = (
            int(hyperparameters.get("vocab_size", 2000))
            if is_language_model
            else max(len(labels), 2)
        )
        module_path = run_dir / MODULE_FILENAME
        module_path.write_text(self._render_architecture(row, width), encoding="utf-8")

        runner = (
            "app.training.runners.architecture_lm_train"
            if is_language_model
            else "app.training.runners.architecture_train"
        )
        return [
            sys.executable,
            "-m",
            runner,
            "--run-dir",
            str(run_dir),
            "--dataset-root",
            str(dataset_root),
            "--model-file",
            str(module_path),
            "--epochs",
            str(params["epochs"]),
            "--batch-size",
            str(params["batch_size"]),
            "--optimizer",
            str(params.get("optimizer", "adamw" if is_language_model else "adam")),
            "--learning-rate",
            str(params.get("learning_rate", 0.001)),
            "--advanced",
            json.dumps(
                {
                    key: value
                    for key, value in hyperparameters.items()
                    if key != ARCHITECTURE_ID_KEY
                }
            ),
        ]

    def _render_architecture(self, row: ArchitectureRow, num_classes: int) -> str:
        graph = ArchitectureGraph.model_validate(row.graph or {})
        resolved = build_graph(graph)
        shapes, _params, shape_issues = infer_shapes(resolved, num_classes)
        blocking = [
            issue.message
            for issue in (*resolved.issues, *shape_issues)
            if issue.severity == "error"
        ]
        if blocking:
            raise ValueError(
                "This architecture still has errors: " + "; ".join(blocking)
            )
        try:
            return emit_module(
                resolved,
                shapes,
                architecture_name=row.name,
                architecture_id=row.id,
                version=row.version,
                default_num_classes=num_classes,
            )
        except EmitError as error:
            raise ValueError(str(error)) from error

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
            # Split on \r as well as \n so Hugging Face download progress bars
            # (which redraw in place with \r) stream to the live log instead of
            # appearing only once the multi-GB download finishes.
            for line in iter_process_lines(cast(TextIO, process.stdout)):
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
        # Popen's bufsize only line-buffers *our* reading of the pipe; the child
        # still block-buffers its own stdout into 8KB chunks when it is not a
        # tty, so per-epoch prints arrived in one burst at exit and the live log
        # appeared to jump from "loading checkpoint" straight to "completed".
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("HF_HOME", str(self.storage.model_assets / "huggingface_home"))
        env.setdefault("TRANSFORMERS_CACHE", str(self.storage.model_assets / "huggingface_cache"))
        # Disable the Xet downloader so base-model downloads stream to a growing
        # `.incomplete` blob under the model cache dir, which the runner's
        # progress poller measures. Xet writes to a separate cache and only
        # lands the blob at the end, which froze the reported percentage while
        # the network kept downloading. Must be set in the child env because
        # huggingface_hub reads this flag at import time.
        env["HF_HUB_DISABLE_XET"] = "1"
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
        is_llm = job.model_family == LLM_SFT_FAMILY
        if is_llm:
            # LLM runs are step-denominated: the runner writes step/max_steps
            # per logging step, which map onto processed/total here.
            processed = int(metrics.get("step", 0)) if metrics else 0
            total = int(metrics.get("max_steps", 0)) or None if metrics else None
        else:
            raw_epoch = int(metrics.get("epoch", 0)) if metrics else 0
            processed = max(raw_epoch, len(history))
            total = epochs or None

        # One 0–100 bar across the whole run (download → load → train → save),
        # so the reported percent tracks real progress instead of racing to 99 %
        # on the download log count while the download is still at 43 %.
        if is_llm:
            percent, phase_label = llm_progress(logs, metrics)
        else:
            percent, phase_label = epoch_progress(processed, total)
        # Never regress mid-run: a truncated log window or a dip in the reported
        # download percentage must not walk the bar backwards.
        previous = ((job.artifacts or {}).get("progress") or {}).get("percent")
        if isinstance(previous, (int, float)) and job.status == "running":
            percent = max(percent, float(previous))

        # Prefer a readable step/loss summary once training is producing metrics;
        # otherwise name the current phase (data prep, download, load, saving).
        if is_llm and metrics and processed and phase_label is None:
            parts = [f"Training — step {processed}" + (f"/{total}" if total else "")]
            if "loss" in metrics:
                parts.append(f"loss {float(metrics['loss']):.3f}")
            if "val_loss" in metrics:
                parts.append(f"val loss {float(metrics['val_loss']):.3f}")
            step = " · ".join(parts)
        elif not is_llm and metrics and processed and phase_label is None:
            step = f"Epoch {processed}" + (f"/{total}" if total else "")
        else:
            step = phase_label or (logs[-1] if logs else "Training running")
        artifacts = dict(job.artifacts or {})
        artifacts["logs"] = logs[-100:]
        if history:
            artifacts["history"] = history
        if metrics:
            artifacts["metrics"] = metrics
        artifacts["progress"] = make_progress(
            percent=percent,
            processed=processed,
            total=total,
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

        base_model_id: str | None = None
        model_format: str | None = None
        if job.model_family == "yolo":
            weights_dir = model_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            stable_model = weights_dir / "best.pt"
            shutil.copy2(best_model, stable_model)
            paths = {"weights": stable_model}
            family = "yolo"
        elif job.model_family == "llm_sft":
            # A LoRA/QLoRA/continued-pretrain run produces a small adapter dir
            # (registered llm_adapter, exported via a merge in phase 15); a full
            # fine-tune produces a standalone HF model (registered llm_hf, served
            # directly). The runner names the output dir accordingly.
            is_full = best_model.name == "model"
            stable_model = model_dir / best_model.name
            if stable_model.exists():
                shutil.rmtree(stable_model)
            shutil.copytree(best_model, stable_model)
            for sidecar in ["metrics.json", "sample_generations.json"]:
                source_sidecar = run_dir / sidecar
                if source_sidecar.exists():
                    shutil.copy2(source_sidecar, model_dir / sidecar)
            paths = {"model": stable_model}
            family = "llm_hf" if is_full else "llm_adapter"
            model_format = "safetensors"
            option_id = str(params.get("model_option_id") or "")
            base_model_id = (
                option_id.removeprefix(LOCAL_BASE_OPTION_PREFIX)
                if option_id.startswith(LOCAL_BASE_OPTION_PREFIX)
                else option_id
            )
            labels = []
        elif job.model_family == LM_FAMILY:
            stable_model = model_dir / "best_model.keras"
            shutil.copy2(best_model, stable_model)
            # The vocabulary is not recoverable from the weights, so the
            # tokenizer travels with them or the model is unusable.
            for sidecar in [
                "tokenizer.json",
                MODULE_FILENAME,
                "metrics.json",
                "sample_generations.json",
            ]:
                source_sidecar = run_dir / sidecar
                if source_sidecar.exists():
                    shutil.copy2(source_sidecar, model_dir / sidecar)
            paths = {"model": stable_model, "tokenizer": model_dir / "tokenizer.json"}
            family = LM_FAMILY
        elif job.model_family in {"keras_classification", ARCHITECTURE_FAMILY}:
            stable_model = model_dir / "best_model.keras"
            shutil.copy2(best_model, stable_model)
            # A graph-built model registers as a plain Keras classifier: the
            # artifact is an ordinary `.keras` file, so testing, inference, and
            # export need no knowledge that a canvas produced it. The source
            # graph is recorded in metadata for provenance.
            for sidecar in [MODULE_FILENAME, "metrics.json", "validation_predictions.json"]:
                source_sidecar = run_dir / sidecar
                if source_sidecar.exists():
                    shutil.copy2(source_sidecar, model_dir / sidecar)
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
            base_model_id=base_model_id,
            format=model_format,
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


def architecture_id_from(params: dict[str, Any]) -> str:
    """The architecture a graph job trains, from the open-ended hyperparameters.

    Phase 13 left `hyperparameters` open exactly so routing values like this
    need no `TrainingJobCreate` field. A top-level key is also accepted so a
    hand-built payload behaves the way people expect.
    """

    hyperparameters = params.get("hyperparameters") or {}
    return str(hyperparameters.get(ARCHITECTURE_ID_KEY) or params.get(ARCHITECTURE_ID_KEY) or "")


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
