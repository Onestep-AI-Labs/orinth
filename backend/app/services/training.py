import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.database import SessionLocal
from app.core.storage import Storage
from app.db.models import TrainingJob
from app.ml.model_registry import ModelRegistry
from app.schemas import TrainingJobCreate


class TrainingService:
    def __init__(self, settings: Settings, storage: Storage, registry: ModelRegistry) -> None:
        self.settings = settings
        self.storage = storage
        self.registry = registry

    def create_job(self, db: Session, payload: TrainingJobCreate) -> TrainingJob:
        job = TrainingJob(
            id=uuid4().hex,
            model_family=payload.model_family,
            status="queued",
            parameters=payload.model_dump(),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    def list_jobs(self, db: Session, limit: int = 25) -> list[TrainingJob]:
        return db.scalars(
            select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit)
        ).all()

    def run_job(self, job_id: str) -> None:
        db = SessionLocal()
        try:
            job = db.get(TrainingJob, job_id)
            if job is None:
                return
            job.status = "running"
            job.updated_at = datetime.utcnow()
            db.commit()

            run_dir = self.storage.training_runs / job.id
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"
            command = self._command_for_job(job, run_dir)

            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.run(
                    command,
                    cwd=self.settings.repo_root / "backend",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

            job = db.get(TrainingJob, job_id)
            if job is None:
                return
            artifacts = {"run_dir": str(run_dir), "log": str(log_path)}
            best_model = find_best_model(run_dir)
            if best_model:
                artifacts["best_model"] = str(best_model)
            job.artifacts = artifacts
            job.status = "completed" if process.returncode == 0 else "failed"
            if process.returncode != 0:
                job.error = f"Training process exited with code {process.returncode}"
            job.updated_at = datetime.utcnow()
            db.commit()
        except Exception as exc:  # noqa: BLE001 - background jobs must persist errors
            job = db.get(TrainingJob, job_id)
            if job is not None:
                job.status = "failed"
                job.error = str(exc)
                job.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

    def promote(self, db: Session, job_id: str) -> str:
        job = db.get(TrainingJob, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.model_family != "yolo":
            raise ValueError("Only YOLO training artifacts can be promoted in this version")
        best_model = job.artifacts.get("best_model") if job.artifacts else None
        if not best_model:
            raise FileNotFoundError("No best_model artifact found for this job")
        model_id = f"trained_{job.model_family}_{job.id[:8]}"
        info = self.registry.promote_yolo_model(
            model_id=model_id,
            weights_path=Path(best_model),
            name=f"Trained {job.model_family} {job.id[:8]}",
        )
        job.promoted_model_id = info.id
        job.updated_at = datetime.utcnow()
        db.commit()
        return info.id

    def _command_for_job(self, job: TrainingJob, run_dir: Path) -> list[str]:
        params = job.parameters
        if job.model_family == "yolo":
            dataset_root = self.settings.datasets_path / "dental dataset_yolov11_format"
            initial_weights = self.settings.models_path / "yolo_11_best" / "weights" / "best.pt"
            return [
                sys.executable,
                "-m",
                "app.training.runners.yolo_train",
                "--run-dir",
                str(run_dir),
                "--dataset-root",
                str(dataset_root),
                "--initial-weights",
                str(initial_weights),
                "--epochs",
                str(params["epochs"]),
                "--image-size",
                str(params["image_size"]),
                "--batch-size",
                str(params["batch_size"]),
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


def find_best_model(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "weights" / "best.pt",
        run_dir / "best.pt",
        run_dir / "best_unet_model.keras",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in run_dir.rglob("best.pt"):
        return candidate
    for candidate in run_dir.rglob("best_unet_model.keras"):
        return candidate
    return None
