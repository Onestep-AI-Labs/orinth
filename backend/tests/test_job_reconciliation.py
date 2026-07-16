"""Startup reconciliation tests for training, evaluation, and inference jobs.

Each service's `reconcile_stale_jobs` must mark orphaned `queued`/`running`
rows as `failed` (simulating a server restart while a job was in flight) and
must leave terminal rows (`completed`, `failed`, `canceled`) untouched.
"""

import app.services.evaluation.service as evaluation_module
import app.services.inference as inference_module
import app.services.training.service as training_module
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.core.database import Base
from app.db.models import EvaluationJob, InferenceJob, TrainingJob
from app.services.evaluation import EvaluationService
from app.services.inference import InferenceService
from app.services.training import TrainingService


@pytest.fixture
def session_local(settings: Settings):
    engine = create_engine(
        settings.sqlalchemy_database_url,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def test_training_reconciliation_fails_stale_rows_and_spares_terminal_rows(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, storage, session_local
) -> None:
    monkeypatch.setattr(training_module, "SessionLocal", session_local)
    from app.ml.model_registry import ModelRegistry

    service = TrainingService(settings, storage, ModelRegistry(settings, storage))

    db = session_local()
    try:
        db.add_all(
            [
                TrainingJob(id="queued-job", model_family="yolo", status="queued"),
                TrainingJob(id="running-job", model_family="yolo", status="running"),
                TrainingJob(id="completed-job", model_family="yolo", status="completed"),
                TrainingJob(id="failed-job", model_family="yolo", status="failed", error="boom"),
                TrainingJob(id="canceled-job", model_family="yolo", status="canceled"),
            ]
        )
        db.commit()
    finally:
        db.close()

    service.reconcile_stale_jobs()

    db = session_local()
    try:
        assert db.get(TrainingJob, "queued-job").status == "failed"
        assert db.get(TrainingJob, "running-job").status == "failed"
        assert db.get(TrainingJob, "completed-job").status == "completed"
        failed_job = db.get(TrainingJob, "failed-job")
        assert failed_job.status == "failed"
        assert failed_job.error == "boom"
        assert db.get(TrainingJob, "canceled-job").status == "canceled"
    finally:
        db.close()


def test_evaluation_reconciliation_fails_stale_rows_and_spares_terminal_rows(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, storage, session_local
) -> None:
    monkeypatch.setattr(evaluation_module, "SessionLocal", session_local)
    from app.ml.model_registry import ModelRegistry

    service = EvaluationService(settings, storage, ModelRegistry(settings, storage))

    db = session_local()
    try:
        db.add_all(
            [
                EvaluationJob(id="queued-job", model_id="m", dataset_key="k", status="queued"),
                EvaluationJob(id="running-job", model_id="m", dataset_key="k", status="running"),
                EvaluationJob(id="completed-job", model_id="m", dataset_key="k", status="completed"),
                EvaluationJob(
                    id="failed-job", model_id="m", dataset_key="k", status="failed", error="boom"
                ),
                EvaluationJob(id="canceled-job", model_id="m", dataset_key="k", status="canceled"),
            ]
        )
        db.commit()
    finally:
        db.close()

    service.reconcile_stale_jobs()

    db = session_local()
    try:
        assert db.get(EvaluationJob, "queued-job").status == "failed"
        assert db.get(EvaluationJob, "running-job").status == "failed"
        assert db.get(EvaluationJob, "completed-job").status == "completed"
        failed_job = db.get(EvaluationJob, "failed-job")
        assert failed_job.status == "failed"
        assert failed_job.error == "boom"
        assert db.get(EvaluationJob, "canceled-job").status == "canceled"
    finally:
        db.close()


def test_inference_reconciliation_fails_stale_rows_and_spares_terminal_rows(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, storage, session_local
) -> None:
    monkeypatch.setattr(inference_module, "SessionLocal", session_local)
    from app.ml.model_registry import ModelRegistry

    service = InferenceService(storage, ModelRegistry(settings, storage))

    db = session_local()
    try:
        db.add_all(
            [
                InferenceJob(id="queued-job", model_id="m", status="queued", input_path="in"),
                InferenceJob(id="running-job", model_id="m", status="running", input_path="in"),
                InferenceJob(id="completed-job", model_id="m", status="completed", input_path="in"),
                InferenceJob(
                    id="failed-job", model_id="m", status="failed", input_path="in", error="boom"
                ),
            ]
        )
        db.commit()
    finally:
        db.close()

    service.reconcile_stale_jobs()

    db = session_local()
    try:
        assert db.get(InferenceJob, "queued-job").status == "failed"
        assert db.get(InferenceJob, "running-job").status == "failed"
        assert db.get(InferenceJob, "completed-job").status == "completed"
        failed_job = db.get(InferenceJob, "failed-job")
        assert failed_job.status == "failed"
        assert failed_job.error == "boom"
    finally:
        db.close()
