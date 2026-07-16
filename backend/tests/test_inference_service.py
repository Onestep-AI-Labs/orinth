from io import BytesIO

import anyio
import pytest
from fastapi import UploadFile
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.services.inference as inference_module
from app.core.config import Settings
from app.core.database import Base
from app.core.storage import Storage
from app.db.models import InferenceJob, InferenceRun
from app.schemas import Box, Detection, InferenceParameters
from app.services.inference import InferenceService


def upload_file(name: str = "sample.jpg", size: tuple[int, int] = (64, 64)) -> UploadFile:
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="JPEG")
    buffer.seek(0)
    return UploadFile(file=buffer, filename=name)


def make_detection(class_name: str = "granuloma", mask_area: float = 100.0) -> Detection:
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=Box(x=1, y=1, width=10, height=10),
        polygon=[],
        mask_area=mask_area,
    )


class FakeSpec:
    def __init__(self, task_type: str = "segmentation") -> None:
        self.task_type = task_type


class FakeDetectionPredictor:
    """Fake predictor exercising the detection (no classify) branch."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = detections or []

    def predict(self, image_path, parameters):
        return self._detections

    def classify(self, image_path, parameters):
        return None


class FakeClassificationPredictor:
    """Fake predictor exercising the classify() branch."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def predict(self, image_path, parameters):
        return []

    def classify(self, image_path, parameters):
        return self._scores


class FakeTextPredictor:
    def predict_text(self, text, parameters):
        return {"label": "positive", "scores": {"positive": 0.9, "negative": 0.1}}


class FakeFailingPredictor:
    def predict(self, image_path, parameters):
        raise RuntimeError("predictor exploded")

    def classify(self, image_path, parameters):
        return None


class FakeRegistry:
    def __init__(self, spec: FakeSpec, predictor) -> None:
        self._spec = spec
        self._predictor = predictor

    def get_spec(self, model_id: str) -> FakeSpec:
        return self._spec

    def get_predictor(self, model_id: str):
        return self._predictor


@pytest.fixture
def sql_engine(settings: Settings):
    engine = create_engine(
        settings.sqlalchemy_database_url,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def session_local(sql_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=sql_engine)


@pytest.fixture
def db(session_local):
    session = session_local()
    try:
        yield session
    finally:
        session.close()


def test_run_sync_detection_happy_path_persists_overlay_and_run(storage: Storage, db) -> None:
    registry = FakeRegistry(FakeSpec("segmentation"), FakeDetectionPredictor([make_detection()]))
    service = InferenceService(storage, registry)

    result = anyio.run(
        service.run,
        db,
        upload_file(),
        "fake_detector",
        InferenceParameters(),
    )

    assert result.image_level_label == "granuloma"
    assert result.detections[0].class_name == "granuloma"
    assert result.overlay_url is not None
    overlay_path = storage.root / result.overlay_url.removeprefix("/media/")
    assert overlay_path.exists()
    original_path = storage.root / result.original_url.removeprefix("/media/")
    assert original_path.exists()
    assert db.query(InferenceRun).count() == 1


def test_run_sync_classification_happy_path_uses_class_scores(storage: Storage, db) -> None:
    registry = FakeRegistry(
        FakeSpec("classification"),
        FakeClassificationPredictor({"granuloma": 0.2, "kista": 0.8}),
    )
    service = InferenceService(storage, registry)

    result = anyio.run(
        service.run,
        db,
        upload_file(),
        "fake_classifier",
        InferenceParameters(),
    )

    assert result.image_level_label == "kista"
    assert result.class_scores == {"granuloma": 0.2, "kista": 0.8}
    assert result.detections == []
    # An overlay is still rendered even with no detections.
    overlay_path = storage.root / result.overlay_url.removeprefix("/media/")
    assert overlay_path.exists()


def test_run_text_input_stores_input_without_overlay(storage: Storage, db) -> None:
    registry = FakeRegistry(FakeSpec("text_classification"), FakeTextPredictor())
    service = InferenceService(storage, registry)

    result = anyio.run(
        service.run,
        db,
        None,
        "fake_text_classifier",
        InferenceParameters(),
        "default-research-project",
        "The workflow completed successfully",
    )

    assert result.input_type == "text"
    assert result.overlay_url is None
    assert result.image_level_label == "positive"
    assert result.nlp_result == {"label": "positive", "scores": {"positive": 0.9, "negative": 0.1}}
    original_path = storage.root / result.original_url.removeprefix("/media/")
    assert original_path.exists()
    assert original_path.parent == storage.uploads


def test_job_run_lifecycle_completes_and_persists_result(
    monkeypatch: pytest.MonkeyPatch, storage: Storage, session_local
) -> None:
    monkeypatch.setattr(inference_module, "SessionLocal", session_local)
    registry = FakeRegistry(FakeSpec("segmentation"), FakeDetectionPredictor([make_detection()]))
    service = InferenceService(storage, registry)

    creation_db = session_local()
    try:
        job = anyio.run(
            service.create_job,
            creation_db,
            upload_file(),
            "fake_detector",
            InferenceParameters(),
        )
        job_id = job.id
        assert job.status == "queued"
    finally:
        creation_db.close()

    service.run_job(job_id)

    verify_db = session_local()
    try:
        job_read = service.get_job(verify_db, job_id)
        stored_job = verify_db.get(InferenceJob, job_id)
    finally:
        verify_db.close()

    assert job_read.status == "completed"
    assert stored_job.status == "completed"
    assert job_read.result is not None
    assert job_read.result.image_level_label == "granuloma"
    assert job_read.progress.percent == 100
    assert job_read.error is None


def test_run_job_persists_failed_status_on_predictor_error(
    monkeypatch: pytest.MonkeyPatch, storage: Storage, session_local
) -> None:
    monkeypatch.setattr(inference_module, "SessionLocal", session_local)
    registry = FakeRegistry(FakeSpec("segmentation"), FakeFailingPredictor())
    service = InferenceService(storage, registry)

    creation_db = session_local()
    try:
        job = anyio.run(
            service.create_job,
            creation_db,
            upload_file(),
            "fake_failing_detector",
            InferenceParameters(),
        )
        job_id = job.id
    finally:
        creation_db.close()

    service.run_job(job_id)

    verify_db = session_local()
    try:
        job_read = service.get_job(verify_db, job_id)
    finally:
        verify_db.close()

    assert job_read.status == "failed"
    assert job_read.error is not None
    assert "predictor exploded" in job_read.error
    assert job_read.result is None


def test_list_get_and_delete_runs_removes_owned_storage_files(storage: Storage, db) -> None:
    registry = FakeRegistry(FakeSpec("segmentation"), FakeDetectionPredictor([make_detection()]))
    service = InferenceService(storage, registry)

    first = anyio.run(service.run, db, upload_file("a.jpg"), "fake_detector", InferenceParameters())
    second = anyio.run(service.run, db, upload_file("b.jpg"), "fake_detector", InferenceParameters())

    listed = service.list(db)
    assert {result.id for result in listed} == {first.id, second.id}

    fetched = service.get(db, first.id)
    assert fetched.id == first.id

    overlay_path = storage.root / first.overlay_url.removeprefix("/media/")
    assert overlay_path.exists()

    deletion = service.delete_runs(db, ids=[first.id])

    assert deletion["deleted"] == 1
    assert deletion["missing"] == []
    assert not overlay_path.exists()
    assert db.query(InferenceRun).count() == 1
