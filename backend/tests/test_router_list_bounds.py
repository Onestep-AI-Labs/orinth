"""HTTP-level `TestClient` tests for the training/testing/inference routers.

`test_api_routes.py` only asserts these routers are wired up; it never drives
the pagination bounds or the 404/409/400 error mapping that each router's
`except` clauses translate service-layer exceptions into. This module covers:

- Bounded `limit` (`ge=1, le=1000`) and `offset` (`ge=0`) validation on the
  three list endpoints (`GET /training/jobs`, `GET /testing/jobs`,
  `GET /inference`), mirroring the datasets items endpoint.
- `offset` paging behavior (newest-first, offset past the end -> empty list)
  via the shared `job_runner.list_jobs` helper.
- 404 mapping for unknown ids, 409 mapping for `ValueError`/domain conflicts,
  and 400 mapping for request-shape errors raised directly by services.
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app import container
from app.api.routes import router
from app.core.database import Base, get_db
from app.db import models  # noqa: F401 - ensures all ORM models are registered on Base
from app.db.models import EvaluationJob, TrainingJob
from app.ml.model_registry import ModelSpec

LIST_ENDPOINTS = ["/api/training/jobs", "/api/testing/jobs", "/api/inference"]


@pytest.fixture
def api(tmp_path: Path) -> Generator[tuple[TestClient, sessionmaker], None, None]:
    """A `TestClient` wired to the real routers, plus the raw session factory.

    The session factory lets tests seed rows directly (bypassing the heavy
    service/job-executor creation paths) to exercise list/pagination
    behavior without spinning up real training/evaluation/inference jobs.
    """

    engine = create_engine(
        f"sqlite:///{tmp_path / 'router-list-bounds.db'}",
        connect_args={"check_same_thread": False},
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_db() -> Generator[Session, None, None]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_db] = override_db

    with TestClient(app) as test_client:
        yield test_client, session_factory


def _insert(session_factory: sessionmaker, model, **fields: object) -> None:
    db = session_factory()
    try:
        db.add(model(**fields))
        db.commit()
    finally:
        db.close()


# --- Pagination validation: out-of-range limit / negative offset -> 422 ---


@pytest.mark.parametrize("path", LIST_ENDPOINTS)
@pytest.mark.parametrize("limit", [0, -1, 1001])
def test_list_endpoints_reject_out_of_range_limit(api, path, limit) -> None:
    client, _ = api
    response = client.get(path, params={"limit": limit})
    assert response.status_code == 422


@pytest.mark.parametrize("path", LIST_ENDPOINTS)
def test_list_endpoints_reject_negative_offset(api, path) -> None:
    client, _ = api
    response = client.get(path, params={"offset": -1})
    assert response.status_code == 422


@pytest.mark.parametrize("path", LIST_ENDPOINTS)
def test_list_endpoints_accept_default_pagination(api, path) -> None:
    client, _ = api
    response = client.get(path)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize("path", LIST_ENDPOINTS)
def test_list_endpoints_accept_boundary_limit_values(api, path) -> None:
    client, _ = api
    assert client.get(path, params={"limit": 1}).status_code == 200
    assert client.get(path, params={"limit": 1000}).status_code == 200


# --- offset behavior: newest first, offset past the end -> empty list, not error ---


def test_training_jobs_list_orders_newest_first_and_pages_with_offset(api) -> None:
    client, session_factory = api
    _insert(session_factory, TrainingJob, id="job-1", model_family="yolo", status="completed")
    _insert(session_factory, TrainingJob, id="job-2", model_family="yolo", status="completed")
    _insert(session_factory, TrainingJob, id="job-3", model_family="yolo", status="completed")

    first_page = client.get("/api/training/jobs", params={"limit": 2, "offset": 0})
    assert first_page.status_code == 200
    assert [job["id"] for job in first_page.json()] == ["job-3", "job-2"]

    second_page = client.get("/api/training/jobs", params={"limit": 2, "offset": 2})
    assert second_page.status_code == 200
    assert [job["id"] for job in second_page.json()] == ["job-1"]

    past_the_end = client.get("/api/training/jobs", params={"limit": 2, "offset": 100})
    assert past_the_end.status_code == 200
    assert past_the_end.json() == []


def test_testing_jobs_list_pages_with_offset_past_the_end_returns_empty(api) -> None:
    client, session_factory = api
    _insert(
        session_factory,
        EvaluationJob,
        id="eval-1",
        model_id="model-a",
        dataset_key="dataset-a",
        status="completed",
    )

    default_page = client.get("/api/testing/jobs")
    assert default_page.status_code == 200
    assert [job["id"] for job in default_page.json()] == ["eval-1"]

    past_the_end = client.get("/api/testing/jobs", params={"offset": 1})
    assert past_the_end.status_code == 200
    assert past_the_end.json() == []


# --- 404 mapping: unknown ids ---


def test_get_training_job_returns_404_for_unknown_id(api) -> None:
    client, _ = api
    response = client.get("/api/training/jobs/does-not-exist")
    assert response.status_code == 404


def test_cancel_training_job_returns_404_for_unknown_id(api) -> None:
    client, _ = api
    response = client.post("/api/training/jobs/does-not-exist/cancel")
    assert response.status_code == 404


def test_promote_training_job_returns_404_for_unknown_id(api) -> None:
    client, _ = api
    response = client.post("/api/training/jobs/does-not-exist/promote")
    assert response.status_code == 404


def test_get_testing_job_returns_404_for_unknown_id(api) -> None:
    client, _ = api
    response = client.get("/api/testing/jobs/does-not-exist")
    assert response.status_code == 404


def test_get_testing_comparison_returns_404_for_unknown_id(api) -> None:
    client, _ = api
    response = client.get("/api/testing/jobs/does-not-exist/comparison")
    assert response.status_code == 404


def test_get_inference_returns_404_for_unknown_id(api) -> None:
    client, _ = api
    response = client.get("/api/inference/does-not-exist")
    assert response.status_code == 404


def test_get_inference_job_returns_404_for_unknown_id(api) -> None:
    client, _ = api
    response = client.get("/api/inference/jobs/does-not-exist")
    assert response.status_code == 404


# --- 409 mapping: domain conflicts (ValueError) ---


def test_create_training_job_returns_409_for_unknown_model_option(api) -> None:
    client, _ = api
    response = client.post("/api/training/jobs", json={"model_option_id": "does-not-exist"})
    assert response.status_code == 409


# --- 404 mapping on create: unknown model/dataset (KeyError) ---


def test_create_testing_job_returns_404_for_unknown_model(api) -> None:
    client, _ = api
    response = client.post(
        "/api/testing/jobs",
        json={"model_id": "does-not-exist", "dataset_key": "does-not-exist"},
    )
    assert response.status_code == 404


# --- 400 mapping: request-shape errors raised directly by the service ---


def test_create_inference_returns_400_when_text_content_missing(api, monkeypatch) -> None:
    client, _ = api
    fake_spec = ModelSpec(
        id="fake-text-model",
        name="Fake Text Model",
        family="nlp",
        description="test double for NLP-branch validation",
        paths={},
        task_type="text_classification",
    )
    monkeypatch.setattr(container.registry, "get_spec", lambda model_id: fake_spec)

    response = client.post("/api/inference", data={"model_id": "fake-text-model"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Text content is required for NLP inference"


def test_create_inference_returns_400_when_image_file_missing(api) -> None:
    client, _ = api
    response = client.post("/api/inference", data={"model_id": "yolo_11_best"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Image file is required for vision inference"
