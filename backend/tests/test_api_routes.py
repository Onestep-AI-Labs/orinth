from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes import router
from app.core.database import Base, get_db
from app.db import models  # noqa: F401


@pytest.fixture
def client(tmp_path: Path) -> Generator[TestClient, None, None]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'api-routes.db'}",
        connect_args={"check_same_thread": False},
    )
    testing_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_db() -> Generator[Session, None, None]:
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_db] = override_db

    with TestClient(app) as test_client:
        yield test_client


def test_core_api_routes_are_wired(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    models_response = client.get("/api/models")
    assert models_response.status_code == 200
    assert isinstance(models_response.json(), list)

    projects_response = client.get("/api/projects")
    assert projects_response.status_code == 200
    assert projects_response.json()[0]["id"] == "default-research-project"

    datasets_response = client.get("/api/datasets")
    assert datasets_response.status_code == 200
    assert isinstance(datasets_response.json(), list)

    testing_datasets_response = client.get("/api/testing/datasets")
    assert testing_datasets_response.status_code == 200
    assert isinstance(testing_datasets_response.json(), list)

    training_options_response = client.get("/api/training/model-options?task_type=classification")
    assert training_options_response.status_code == 200
    assert any(option["family"] == "keras_classification" for option in training_options_response.json())
