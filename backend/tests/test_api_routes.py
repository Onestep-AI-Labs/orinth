from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes import router
from app.api.routers import settings as settings_router
from app.core.config import Settings
from app.core.database import Base, get_db
from app.db import models  # noqa: F401
from app.services.settings import SettingsService


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

    llm_environment_response = client.get("/api/training/llm/environment")
    assert llm_environment_response.status_code == 200
    assert llm_environment_response.json()["recommended_backend"] in {"unsloth", "peft"}


def test_delete_project_ignores_shared_sample_datasets(client: TestClient) -> None:
    """A project owning nothing is deletable even though samples are visible to it.

    The shared ``sample_*`` NLP datasets are injected into every project's listing
    but belong to none of them, and they are read-only — so counting them as
    blockers made every non-default project permanently undeletable.
    """
    created = client.post(
        "/api/projects",
        # Shared samples are filtered to the project's own task types, so the
        # project has to declare one a sample covers for the precondition to hold.
        json={"name": "Disposable Project", "task_types": ["text_classification"]},
    )
    assert created.status_code == 200
    project_id = created.json()["id"]

    visible = client.get(f"/api/datasets?project_id={project_id}").json()
    assert any(dataset["shared"] for dataset in visible), (
        "precondition: shared samples should be visible to the new project"
    )

    deleted = client.delete(f"/api/projects/{project_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] == 1

    remaining = {dataset["id"] for dataset in client.get("/api/datasets").json()}
    assert {
        "sample_image_classification",
        "sample_text_classification",
        "sample_summarization",
        "sample_question_answering",
    }.issubset(remaining), "sample data must survive project deletion"


def test_shared_samples_are_filtered_to_the_project_task_types(client: TestClient) -> None:
    """A project is only offered the starter samples it can actually use.

    Listing every sample in every project put image datasets in front of an
    NLP-only workspace, where none of the training options accept them.
    """
    nlp_project = client.post(
        "/api/projects",
        json={"name": "NLP Only", "task_types": ["text_classification", "summarization"]},
    ).json()["id"]
    vision_project = client.post(
        "/api/projects", json={"name": "Vision Only", "task_types": ["classification"]}
    ).json()["id"]

    nlp_visible = {d["id"] for d in client.get(f"/api/datasets?project_id={nlp_project}").json()}
    assert {"sample_text_classification", "sample_summarization"}.issubset(nlp_visible)
    assert "sample_image_classification" not in nlp_visible
    # question_answering is a sample task but not one this project declared.
    assert "sample_question_answering" not in nlp_visible

    vision_visible = {
        d["id"] for d in client.get(f"/api/datasets?project_id={vision_project}").json()
    }
    assert "sample_image_classification" in vision_visible
    assert not any(dataset.startswith("sample_") and "classification" not in dataset for dataset in vision_visible)
    assert "sample_summarization" not in vision_visible


def test_settings_routes_save_hf_token_to_temp_env(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_settings,
) -> None:
    env_path = tmp_path / ".env"

    def build_settings(token: str | None = None) -> Settings:
        return make_settings(
            DATABASE_URL=f"sqlite:///{tmp_path / 'settings.db'}",
            HF_TOKEN="",
            HUGGINGFACE_HUB_TOKEN=token or "",
        )

    service = SettingsService(build_settings(), env_path=env_path)

    def refresh_settings() -> None:
        token = None
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("HUGGINGFACE_HUB_TOKEN="):
                    token = line.split("=", 1)[1] or None
        service.settings = build_settings(token)

    monkeypatch.setattr(settings_router, "settings_service", service)
    monkeypatch.setattr(settings_router, "refresh_settings", refresh_settings)
    env_path.write_text("HF_TOKEN=legacy_alias\n", encoding="utf-8")

    unset = {
        "huggingface_hub_token_configured": False,
        "openrouter_api_key_configured": False,
        "openrouter_model": None,
    }
    assert client.get("/api/settings").json() == unset

    saved = client.patch("/api/settings", json={"huggingface_hub_token": "hf_test_token"})
    assert saved.status_code == 200
    assert saved.json() == {**unset, "huggingface_hub_token_configured": True}
    saved_env = env_path.read_text(encoding="utf-8")
    assert "hf_test_token" in saved_env
    assert "HF_TOKEN" not in saved_env
    assert "hf_test_token" not in str(saved.json())

    rejected = client.patch("/api/settings", json={"huggingface_hub_token": "bad\ntoken"})
    assert rejected.status_code == 400

    cleared = client.patch("/api/settings", json={"huggingface_hub_token": None})
    assert cleared.status_code == 200
    assert cleared.json() == unset
    assert "HUGGINGFACE_HUB_TOKEN" not in env_path.read_text(encoding="utf-8")
