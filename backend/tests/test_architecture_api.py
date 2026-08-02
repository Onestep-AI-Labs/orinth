"""Architecture studio endpoints and service behavior (phase 17)."""

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes import router
from app.container import architecture_service
from app.core.config import Settings
from app.core.database import Base, get_db
from app.core.storage import Storage
from app.db import models  # noqa: F401 - registers ORM models on Base
from app.ml.architecture.templates import chain, node
from app.schemas import ArchitectureGraph


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'architectures.db'}",
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

    # The container's service is a module-level singleton pointed at the real
    # storage root; redirect its snapshots into tmp_path for the test.
    storage = Storage(
        Settings(
            MODELS_DIR=str(tmp_path / "models"),
            DATASETS_DIR=str(tmp_path / "datasets"),
            STORAGE_DIR=str(tmp_path / "storage"),
            DATABASE_URL=f"sqlite:///{tmp_path / 'architectures.db'}",
        )
    )
    storage.ensure()
    monkeypatch.setattr(architecture_service, "storage", storage)

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client:
        yield test_client


def create(client: TestClient, **overrides) -> dict:
    payload = {"name": "Test architecture", "task_type": "classification", **overrides}
    response = client.post("/api/architectures", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def tiny_graph() -> dict:
    return ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("out", "output"),
        ],
        edges=chain("in", "gap", "head", "out"),
    ).model_dump()


# --- catalog and templates -------------------------------------------------


def test_node_catalog_is_served_with_renderable_param_specs(client: TestClient):
    response = client.get("/api/architectures/node-catalog")

    assert response.status_code == 200
    specs = {spec["type"]: spec for spec in response.json()}
    assert {"input", "output", "conv2d", "dense"} <= set(specs)
    # Params must arrive in the same shape the training form's advanced
    # accordion already renders, so the inspector needs no new field renderer.
    conv_params = {param["key"]: param for param in specs["conv2d"]["params"]}
    assert conv_params["padding"]["type"] == "select"
    assert conv_params["padding"]["options"] == ["same", "valid"]
    assert conv_params["filters"]["type"] == "int"


def test_node_catalog_filters_by_task_type(client: TestClient):
    response = client.get("/api/architectures/node-catalog", params={"task_type": "classification"})

    assert "embedding" not in {spec["type"] for spec in response.json()}


def test_node_categories_are_served_in_display_order(client: TestClient):
    response = client.get("/api/architectures/node-categories")

    assert response.status_code == 200
    assert response.json()[0] == "Input & Output"


def test_templates_are_served_and_filterable(client: TestClient):
    every = client.get("/api/architectures/templates").json()
    image_only = client.get(
        "/api/architectures/templates", params={"task_type": "classification"}
    ).json()

    assert {"small_cnn", "residual_block", "bilstm_text"} <= {item["id"] for item in every}
    assert "bilstm_text" not in {item["id"] for item in image_only}


def test_literal_paths_are_not_captured_as_an_architecture_id(client: TestClient):
    """`/node-catalog` must not be read as `/{architecture_id}`."""

    assert client.get("/api/architectures/node-catalog").status_code == 200
    assert client.get("/api/architectures/templates").status_code == 200
    assert client.get("/api/architectures/does-not-exist").status_code == 404


# --- CRUD ------------------------------------------------------------------


def test_create_from_a_template_copies_its_graph(client: TestClient):
    created = create(client, template_id="small_cnn", name="From template")

    assert created["version"] == 1
    assert len(created["graph"]["nodes"]) == 13
    assert created["id"].startswith("arch_")


def test_create_from_an_unknown_template_is_a_404(client: TestClient):
    response = client.post(
        "/api/architectures", json={"name": "Nope", "template_id": "does_not_exist"}
    )

    assert response.status_code == 404


def test_list_returns_summaries_without_the_graph(client: TestClient):
    create(client, template_id="small_cnn", name="Listed")

    listed = client.get("/api/architectures").json()

    assert listed[0]["name"] == "Listed"
    assert listed[0]["node_count"] == 13
    assert "graph" not in listed[0]


def test_list_filters_by_project_and_task_type(client: TestClient):
    create(client, name="Alpha", project_id="p1", task_type="classification")
    create(client, name="Beta", project_id="p2", task_type="classification")
    create(client, name="Gamma", project_id="p1", task_type="text_classification")

    by_project = client.get("/api/architectures", params={"project_id": "p1"}).json()
    by_task = client.get(
        "/api/architectures", params={"project_id": "p1", "task_type": "text_classification"}
    ).json()

    assert {item["name"] for item in by_project} == {"Alpha", "Gamma"}
    assert {item["name"] for item in by_task} == {"Gamma"}


def test_saving_a_graph_bumps_the_version(client: TestClient):
    created = create(client)

    saved = client.put(
        f"/api/architectures/{created['id']}",
        json={"graph": tiny_graph(), "version": created["version"]},
    )

    assert saved.status_code == 200
    assert saved.json()["version"] == 2


def test_renaming_without_a_graph_does_not_bump_the_version(client: TestClient):
    created = create(client)

    saved = client.put(f"/api/architectures/{created['id']}", json={"name": "Renamed"})

    assert saved.json()["name"] == "Renamed"
    assert saved.json()["version"] == created["version"]


def test_a_stale_version_is_rejected_rather_than_overwriting(client: TestClient):
    created = create(client)
    client.put(
        f"/api/architectures/{created['id']}", json={"graph": tiny_graph(), "version": 1}
    )

    # A second session still holding version 1 tries to save.
    conflict = client.put(
        f"/api/architectures/{created['id']}", json={"graph": tiny_graph(), "version": 1}
    )

    assert conflict.status_code == 409
    assert "saved elsewhere" in conflict.json()["detail"]


def test_every_save_is_snapshotted_to_storage(client: TestClient, tmp_path: Path):
    created = create(client)
    client.put(
        f"/api/architectures/{created['id']}", json={"graph": tiny_graph(), "version": 1}
    )

    snapshots = sorted(
        path.name for path in (architecture_service.storage.architectures / created["id"]).iterdir()
    )

    assert snapshots == ["v1.json", "v2.json"]


def test_duplicate_copies_the_graph_into_a_new_architecture(client: TestClient):
    created = create(client, template_id="small_cnn")

    copy = client.post(f"/api/architectures/{created['id']}/duplicate").json()

    assert copy["id"] != created["id"]
    assert copy["name"] == f"{created['name']} copy"
    assert copy["version"] == 1
    assert len(copy["graph"]["nodes"]) == len(created["graph"]["nodes"])


def test_delete_removes_the_row_and_its_snapshots(client: TestClient):
    created = create(client)
    directory = architecture_service.storage.architectures / created["id"]
    assert directory.exists()

    deleted = client.delete(f"/api/architectures/{created['id']}")

    assert deleted.json()["deleted"] == 1
    assert client.get(f"/api/architectures/{created['id']}").status_code == 404
    assert not directory.exists()


# --- validation ------------------------------------------------------------


def test_validate_returns_shapes_for_an_unsaved_graph(client: TestClient):
    response = client.post(
        "/api/architectures/validate", json=tiny_graph(), params={"num_classes": 4}
    )

    body = response.json()
    assert body["ok"] is True
    assert body["node_shapes"]["gap"] == [3]
    assert body["node_shapes"]["head"] == [4]
    assert body["layer_count"] == 4


def test_validate_reports_errors_without_failing_the_request(client: TestClient):
    graph = tiny_graph()
    graph["edges"].append({"id": "loop", "source": "head", "target": "gap"})

    response = client.post("/api/architectures/validate", json=graph)

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert any(issue["severity"] == "error" for issue in body["issues"])


def test_validate_on_a_saved_architecture_reads_its_stored_graph(client: TestClient):
    created = create(client, graph=tiny_graph())

    response = client.post(f"/api/architectures/{created['id']}/validate")

    assert response.json()["ok"] is True


# --- code generation -------------------------------------------------------


def test_code_endpoint_returns_a_runnable_module(client: TestClient):
    created = create(client, graph=tiny_graph())

    body = client.get(
        f"/api/architectures/{created['id']}/code", params={"num_classes": 5}
    ).json()

    assert body["filename"] == "generated_model.py"
    assert "def build_model(num_classes: int = 5)" in body["code"]
    assert "import tensorflow as tf" in body["code"]


def test_code_download_is_served_as_an_attachment(client: TestClient):
    created = create(client, graph=tiny_graph())

    response = client.get(f"/api/architectures/{created['id']}/code/download")

    assert response.headers["content-type"].startswith("text/x-python")
    assert "generated_model.py" in response.headers["content-disposition"]


def test_code_generation_on_a_broken_graph_explains_what_to_fix(client: TestClient):
    graph = tiny_graph()
    graph["edges"].append({"id": "loop", "source": "head", "target": "gap"})
    created = create(client, graph=graph)

    response = client.get(f"/api/architectures/{created['id']}/code")

    assert response.status_code == 400
    assert "part of a loop" in response.json()["detail"]


# --- import and export -----------------------------------------------------


def test_export_then_import_round_trips_the_graph(client: TestClient):
    created = create(client, template_id="residual_block", name="Round trip")
    exported = client.get(f"/api/architectures/{created['id']}/export")
    assert exported.headers["content-disposition"].endswith(f'{created["id"]}.json"')

    imported = client.post(
        "/api/architectures/import",
        files={"file": ("arch.json", exported.content, "application/json")},
    )

    assert imported.status_code == 200
    body = imported.json()
    assert body["id"] != created["id"]
    assert body["name"] == "Round trip"
    assert body["graph"] == created["graph"]


def test_import_lays_out_a_graph_that_has_no_positions(client: TestClient):
    payload = {
        "schema_version": 1,
        "name": "Handwritten",
        "task_type": "classification",
        "graph": {
            "nodes": [
                {"id": "in", "type": "input", "params": {"shape": "8,8,3"}},
                {"id": "gap", "type": "global_avg_pool2d"},
                {"id": "out", "type": "output"},
            ],
            "edges": [
                {"id": "e1", "source": "in", "target": "gap"},
                {"id": "e2", "source": "gap", "target": "out"},
            ],
        },
    }

    body = client.post(
        "/api/architectures/import",
        files={"file": ("arch.json", json.dumps(payload).encode(), "application/json")},
    ).json()

    positions = [tuple(item["position"].values()) for item in body["graph"]["nodes"]]
    assert len(set(positions)) == 3


def test_import_rejects_a_newer_schema_version(client: TestClient):
    payload = json.dumps({"schema_version": 99, "name": "Future", "graph": {}}).encode()

    response = client.post(
        "/api/architectures/import",
        files={"file": ("arch.json", payload, "application/json")},
    )

    assert response.status_code == 400
    assert "newer version" in response.json()["detail"]


def test_import_rejects_a_file_that_is_not_json(client: TestClient):
    response = client.post(
        "/api/architectures/import",
        files={"file": ("arch.json", b"not json at all", "application/json")},
    )

    assert response.status_code == 400
    assert "not valid JSON" in response.json()["detail"]
