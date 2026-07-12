from pathlib import Path
import json

from app.core.config import Settings, get_settings
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry, model_storage_dir_name


def test_model_registry_lists_static_models():
    settings = get_settings()
    registry = ModelRegistry(settings, Storage(settings))

    model_ids = {model.id for model in registry.list_models()}

    assert "yolo_11_best" in model_ids
    assert "unet_inception" in model_ids


def test_model_registry_renames_and_deletes_trained_models(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    registry = ModelRegistry(settings, storage)
    model_id = "trained_yolo_example"
    model_dir = storage.trained_models / model_id / "weights"
    model_dir.mkdir(parents=True)
    weights_path = model_dir / "best.pt"
    weights_path.write_text("weights", encoding="utf-8")

    registry.register_model(
        model_id=model_id,
        name="Original model",
        family="yolo",
        task_type="segmentation",
        paths={"weights": weights_path},
        labels=["granuloma", "kista"],
    )

    renamed = registry.update_model(model_id, name="Renamed model")
    renamed_dir = storage.trained_models / model_storage_dir_name("Renamed model", model_id)
    assert renamed.name == "Renamed model"
    assert renamed_dir.exists()
    assert renamed.paths["weights"].startswith(str(renamed_dir))

    registry.delete_model(model_id)

    assert model_id not in {model.id for model in registry.list_models()}
    assert not (storage.trained_models / model_id).exists()
    assert not renamed_dir.exists()


def test_model_registry_stores_trained_model_paths_relative_to_storage(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    registry = ModelRegistry(settings, storage)
    model_id = "trained_yolo_relative"
    weights_path = storage.trained_models / model_id / "weights" / "best.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_text("weights", encoding="utf-8")

    registry.register_model(
        model_id=model_id,
        name="Relative model",
        family="yolo",
        task_type="segmentation",
        paths={"weights": weights_path},
        labels=["granuloma", "kista"],
    )

    registry_payload = json.loads(storage.registry_file.read_text(encoding="utf-8"))
    stored_path = registry_payload[model_id]["paths"]["weights"]
    listed_model = next(model for model in registry.list_models() if model.id == model_id)

    assert stored_path == f"trained_models/{model_id}/weights/best.pt"
    assert listed_model.available is True
    assert listed_model.paths["weights"] == str(weights_path)


def test_model_registry_filters_models_to_exact_project(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    registry = ModelRegistry(settings, storage)
    project_a_model = "trained_yolo_project_a"
    project_b_model = "trained_yolo_project_b"
    for model_id in [project_a_model, project_b_model]:
        weights_path = storage.trained_models / model_id / "weights" / "best.pt"
        weights_path.parent.mkdir(parents=True)
        weights_path.write_text("weights", encoding="utf-8")
        registry.register_model(
            model_id=model_id,
            name=model_id,
            family="yolo",
            task_type="segmentation",
            paths={"weights": weights_path},
            labels=["granuloma", "kista"],
            project_id="project-a" if model_id == project_a_model else "project-b",
        )

    project_a_ids = {model.id for model in registry.list_models(project_id="project-a")}
    project_b_ids = {model.id for model in registry.list_models(project_id="project-b")}

    assert project_a_ids == {project_a_model}
    assert project_b_ids == {project_b_model}
    assert "yolo_11_best" not in project_a_ids


def test_model_registry_recovers_trained_model_paths_after_repo_move(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "new-repo" / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    model_id = "trained_keras_classification_moved"
    model_path = storage.trained_models / model_id / "best_model.keras"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("keras", encoding="utf-8")
    old_repo_path = tmp_path / "old-repo" / "storage" / "trained_models" / model_id / "best_model.keras"
    storage.registry_file.write_text(
        json.dumps(
            {
                model_id: {
                    "id": model_id,
                    "name": "Moved model",
                    "family": "keras_classification",
                    "description": "Moved trained model",
                    "paths": {"model": str(old_repo_path)},
                    "promoted": True,
                    "project_id": "default-research-project",
                    "task_type": "classification",
                    "labels": ["trash"],
                    "source": "trained",
                }
            }
        ),
        encoding="utf-8",
    )

    listed_model = next(model for model in ModelRegistry(settings, storage).list_models() if model.id == model_id)

    assert listed_model.available is True
    assert listed_model.paths["model"] == str(model_path)


def test_model_storage_dir_name_uses_name_and_id():
    assert (
        model_storage_dir_name("My Keras Model!", "trained_keras_classification_abcd1234")
        == "my-keras-model-trained_keras_classification_abcd1234"
    )


def test_model_registry_creates_download_archives_for_multi_asset_models(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    model_dir = settings.models_path / "unet_inception"
    model_dir.mkdir(parents=True)
    (model_dir / "best_unet_model.keras").write_text("unet", encoding="utf-8")
    (model_dir / "best_classifier_inception.keras").write_text("classifier", encoding="utf-8")

    archive_path, filename = ModelRegistry(settings, storage).model_download("unet_inception")

    assert filename.endswith(".zip")
    assert archive_path.exists()
    assert archive_path.parent == storage.model_downloads
