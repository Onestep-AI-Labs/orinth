from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry


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
    assert renamed.name == "Renamed model"

    registry.delete_model(model_id)

    assert model_id not in {model.id for model in registry.list_models()}
    assert not (storage.trained_models / model_id).exists()
