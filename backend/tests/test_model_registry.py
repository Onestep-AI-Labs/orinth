from app.core.config import get_settings
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry


def test_model_registry_lists_static_models():
    settings = get_settings()
    registry = ModelRegistry(settings, Storage(settings))

    model_ids = {model.id for model in registry.list_models()}

    assert "yolo_11_best" in model_ids
    assert "unet_inception" in model_ids
