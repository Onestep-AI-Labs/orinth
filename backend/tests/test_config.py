from app.core.config import get_settings


def test_settings_resolve_workspace_paths():
    settings = get_settings()

    assert settings.repo_root.name == "dental_segmentation_v2"
    assert settings.models_path.name == "models"
    assert settings.datasets_path.name == "datasets"
    assert settings.database_path.name == "app.db"
