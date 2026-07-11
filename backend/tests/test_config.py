from app.core.config import get_settings


def test_settings_resolve_workspace_paths():
    settings = get_settings()

    assert (settings.repo_root / "backend").is_dir()
    assert (settings.repo_root / "frontend").is_dir()
    assert settings.models_path.name == "models"
    assert settings.datasets_path.name == "datasets"
    assert settings.database_path.name == "app.db"
