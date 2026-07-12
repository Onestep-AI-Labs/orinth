from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Onestep AI Platform"
    api_prefix: str = "/api"
    models_dir: str = Field(default="./models", alias="MODELS_DIR")
    datasets_dir: str = Field(default="./datasets", alias="DATASETS_DIR")
    storage_dir: str = Field(default="./storage", alias="STORAGE_DIR")
    database_url: str = Field(default="sqlite:///./storage/app.db", alias="DATABASE_URL")
    backend_cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="BACKEND_CORS_ORIGINS",
    )
    huggingface_hub_token: str | None = Field(default=None, alias="HUGGINGFACE_HUB_TOKEN")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    job_executor_workers: int = Field(default=2, alias="JOB_EXECUTOR_WORKERS")

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.repo_root / path).resolve()

    @property
    def models_path(self) -> Path:
        return self.resolve_path(self.models_dir)

    @property
    def datasets_path(self) -> Path:
        return self.resolve_path(self.datasets_dir)

    @property
    def storage_path(self) -> Path:
        return self.resolve_path(self.storage_dir)

    @property
    def database_path(self) -> Path:
        if not self.database_url.startswith("sqlite:///"):
            return self.storage_path / "app.db"
        raw_path = self.database_url.replace("sqlite:///", "", 1)
        return self.resolve_path(raw_path)

    @property
    def sqlalchemy_database_url(self) -> str:
        if not self.database_url.startswith("sqlite:///"):
            return self.database_url
        return f"sqlite:///{self.database_path}"

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]

    @property
    def huggingface_token(self) -> str | None:
        token = (self.hf_token or self.huggingface_hub_token or "").strip()
        return token or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
