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

    app_name: str = "Orinth"
    api_prefix: str = "/api"
    models_dir: str = Field(default="./models", alias="MODELS_DIR")
    datasets_dir: str = Field(default="./datasets", alias="DATASETS_DIR")
    storage_dir: str = Field(default="./storage", alias="STORAGE_DIR")
    # Tracked starter datasets shared read-only with every project. Overridable
    # so a packaged build (the macOS app) can point at its own copy instead of
    # depending on this file's depth below the repo root.
    sample_data_dir: str = Field(default="./sample_data", alias="SAMPLE_DATA_DIR")
    database_url: str = Field(default="sqlite:///./storage/app.db", alias="DATABASE_URL")
    backend_cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="BACKEND_CORS_ORIGINS",
    )
    huggingface_hub_token: str | None = Field(default=None, alias="HUGGINGFACE_HUB_TOKEN")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    # OpenRouter powers phase-11 LLM-assisted record generation. The key is
    # write-only from the UI's perspective (see SettingsService) and is injected
    # server-side at call time — never echoed to the client or written to a
    # recipe manifest.
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_model: str | None = Field(default=None, alias="OPENROUTER_MODEL")
    # Separate pools per job domain so long-running training/evaluation jobs
    # can't starve inference (or each other) out of worker threads.
    training_executor_workers: int = Field(default=1, alias="TRAINING_EXECUTOR_WORKERS")
    evaluation_executor_workers: int = Field(default=1, alias="EVALUATION_EXECUTOR_WORKERS")
    inference_executor_workers: int = Field(default=2, alias="INFERENCE_EXECUTOR_WORKERS")
    recipe_executor_workers: int = Field(default=1, alias="RECIPE_EXECUTOR_WORKERS")
    # Upload cap for custom model weights (phase 12). Default 10 GB admits
    # multi-GB LLM directories while still bounding disk use; it also caps
    # zip-bomb expansion.
    model_upload_max_bytes: int = Field(
        default=10 * 1024 * 1024 * 1024, alias="MODEL_UPLOAD_MAX_BYTES"
    )
    # LLM export jobs (phase 15) run on their own single-worker pool so a long
    # merge/GGUF conversion can't starve evaluations (or vice versa).
    export_executor_workers: int = Field(default=1, alias="EXPORT_EXECUTOR_WORKERS")
    # llama.cpp serving (phase 15). The port range is walked for the first free
    # port; the server binds localhost only. Idle timeout stops the subprocess
    # after no chat activity; the ready timeout bounds startup (model load can
    # take minutes for large GGUFs on first touch).
    serving_port_range: str = Field(default="8600-8699", alias="SERVING_PORT_RANGE")
    serving_idle_timeout_seconds: int = Field(default=900, alias="SERVING_IDLE_TIMEOUT_SECONDS")
    serving_ready_timeout_seconds: int = Field(default=600, alias="SERVING_READY_TIMEOUT_SECONDS")

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
    def sample_data_path(self) -> Path:
        return self.resolve_path(self.sample_data_dir)

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

    @property
    def openrouter_key(self) -> str | None:
        token = (self.openrouter_api_key or "").strip()
        return token or None

    @property
    def serving_ports(self) -> tuple[int, int]:
        """Parsed inclusive (start, end) of `serving_port_range`; malformed input falls back to the default."""
        raw = self.serving_port_range.strip()
        try:
            start_text, _, end_text = raw.partition("-")
            start, end = int(start_text), int(end_text or start_text)
        except ValueError:
            return (8600, 8699)
        if start < 1 or end > 65535 or end < start:
            return (8600, 8699)
        return (start, end)


@lru_cache
def get_settings() -> Settings:
    return Settings()
