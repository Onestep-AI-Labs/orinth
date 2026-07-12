"""Shared pytest fixtures for the backend test suite.

These fixtures replace the near-identical `make_settings(tmp_path)` helpers
that used to be copy-pasted across individual test modules.
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.database import Base
from app.core.storage import Storage
from app.db import models  # noqa: F401 - ensures all ORM models are registered on Base


@pytest.fixture
def make_settings(tmp_path: Path):
    """Return a factory for building an isolated `Settings` instance rooted at tmp_path.

    Call it with no arguments for the common case, or pass field overrides
    (e.g. ``make_settings(HF_TOKEN="hf_alias_token")``) for tests that need
    to tweak specific settings.
    """

    def _make_settings(**overrides: object) -> Settings:
        defaults = {
            "MODELS_DIR": str(tmp_path / "models"),
            "DATASETS_DIR": str(tmp_path / "datasets"),
            "STORAGE_DIR": str(tmp_path / "storage"),
            "DATABASE_URL": f"sqlite:///{tmp_path / 'app.db'}",
        }
        defaults.update(overrides)
        return Settings(**defaults)

    return _make_settings


@pytest.fixture
def settings(make_settings) -> Settings:
    """A ready-to-use `Settings` instance rooted at a fresh tmp_path."""

    return make_settings()


@pytest.fixture
def storage(settings: Settings) -> Storage:
    """A `Storage` instance with its directory tree already created."""

    storage = Storage(settings)
    storage.ensure()
    return storage


@pytest.fixture
def db_session(settings: Settings) -> Generator[Session, None, None]:
    """A SQLite-backed SQLAlchemy session using the shared `settings` fixture."""

    engine = create_engine(
        settings.sqlalchemy_database_url,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_local()
    try:
        yield db
    finally:
        db.close()
