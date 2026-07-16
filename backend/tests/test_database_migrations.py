"""Regression tests for the pre-Alembic-to-Alembic stamping guard.

See `app.core.database._stamp_or_upgrade`: a legacy SQLite database that has
application tables but no `alembic_version` table must only be stamped at
the baseline revision when its schema actually matches that baseline. If it
predates columns/tables the baseline expects, stamping it would silently
leave it broken instead of raising.
"""

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from app.core.config import Settings
from app.core.database import Base, _stamp_or_upgrade
from app.db import models  # noqa: F401 - registers ORM models on Base


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )


def test_stamp_or_upgrade_rejects_legacy_schema_missing_columns(tmp_path: Path) -> None:
    """A pre-Alembic DB missing a column the baseline expects must raise, not silently stamp."""

    settings = _make_settings(tmp_path)
    engine = create_engine(settings.sqlalchemy_database_url, connect_args={"check_same_thread": False})

    # Simulate a database created before `project_id`/`comparison_id` and the
    # `projects`/`inference_jobs` tables existed: only a bare `inference_runs`
    # table, no `alembic_version` table.
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE inference_runs (id VARCHAR PRIMARY KEY)"))

    with pytest.raises(RuntimeError, match="does not match"):
        _stamp_or_upgrade(target_engine=engine, target_settings=settings)

    # And it must not have been stamped as a side effect of the failed attempt.
    assert "alembic_version" not in inspect(engine).get_table_names()


def test_stamp_or_upgrade_stamps_a_schema_that_actually_matches_baseline(tmp_path: Path) -> None:
    """A pre-Alembic DB whose schema is already fully current stamps and upgrades cleanly."""

    settings = _make_settings(tmp_path)
    engine = create_engine(settings.sqlalchemy_database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)

    _stamp_or_upgrade(target_engine=engine, target_settings=settings)

    assert "alembic_version" in inspect(engine).get_table_names()


def test_stamp_or_upgrade_upgrades_a_brand_new_database(tmp_path: Path) -> None:
    """An empty database (no app tables yet) just upgrades straight to head."""

    settings = _make_settings(tmp_path)
    engine = create_engine(settings.sqlalchemy_database_url, connect_args={"check_same_thread": False})

    _stamp_or_upgrade(target_engine=engine, target_settings=settings)

    table_names = set(inspect(engine).get_table_names())
    assert "alembic_version" in table_names
    assert "projects" in table_names
