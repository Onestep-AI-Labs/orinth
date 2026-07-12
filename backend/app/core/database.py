from collections.abc import Generator
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.defaults import DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME, DEFAULT_TASK_TYPE
from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
settings.database_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.sqlalchemy_database_url,
    connect_args={"check_same_thread": False}
    if settings.sqlalchemy_database_url.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# backend/ directory: parents[0]=core, [1]=app, [2]=backend
BACKEND_DIR = Path(__file__).resolve().parents[2]


def _alembic_config() -> AlembicConfig:
    alembic_cfg = AlembicConfig(str(BACKEND_DIR / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    alembic_cfg.set_main_option("sqlalchemy.url", settings.sqlalchemy_database_url)
    return alembic_cfg


def init_db() -> None:
    from app.db.models import (  # noqa: F401
        EvaluationJob,
        InferenceJob,
        InferenceRun,
        Project,
        TrainingJob,
    )

    _stamp_or_upgrade()
    _ensure_default_project()


def _stamp_or_upgrade() -> None:
    """Bring the database schema up to date via Alembic migrations.

    A database created by the pre-Alembic `create_all` + hand-patch path has
    tables but no `alembic_version` table. Its schema already matches the
    baseline migration, so it is stamped at the baseline revision (not
    literally re-run) before upgrading to head, in case newer migrations
    have been added since. A brand-new (empty) database, or one that already
    has an `alembic_version` table, just upgrades straight to head.
    """

    alembic_cfg = _alembic_config()
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    has_alembic_version = "alembic_version" in table_names
    has_app_tables = bool(table_names - {"alembic_version"})

    if has_app_tables and not has_alembic_version:
        script_dir = ScriptDirectory.from_config(alembic_cfg)
        baseline_revisions = script_dir.get_bases()
        baseline_revision = baseline_revisions[0] if baseline_revisions else "head"
        command.stamp(alembic_cfg, baseline_revision)

    command.upgrade(alembic_cfg, "head")


def _ensure_default_project() -> None:
    from app.db.models import Project

    db = SessionLocal()
    try:
        project = db.get(Project, DEFAULT_PROJECT_ID)
        if project is None:
            project = Project(
                id=DEFAULT_PROJECT_ID,
                name=DEFAULT_PROJECT_NAME,
                description="Default local AI research workspace.",
                task_types=[DEFAULT_TASK_TYPE, "classification", "object_detection"],
                metadata_json={"created_from": "system_default"},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(project)
            db.commit()
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
