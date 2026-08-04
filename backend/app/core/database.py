from collections.abc import Generator
from datetime import datetime
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.defaults import DEFAULT_PROJECT_ID, DEFAULT_PROJECT_NAME, DEFAULT_TASK_TYPE
from app.core.config import Settings, get_settings


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


def _alembic_config(target_settings: Settings) -> AlembicConfig:
    alembic_cfg = AlembicConfig(str(BACKEND_DIR / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    alembic_cfg.set_main_option("sqlalchemy.url", target_settings.sqlalchemy_database_url)
    # migrations/env.py reads this attribute (falling back to get_settings()
    # when absent) so a caller-provided settings object actually takes
    # effect instead of being silently overridden by the global singleton.
    alembic_cfg.attributes["configure_url"] = target_settings.sqlalchemy_database_url
    return alembic_cfg


def init_db() -> None:
    from app.db.models import (  # noqa: F401
        Architecture,
        EvaluationJob,
        InferenceJob,
        InferenceRun,
        Project,
        TrainingJob,
    )

    _stamp_or_upgrade()
    _ensure_default_project()


def _stamp_or_upgrade(
    target_engine: Engine = engine, target_settings: Settings = settings
) -> None:
    """Bring the database schema up to date via Alembic migrations.

    A database created by the pre-Alembic `create_all` + hand-patch path has
    tables but no `alembic_version` table. If its schema actually matches the
    current ORM metadata, it is stamped (not literally re-run) and then
    upgraded, which is a no-op. A brand-new (empty) database, or one that
    already has an `alembic_version` table, just upgrades straight to head.

    Stamping is only safe when the live schema truly matches — an older
    pre-Alembic database may predate columns or tables the ORM expects (e.g. a
    `project_id` column, or the `projects` table). Stamping such a database
    would mark it "up to date" while `command.upgrade` then no-ops, silently
    leaving it missing objects the app assumes exist. Guard against that by
    diffing the live schema against the ORM metadata first and refusing to
    stamp on a mismatch, so the failure is a loud, actionable error at startup
    instead of a `sqlite3.OperationalError` on first use.

    The stamp goes at `head`, not at the baseline revision. An empty diff
    means the live schema equals the current ORM metadata, which is head by
    definition — stamping at baseline would then replay every migration since
    baseline against tables that already exist. That was invisible while
    baseline *was* head; it stopped being true with the first migration added
    after the baseline.
    """

    alembic_cfg = _alembic_config(target_settings)
    inspector = inspect(target_engine)
    table_names = set(inspector.get_table_names())
    has_alembic_version = "alembic_version" in table_names
    has_app_tables = bool(table_names - {"alembic_version"})

    if has_app_tables and not has_alembic_version:
        with target_engine.connect() as connection:
            diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)
        if diff:
            raise RuntimeError(
                "Found an existing database with application tables but no "
                "Alembic version tracking, and its schema does not match the "
                "expected schema (differences: "
                f"{diff!r}). This looks like a database created by an older "
                "schema version. Back it up, then either migrate it by hand "
                "to match the migrations in backend/migrations/versions/, or "
                "delete it so init_db() can recreate it from scratch."
            )

        command.stamp(alembic_cfg, "head")

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
