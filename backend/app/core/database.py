from collections.abc import Generator
from datetime import datetime

from sqlalchemy import create_engine, inspect, text
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


def init_db() -> None:
    from app.db.models import (  # noqa: F401
        EvaluationJob,
        InferenceJob,
        InferenceRun,
        Project,
        TrainingJob,
    )

    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_schema()
    _ensure_default_project()


def _ensure_sqlite_schema() -> None:
    if not settings.sqlalchemy_database_url.startswith("sqlite"):
        return

    inspector = inspect(engine)
    table_columns = {
        table_name: {column["name"] for column in inspector.get_columns(table_name)}
        for table_name in inspector.get_table_names()
    }
    project_columns = {
        "inference_runs": "project_id VARCHAR(64)",
        "inference_jobs": "project_id VARCHAR(64)",
        "evaluation_jobs": "project_id VARCHAR(64)",
        "training_jobs": "project_id VARCHAR(64)",
    }
    extra_columns = {
        "evaluation_jobs": {"comparison_id": "comparison_id VARCHAR(64)"},
    }
    with engine.begin() as connection:
        for table_name, column_sql in project_columns.items():
            if table_name in table_columns and "project_id" not in table_columns[table_name]:
                connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}"))
        for table_name, columns in extra_columns.items():
            for column_name, column_sql in columns.items():
                if table_name in table_columns and column_name not in table_columns[table_name]:
                    connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}"))
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_evaluation_jobs_comparison_id "
                "ON evaluation_jobs (comparison_id)"
            )
        )


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
