from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest
from fastapi import HTTPException

from app.core.defaults import DEFAULT_PROJECT_ID, DEFAULT_PROJECT_TASK_TYPES
from app.core.database import Base
from app.db.models import Project, TrainingJob
from app.schemas import ProjectCreate
from app.services.projects import ProjectService


def test_project_service_creates_default_and_user_project(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'projects.db'}")
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)
    db = session_local()
    service = ProjectService()

    try:
        projects = service.list_projects(db)
        assert projects[0].id == DEFAULT_PROJECT_ID
        assert projects[0].task_types == DEFAULT_PROJECT_TASK_TYPES

        created = service.create_project(
            db,
            ProjectCreate(
                name="General Images",
                task_types=["classification", "object_detection", "segmentation"],
            ),
        )

        assert created.name == "General Images"
        assert db.get(Project, created.id) is not None

        deleted = service.delete_project(db, created.id)

        assert deleted.deleted == 1
        assert db.get(Project, created.id) is None

        with pytest.raises(HTTPException) as exc:
            service.delete_project(db, DEFAULT_PROJECT_ID)
        assert exc.value.status_code == 409
    finally:
        db.close()


def test_delete_project_reports_specific_blockers(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'projects-blockers.db'}")
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)
    db = session_local()
    service = ProjectService()

    try:
        created = service.create_project(db, ProjectCreate(name="Blocked Project"))
        db.add(TrainingJob(id="job-1", project_id=created.id, model_family="yolo"))
        db.commit()

        with pytest.raises(HTTPException) as exc:
            service.delete_project(db, created.id, dataset_count=2)
        assert exc.value.status_code == 409
        assert exc.value.detail == (
            "This project still has 2 datasets and 1 training job. Delete those first."
        )
    finally:
        db.close()


def test_project_service_backfills_default_nlp_tasks(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'projects-backfill.db'}")
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(bind=engine)
    db = session_local()
    service = ProjectService()
    try:
        db.add(
            Project(
                id=DEFAULT_PROJECT_ID,
                name="Legacy Default",
                task_types=["segmentation", "classification"],
                metadata_json={},
            )
        )
        db.commit()

        default = service.ensure_default(db)

        assert default.task_types == [
            "segmentation",
            "classification",
            "object_detection",
            "text_classification",
            "summarization",
            "question_answering",
        ]
    finally:
        db.close()
