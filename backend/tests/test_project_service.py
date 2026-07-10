from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest
from fastapi import HTTPException

from app.core.defaults import DEFAULT_PROJECT_ID
from app.core.database import Base
from app.db.models import Project
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
