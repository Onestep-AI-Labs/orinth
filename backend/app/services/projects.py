from datetime import datetime
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.defaults import (
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_NAME,
    DEFAULT_PROJECT_TASK_TYPES,
    DEFAULT_TASK_TYPE,
)
from app.db.models import EvaluationJob, InferenceJob, InferenceRun, Project, TrainingJob
from app.schemas import DeleteResponse, ProjectCreate, ProjectSummary, ProjectUpdate


class ProjectService:
    def list_projects(self, db: Session) -> list[ProjectSummary]:
        self.ensure_default(db)
        projects = db.scalars(select(Project).order_by(Project.created_at.asc())).all()
        return [project_read(project) for project in projects]

    def create_project(self, db: Session, payload: ProjectCreate) -> ProjectSummary:
        task_types = payload.task_types or [DEFAULT_TASK_TYPE]
        project = Project(
            id=self._new_project_id(payload.name),
            name=payload.name,
            description=payload.description,
            task_types=task_types,
            metadata_json=payload.metadata,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        return project_read(project)

    def update_project(self, db: Session, project_id: str, payload: ProjectUpdate) -> ProjectSummary:
        project = db.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if payload.name is not None:
            project.name = payload.name
        if payload.description is not None:
            project.description = payload.description
        if payload.task_types is not None:
            project.task_types = payload.task_types
        if payload.metadata is not None:
            project.metadata_json = payload.metadata
        project.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(project)
        return project_read(project)

    def delete_project(
        self, db: Session, project_id: str, has_datasets: bool = False
    ) -> DeleteResponse:
        if project_id == DEFAULT_PROJECT_ID:
            raise HTTPException(status_code=409, detail="Default project cannot be deleted")
        project = db.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if has_datasets or self._has_project_history(db, project_id):
            raise HTTPException(
                status_code=409,
                detail="Project still has datasets or history. Delete those first.",
            )
        db.delete(project)
        db.commit()
        return DeleteResponse(deleted=1)

    def ensure_default(self, db: Session) -> Project:
        project = db.get(Project, DEFAULT_PROJECT_ID)
        if project is not None:
            task_types = list(project.task_types or [])
            missing = [task for task in DEFAULT_PROJECT_TASK_TYPES if task not in task_types]
            if missing:
                project.task_types = [*task_types, *missing]
                project.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(project)
            return project
        project = Project(
            id=DEFAULT_PROJECT_ID,
            name=DEFAULT_PROJECT_NAME,
            description="Default local AI research workspace.",
            task_types=DEFAULT_PROJECT_TASK_TYPES.copy(),
            metadata_json={"created_from": "system_default"},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        return project

    def _new_project_id(self, name: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
        slug = "-".join(part for part in slug.split("-") if part)[:44] or "project"
        return f"{slug}-{uuid4().hex[:8]}"

    def _has_project_history(self, db: Session, project_id: str) -> bool:
        for model in (InferenceRun, InferenceJob, EvaluationJob, TrainingJob):
            count = db.scalar(
                select(func.count()).select_from(model).where(model.project_id == project_id)
            )
            if count:
                return True
        return False


def project_read(project: Project) -> ProjectSummary:
    return ProjectSummary(
        id=project.id,
        name=project.name,
        description=project.description,
        task_types=project.task_types or [DEFAULT_TASK_TYPE],
        metadata=project.metadata_json or {},
        created_at=project.created_at,
        updated_at=project.updated_at,
    )
