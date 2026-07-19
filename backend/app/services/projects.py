from datetime import UTC, datetime
from typing import cast
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
from app.schemas import DeleteResponse, ProjectCreate, ProjectSummary, ProjectUpdate, TaskType

# `DEFAULT_TASK_TYPE`/`DEFAULT_PROJECT_TASK_TYPES` live in app.core.defaults as plain
# `str`/`list[str]` (shared with non-Pydantic code); cast once here where they need to
# satisfy the `TaskType` Literal used by the Pydantic schemas.
_DEFAULT_TASK_TYPE_LITERAL = cast(TaskType, DEFAULT_TASK_TYPE)


class ProjectService:
    def list_projects(self, db: Session) -> list[ProjectSummary]:
        self.ensure_default(db)
        projects = db.scalars(select(Project).order_by(Project.created_at.asc())).all()
        return [project_read(project) for project in projects]

    def create_project(self, db: Session, payload: ProjectCreate) -> ProjectSummary:
        task_types = payload.task_types or [_DEFAULT_TASK_TYPE_LITERAL]
        project = Project(
            id=self._new_project_id(payload.name),
            name=payload.name,
            description=payload.description,
            task_types=task_types,
            metadata_json=payload.metadata,
            created_at=datetime.now(UTC).replace(tzinfo=None),
            updated_at=datetime.now(UTC).replace(tzinfo=None),
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
            project.task_types = [str(task_type) for task_type in payload.task_types]
        if payload.metadata is not None:
            project.metadata_json = payload.metadata
        project.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        db.refresh(project)
        return project_read(project)

    def delete_project(
        self, db: Session, project_id: str, dataset_count: int = 0
    ) -> DeleteResponse:
        if project_id == DEFAULT_PROJECT_ID:
            raise HTTPException(status_code=409, detail="Default project cannot be deleted")
        project = db.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        blockers = self._describe_project_blockers(db, project_id, dataset_count)
        if blockers:
            raise HTTPException(
                status_code=409,
                detail=f"This project still has {_join_with_and(blockers)}. Delete those first.",
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
                project.updated_at = datetime.now(UTC).replace(tzinfo=None)
                db.commit()
                db.refresh(project)
            return project
        project = Project(
            id=DEFAULT_PROJECT_ID,
            name=DEFAULT_PROJECT_NAME,
            description="Default local AI research workspace.",
            task_types=DEFAULT_PROJECT_TASK_TYPES.copy(),
            metadata_json={"created_from": "system_default"},
            created_at=datetime.now(UTC).replace(tzinfo=None),
            updated_at=datetime.now(UTC).replace(tzinfo=None),
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        return project

    def _new_project_id(self, name: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
        slug = "-".join(part for part in slug.split("-") if part)[:44] or "project"
        return f"{slug}-{uuid4().hex[:8]}"

    def _describe_project_blockers(
        self, db: Session, project_id: str, dataset_count: int
    ) -> list[str]:
        blockers = []
        if dataset_count:
            blockers.append(_pluralize(dataset_count, "dataset"))
        job_labels = (
            (InferenceRun, "inference result"),
            (InferenceJob, "inference job"),
            (EvaluationJob, "testing job"),
            (TrainingJob, "training job"),
        )
        for model, label in job_labels:
            count = db.scalar(
                select(func.count()).select_from(model).where(model.project_id == project_id)
            )
            if count:
                blockers.append(_pluralize(count, label))
        return blockers


def _pluralize(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _join_with_and(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def project_read(project: Project) -> ProjectSummary:
    return ProjectSummary(
        id=project.id,
        name=project.name,
        description=project.description,
        # project.task_types is a `list[str]` DB column; values are always drawn from
        # the TaskType Literal at write time (see create_project/update_project above).
        task_types=cast("list[TaskType]", project.task_types) or [_DEFAULT_TASK_TYPE_LITERAL],
        metadata=project.metadata_json or {},
        created_at=project.created_at,
        updated_at=project.updated_at,
    )
