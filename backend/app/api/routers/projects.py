from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.container import dataset_service, project_service
from app.core.database import get_db
from app.schemas import DeleteResponse, ProjectCreate, ProjectSummary, ProjectUpdate

router = APIRouter(prefix="/projects")


@router.get("", response_model=list[ProjectSummary])
def list_projects(db: Session = Depends(get_db)) -> list[ProjectSummary]:
    return project_service.list_projects(db)


@router.post("", response_model=ProjectSummary)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> ProjectSummary:
    return project_service.create_project(db, payload)


@router.patch("/{project_id}", response_model=ProjectSummary)
def update_project(
    project_id: str, payload: ProjectUpdate, db: Session = Depends(get_db)
) -> ProjectSummary:
    return project_service.update_project(db, project_id, payload)


@router.delete("/{project_id}", response_model=DeleteResponse)
def delete_project(project_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    has_datasets = bool(dataset_service.list_datasets(project_id))
    return project_service.delete_project(db, project_id, has_datasets)
