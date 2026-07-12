from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.serializers import training_job_read
from app.container import job_executor, training_service
from app.core.database import get_db
from app.db.models import TrainingJob
from app.schemas import (
    DeleteRequest,
    DeleteResponse,
    ModelAssetPrepareRequest,
    ModelAssetStatus,
    TrainingJobCreate,
    TrainingJobRead,
    TrainingModelOption,
)

router = APIRouter(prefix="/training")


@router.get("/model-options", response_model=list[TrainingModelOption])
def list_training_model_options(task_type: str | None = Query(default=None)) -> list[TrainingModelOption]:
    return training_service.model_options(task_type)


@router.post("/model-assets/prepare", response_model=ModelAssetStatus)
def prepare_training_model_asset(payload: ModelAssetPrepareRequest) -> ModelAssetStatus:
    return training_service.prepare_model_asset(payload)


@router.post("/jobs", response_model=TrainingJobRead)
def create_training_job(
    payload: TrainingJobCreate,
    db: Session = Depends(get_db),
) -> TrainingJobRead:
    try:
        job = training_service.create_job(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    job_executor.submit(training_service.run_job, job.id)
    return training_job_read(job)


@router.get("/jobs", response_model=list[TrainingJobRead])
def list_training_jobs(
    limit: int = Query(default=25, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[TrainingJobRead]:
    return [
        training_job_read(job)
        for job in training_service.list_jobs(db, limit=limit, offset=offset, project_id=project_id)
    ]


@router.delete("/jobs", response_model=DeleteResponse)
def clear_training_jobs(
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> DeleteResponse:
    return DeleteResponse.model_validate(training_service.delete_jobs(db, project_id=project_id))


@router.post("/jobs/delete", response_model=DeleteResponse)
def delete_training_jobs(payload: DeleteRequest, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(
        training_service.delete_jobs(
            db,
            ids=None if payload.clear_all else payload.ids,
            project_id=payload.project_id,
        )
    )


@router.get("/jobs/{job_id}", response_model=TrainingJobRead)
def get_training_job(job_id: str, db: Session = Depends(get_db)) -> TrainingJobRead:
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    return training_job_read(job)


@router.delete("/jobs/{job_id}", response_model=DeleteResponse)
def delete_training_job(job_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(training_service.delete_jobs(db, ids=[job_id]))


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobRead)
def cancel_training_job(job_id: str, db: Session = Depends(get_db)) -> TrainingJobRead:
    try:
        job = training_service.cancel(db, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Training job not found") from exc
    return training_job_read(job)


@router.post("/jobs/{job_id}/promote")
def promote_training_job(job_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        model_id = training_service.promote(db, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Training job not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"model_id": model_id}
