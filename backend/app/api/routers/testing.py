from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.serializers import comparison_metric_summary, evaluation_job_read
from app.container import evaluation_service
from app.core.database import get_db
from app.core.defaults import DEFAULT_PROJECT_ID
from app.db.models import EvaluationJob
from app.schemas import (
    DeleteRequest,
    DeleteResponse,
    EvaluationComparisonRead,
    EvaluationDatasetInfo,
    EvaluationJobBatchCreate,
    EvaluationJobCreate,
    EvaluationJobRead,
    EvaluationPerImageRow,
)

router = APIRouter(prefix="/testing")


@router.get("/datasets", response_model=list[EvaluationDatasetInfo])
def list_testing_datasets(
    project_id: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
) -> list[EvaluationDatasetInfo]:
    return evaluation_service.list_datasets(project_id, task_type)


@router.post("/jobs", response_model=EvaluationJobRead)
def create_testing_job(
    payload: EvaluationJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> EvaluationJobRead:
    try:
        job = evaluation_service.create_job(db, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model or dataset: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(evaluation_service.run_job, job.id)
    return evaluation_job_read(job)


@router.post("/jobs/batch", response_model=list[EvaluationJobRead])
def create_testing_jobs_batch(
    payload: EvaluationJobBatchCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> list[EvaluationJobRead]:
    try:
        jobs = evaluation_service.create_jobs(db, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model or dataset: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    for job in jobs:
        background_tasks.add_task(evaluation_service.run_job, job.id)
    return [evaluation_job_read(job) for job in jobs]


@router.get("/jobs", response_model=list[EvaluationJobRead])
def list_testing_jobs(
    limit: int = 25,
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[EvaluationJobRead]:
    return [
        evaluation_job_read(job)
        for job in evaluation_service.list_jobs(db, limit=limit, project_id=project_id)
    ]


@router.delete("/jobs", response_model=DeleteResponse)
def clear_testing_jobs(
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> DeleteResponse:
    return DeleteResponse.model_validate(evaluation_service.delete_jobs(db, project_id=project_id))


@router.post("/jobs/delete", response_model=DeleteResponse)
def delete_testing_jobs(payload: DeleteRequest, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(
        evaluation_service.delete_jobs(
            db,
            ids=None if payload.clear_all else payload.ids,
            project_id=payload.project_id,
        )
    )


@router.get("/jobs/{job_id}", response_model=EvaluationJobRead)
def get_testing_job(job_id: str, db: Session = Depends(get_db)) -> EvaluationJobRead:
    job = db.get(EvaluationJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    return evaluation_job_read(job)


@router.get("/jobs/{job_id}/comparison", response_model=EvaluationComparisonRead)
def get_testing_comparison(job_id: str, db: Session = Depends(get_db)) -> EvaluationComparisonRead:
    try:
        jobs = evaluation_service.comparison_jobs(db, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Evaluation job not found") from exc
    first = jobs[0]
    comparison_id = first.comparison_id or first.id
    return EvaluationComparisonRead(
        comparison_id=comparison_id,
        project_id=first.project_id or DEFAULT_PROJECT_ID,
        dataset_key=first.dataset_key,
        jobs=[evaluation_job_read(job) for job in jobs],
        metrics=[comparison_metric_summary(job) for job in jobs],
    )


@router.delete("/jobs/{job_id}", response_model=DeleteResponse)
def delete_testing_job(job_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(evaluation_service.delete_jobs(db, ids=[job_id]))


@router.get("/jobs/{job_id}/per-image", response_model=list[EvaluationPerImageRow])
def get_testing_per_image(job_id: str, db: Session = Depends(get_db)) -> list[EvaluationPerImageRow]:
    try:
        return evaluation_service.per_image_rows(db, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Evaluation job not found") from exc
