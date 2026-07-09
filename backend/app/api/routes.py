from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry
from app.schemas import (
    EvaluationDatasetInfo,
    EvaluationJobCreate,
    EvaluationJobRead,
    InferenceParameters,
    InferenceResult,
    ModelInfo,
    TrainingJobCreate,
    TrainingJobRead,
)
from app.services.evaluation import EvaluationService
from app.services.inference import InferenceService
from app.services.training import TrainingService

router = APIRouter()

settings = get_settings()
storage = Storage(settings)
registry = ModelRegistry(settings, storage)
inference_service = InferenceService(storage, registry)
evaluation_service = EvaluationService(settings, storage, registry)
training_service = TrainingService(settings, storage, registry)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return registry.list_models()


@router.post("/inference", response_model=InferenceResult)
async def create_inference(
    model_id: str = Form(...),
    confidence_threshold: float = Form(0.65),
    iou_threshold: float = Form(0.7),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> InferenceResult:
    parameters = InferenceParameters(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    return await inference_service.run(db, file, model_id, parameters)


@router.get("/inference", response_model=list[InferenceResult])
def list_inference(limit: int = 25, db: Session = Depends(get_db)) -> list[InferenceResult]:
    return inference_service.list(db, limit=limit)


@router.get("/inference/{inference_id}", response_model=InferenceResult)
def get_inference(inference_id: str, db: Session = Depends(get_db)) -> InferenceResult:
    return inference_service.get(db, inference_id)


@router.get("/testing/datasets", response_model=list[EvaluationDatasetInfo])
def list_testing_datasets() -> list[EvaluationDatasetInfo]:
    return evaluation_service.list_datasets()


@router.post("/testing/jobs", response_model=EvaluationJobRead)
def create_testing_job(
    payload: EvaluationJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> EvaluationJobRead:
    try:
        job = evaluation_service.create_job(db, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model or dataset: {exc}") from exc
    background_tasks.add_task(evaluation_service.run_job, job.id)
    return evaluation_job_read(job)


@router.get("/testing/jobs", response_model=list[EvaluationJobRead])
def list_testing_jobs(limit: int = 25, db: Session = Depends(get_db)) -> list[EvaluationJobRead]:
    return [evaluation_job_read(job) for job in evaluation_service.list_jobs(db, limit=limit)]


@router.get("/testing/jobs/{job_id}", response_model=EvaluationJobRead)
def get_testing_job(job_id: str, db: Session = Depends(get_db)) -> EvaluationJobRead:
    job = db.get(evaluation_service_job_model(), job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    return evaluation_job_read(job)


@router.post("/training/jobs", response_model=TrainingJobRead)
def create_training_job(
    payload: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TrainingJobRead:
    job = training_service.create_job(db, payload)
    background_tasks.add_task(training_service.run_job, job.id)
    return training_job_read(job)


@router.get("/training/jobs", response_model=list[TrainingJobRead])
def list_training_jobs(limit: int = 25, db: Session = Depends(get_db)) -> list[TrainingJobRead]:
    return [training_job_read(job) for job in training_service.list_jobs(db, limit=limit)]


@router.get("/training/jobs/{job_id}", response_model=TrainingJobRead)
def get_training_job(job_id: str, db: Session = Depends(get_db)) -> TrainingJobRead:
    job = db.get(training_service_job_model(), job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    return training_job_read(job)


@router.post("/training/jobs/{job_id}/promote")
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


def evaluation_job_read(job) -> EvaluationJobRead:
    return EvaluationJobRead(
        id=job.id,
        model_id=job.model_id,
        dataset_key=job.dataset_key,
        status=job.status,
        limit=job.limit,
        metrics=job.metrics or {},
        artifacts=job.artifacts or {},
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def training_job_read(job) -> TrainingJobRead:
    return TrainingJobRead(
        id=job.id,
        model_family=job.model_family,
        status=job.status,
        parameters=job.parameters or {},
        artifacts=job.artifacts or {},
        promoted_model_id=job.promoted_model_id,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def evaluation_service_job_model():
    from app.db.models import EvaluationJob

    return EvaluationJob


def training_service_job_model():
    from app.db.models import TrainingJob

    return TrainingJob
