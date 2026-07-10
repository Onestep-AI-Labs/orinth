from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.defaults import DEFAULT_PROJECT_ID, DEFAULT_TASK_TYPE
from app.core.database import get_db
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry
from app.schemas import (
    EvaluationDatasetInfo,
    EvaluationComparisonRead,
    EvaluationJobBatchCreate,
    EvaluationJobCreate,
    EvaluationJobRead,
    EvaluationPerImageRow,
    DeleteRequest,
    DeleteResponse,
    DatasetAnnotationSave,
    DatasetCloneRequest,
    DatasetCreate,
    DatasetEdaSummary,
    DatasetImportRequest,
    DatasetItemBatchUploadResponse,
    DatasetItemBulkLabelUpdate,
    DatasetItemBulkLabelUpdateResponse,
    DatasetItemDeleteRequest,
    DatasetItemDetail,
    DatasetItemLabelUpdate,
    DatasetItemMoveRequest,
    DatasetItemMoveResponse,
    DatasetItemSummary,
    DatasetProcessRequest,
    DatasetProcessResponse,
    DatasetLabelCreate,
    DatasetLabelUpdate,
    DatasetPreprocessPreview,
    DatasetPreprocessPreviewRequest,
    DatasetSummary,
    DatasetUpdate,
    DatasetVersionCreate,
    DatasetVersionSummary,
    InferenceJobRead,
    InferenceParameters,
    InferenceResult,
    ModelAssetPrepareRequest,
    ModelAssetStatus,
    ModelInfo,
    ProjectCreate,
    ProjectSummary,
    ProjectUpdate,
    TrainingJobCreate,
    TrainingJobRead,
    TrainingModelOption,
)
from app.services.datasets import DatasetService
from app.services.evaluation import EvaluationService
from app.services.inference import InferenceService
from app.services.job_progress import progress_from_artifacts
from app.services.projects import ProjectService
from app.services.training import TrainingService

router = APIRouter()

settings = get_settings()
storage = Storage(settings)
registry = ModelRegistry(settings, storage)
project_service = ProjectService()
dataset_service = DatasetService(settings, storage)
inference_service = InferenceService(storage, registry)
evaluation_service = EvaluationService(settings, storage, registry, dataset_service)
training_service = TrainingService(settings, storage, registry, dataset_service)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/models", response_model=list[ModelInfo])
def list_models(
    available_only: bool = Query(default=False),
    project_id: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
) -> list[ModelInfo]:
    models = registry.list_models(project_id=project_id, task_type=task_type)
    if available_only:
        models = [model for model in models if model.available]
    return models


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(db: Session = Depends(get_db)) -> list[ProjectSummary]:
    return project_service.list_projects(db)


@router.post("/projects", response_model=ProjectSummary)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> ProjectSummary:
    return project_service.create_project(db, payload)


@router.patch("/projects/{project_id}", response_model=ProjectSummary)
def update_project(
    project_id: str, payload: ProjectUpdate, db: Session = Depends(get_db)
) -> ProjectSummary:
    return project_service.update_project(db, project_id, payload)


@router.delete("/projects/{project_id}", response_model=DeleteResponse)
def delete_project(project_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    has_datasets = bool(dataset_service.list_datasets(project_id))
    return project_service.delete_project(db, project_id, has_datasets)


@router.get("/datasets", response_model=list[DatasetSummary])
def list_datasets(project_id: str | None = Query(default=None)) -> list[DatasetSummary]:
    return dataset_service.list_datasets(project_id)


@router.post("/datasets", response_model=DatasetSummary)
def create_dataset(payload: DatasetCreate) -> DatasetSummary:
    return dataset_service.create_dataset(payload)


@router.patch("/datasets/{dataset_id}", response_model=DatasetSummary)
def update_dataset(dataset_id: str, payload: DatasetUpdate) -> DatasetSummary:
    return dataset_service.update_dataset(dataset_id, payload)


@router.post("/datasets/{dataset_id}/process", response_model=DatasetProcessResponse)
def process_dataset(dataset_id: str, payload: DatasetProcessRequest) -> DatasetProcessResponse:
    return dataset_service.process_dataset(dataset_id, payload)


@router.post("/datasets/import", response_model=DatasetSummary)
def import_dataset(payload: DatasetImportRequest) -> DatasetSummary:
    return dataset_service.import_dataset(payload)


@router.post("/datasets/{dataset_id}/clone", response_model=DatasetSummary)
def clone_dataset(dataset_id: str, payload: DatasetCloneRequest) -> DatasetSummary:
    return dataset_service.clone_dataset(dataset_id, payload.name, payload.project_id)


@router.delete("/datasets/{dataset_id}", response_model=DeleteResponse)
def delete_dataset(dataset_id: str) -> DeleteResponse:
    dataset_service.delete_dataset(dataset_id)
    return DeleteResponse(deleted=1)


@router.post("/datasets/{dataset_id}/labels", response_model=DatasetSummary)
def add_dataset_label(dataset_id: str, payload: DatasetLabelCreate) -> DatasetSummary:
    return dataset_service.add_label(dataset_id, payload.name)


@router.patch("/datasets/{dataset_id}/labels/{label_index}", response_model=DatasetSummary)
def rename_dataset_label(
    dataset_id: str, label_index: int, payload: DatasetLabelUpdate
) -> DatasetSummary:
    return dataset_service.rename_label(dataset_id, label_index, payload.name)


@router.delete("/datasets/{dataset_id}/labels/{label_index}", response_model=DatasetSummary)
def delete_dataset_label(
    dataset_id: str,
    label_index: int,
    force: bool = Query(default=False),
) -> DatasetSummary:
    return dataset_service.delete_label(dataset_id, label_index, force)


@router.get("/datasets/{dataset_id}/items", response_model=list[DatasetItemSummary])
def list_dataset_items(
    dataset_id: str,
    split: str = Query("train"),
    class_name: str | None = Query(default=None),
    unlabeled: bool = Query(default=False),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> list[DatasetItemSummary]:
    return dataset_service.list_items(dataset_id, split, class_name, unlabeled, limit, offset)


@router.post("/datasets/{dataset_id}/items", response_model=DatasetItemDetail)
async def upload_dataset_item(
    dataset_id: str,
    split: str = Form("unassigned"),
    class_id: int | None = Form(default=None),
    class_name: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> DatasetItemDetail:
    return await dataset_service.upload_image(dataset_id, split, file, class_id, class_name)


@router.post("/datasets/{dataset_id}/items/batch", response_model=DatasetItemBatchUploadResponse)
async def upload_dataset_items_batch(
    dataset_id: str,
    split: str = Form("unassigned"),
    class_id: int | None = Form(default=None),
    class_name: str | None = Form(default=None),
    files: list[UploadFile] = File(...),
) -> DatasetItemBatchUploadResponse:
    return await dataset_service.upload_images(dataset_id, split, files, class_id, class_name)


@router.post("/datasets/{dataset_id}/items/delete", response_model=DeleteResponse)
def delete_dataset_items(dataset_id: str, payload: DatasetItemDeleteRequest) -> DeleteResponse:
    return dataset_service.delete_items(dataset_id, payload)


@router.post("/datasets/{dataset_id}/items/move", response_model=DatasetItemMoveResponse)
def move_dataset_items(
    dataset_id: str,
    payload: DatasetItemMoveRequest,
) -> DatasetItemMoveResponse:
    return dataset_service.move_items(dataset_id, payload)


@router.patch(
    "/datasets/{dataset_id}/items/{split}/labels",
    response_model=DatasetItemBulkLabelUpdateResponse,
)
def update_dataset_item_labels(
    dataset_id: str,
    split: str,
    payload: DatasetItemBulkLabelUpdate,
) -> DatasetItemBulkLabelUpdateResponse:
    return dataset_service.bulk_set_item_labels(dataset_id, split, payload)


@router.get("/datasets/{dataset_id}/items/{split}/{item_id}", response_model=DatasetItemDetail)
def get_dataset_item(dataset_id: str, split: str, item_id: str) -> DatasetItemDetail:
    return dataset_service.item_detail(dataset_id, split, item_id)


@router.get("/datasets/{dataset_id}/items/{split}/{item_id}/image")
def get_dataset_image(dataset_id: str, split: str, item_id: str) -> FileResponse:
    return FileResponse(dataset_service.image_path(dataset_id, split, item_id))


@router.put(
    "/datasets/{dataset_id}/items/{split}/{item_id}/annotations",
    response_model=DatasetItemDetail,
)
def save_dataset_annotations(
    dataset_id: str,
    split: str,
    item_id: str,
    payload: DatasetAnnotationSave,
) -> DatasetItemDetail:
    return dataset_service.save_annotations(dataset_id, split, item_id, payload)


@router.patch(
    "/datasets/{dataset_id}/items/{split}/{item_id}/label",
    response_model=DatasetItemDetail,
)
def update_dataset_item_label(
    dataset_id: str,
    split: str,
    item_id: str,
    payload: DatasetItemLabelUpdate,
) -> DatasetItemDetail:
    return dataset_service.set_item_label(dataset_id, split, item_id, payload)


@router.post(
    "/datasets/{dataset_id}/items/{split}/{item_id}/preprocess-preview",
    response_model=DatasetPreprocessPreview,
)
def preview_dataset_preprocess(
    dataset_id: str,
    split: str,
    item_id: str,
    payload: DatasetPreprocessPreviewRequest,
) -> DatasetPreprocessPreview:
    return dataset_service.preprocess_preview(dataset_id, split, item_id, payload.config)


@router.get("/datasets/{dataset_id}/versions", response_model=list[DatasetVersionSummary])
def list_dataset_versions(dataset_id: str) -> list[DatasetVersionSummary]:
    return dataset_service.list_versions(dataset_id)


@router.post("/datasets/{dataset_id}/versions", response_model=DatasetVersionSummary)
def create_dataset_version(
    dataset_id: str,
    payload: DatasetVersionCreate,
) -> DatasetVersionSummary:
    return dataset_service.create_version(dataset_id, payload)


@router.get("/datasets/{dataset_id}/eda", response_model=DatasetEdaSummary)
def dataset_eda(dataset_id: str, split: str = Query("train")) -> DatasetEdaSummary:
    return dataset_service.eda_summary(dataset_id, split)


@router.post("/inference", response_model=InferenceResult)
async def create_inference(
    model_id: str = Form(...),
    project_id: str = Form(DEFAULT_PROJECT_ID),
    confidence_threshold: float = Form(0.65),
    iou_threshold: float = Form(0.7),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> InferenceResult:
    parameters = InferenceParameters(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    return await inference_service.run(db, file, model_id, parameters, project_id)


@router.post("/inference/jobs", response_model=InferenceJobRead)
async def create_inference_job(
    background_tasks: BackgroundTasks,
    model_id: str = Form(...),
    project_id: str = Form(DEFAULT_PROJECT_ID),
    confidence_threshold: float = Form(0.65),
    iou_threshold: float = Form(0.7),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> InferenceJobRead:
    parameters = InferenceParameters(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    job = await inference_service.create_job(db, file, model_id, parameters, project_id)
    background_tasks.add_task(inference_service.run_job, job.id)
    return inference_service.get_job(db, job.id)


@router.get("/inference", response_model=list[InferenceResult])
def list_inference(
    limit: int = 25,
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[InferenceResult]:
    return inference_service.list(db, limit=limit, project_id=project_id)


@router.delete("/inference", response_model=DeleteResponse)
def clear_inference(
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> DeleteResponse:
    return DeleteResponse.model_validate(inference_service.delete_runs(db, project_id=project_id))


@router.post("/inference/delete", response_model=DeleteResponse)
def delete_inference_many(payload: DeleteRequest, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(
        inference_service.delete_runs(
            db,
            ids=None if payload.clear_all else payload.ids,
            project_id=payload.project_id,
        )
    )


@router.get("/inference/{inference_id}", response_model=InferenceResult)
def get_inference(inference_id: str, db: Session = Depends(get_db)) -> InferenceResult:
    return inference_service.get(db, inference_id)


@router.delete("/inference/{inference_id}", response_model=DeleteResponse)
def delete_inference(inference_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(inference_service.delete_runs(db, ids=[inference_id]))


@router.get("/inference/jobs/{job_id}", response_model=InferenceJobRead)
def get_inference_job(job_id: str, db: Session = Depends(get_db)) -> InferenceJobRead:
    return inference_service.get_job(db, job_id)


@router.get("/testing/datasets", response_model=list[EvaluationDatasetInfo])
def list_testing_datasets(
    project_id: str | None = Query(default=None),
) -> list[EvaluationDatasetInfo]:
    return evaluation_service.list_datasets(project_id)


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
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(evaluation_service.run_job, job.id)
    return evaluation_job_read(job)


@router.post("/testing/jobs/batch", response_model=list[EvaluationJobRead])
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


@router.get("/testing/jobs", response_model=list[EvaluationJobRead])
def list_testing_jobs(
    limit: int = 25,
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[EvaluationJobRead]:
    return [
        evaluation_job_read(job)
        for job in evaluation_service.list_jobs(db, limit=limit, project_id=project_id)
    ]


@router.delete("/testing/jobs", response_model=DeleteResponse)
def clear_testing_jobs(
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> DeleteResponse:
    return DeleteResponse.model_validate(evaluation_service.delete_jobs(db, project_id=project_id))


@router.post("/testing/jobs/delete", response_model=DeleteResponse)
def delete_testing_jobs(payload: DeleteRequest, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(
        evaluation_service.delete_jobs(
            db,
            ids=None if payload.clear_all else payload.ids,
            project_id=payload.project_id,
        )
    )


@router.get("/testing/jobs/{job_id}", response_model=EvaluationJobRead)
def get_testing_job(job_id: str, db: Session = Depends(get_db)) -> EvaluationJobRead:
    job = db.get(evaluation_service_job_model(), job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    return evaluation_job_read(job)


@router.get("/testing/jobs/{job_id}/comparison", response_model=EvaluationComparisonRead)
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


@router.delete("/testing/jobs/{job_id}", response_model=DeleteResponse)
def delete_testing_job(job_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(evaluation_service.delete_jobs(db, ids=[job_id]))


@router.get("/testing/jobs/{job_id}/per-image", response_model=list[EvaluationPerImageRow])
def get_testing_per_image(job_id: str, db: Session = Depends(get_db)) -> list[EvaluationPerImageRow]:
    try:
        return evaluation_service.per_image_rows(db, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Evaluation job not found") from exc


@router.get("/training/model-options", response_model=list[TrainingModelOption])
def list_training_model_options(task_type: str | None = Query(default=None)) -> list[TrainingModelOption]:
    return training_service.model_options(task_type)


@router.post("/training/model-assets/prepare", response_model=ModelAssetStatus)
def prepare_training_model_asset(payload: ModelAssetPrepareRequest) -> ModelAssetStatus:
    return training_service.prepare_model_asset(payload)


@router.post("/training/jobs", response_model=TrainingJobRead)
def create_training_job(
    payload: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TrainingJobRead:
    try:
        job = training_service.create_job(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(training_service.run_job, job.id)
    return training_job_read(job)


@router.get("/training/jobs", response_model=list[TrainingJobRead])
def list_training_jobs(
    limit: int = 25,
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[TrainingJobRead]:
    return [
        training_job_read(job)
        for job in training_service.list_jobs(db, limit=limit, project_id=project_id)
    ]


@router.delete("/training/jobs", response_model=DeleteResponse)
def clear_training_jobs(
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> DeleteResponse:
    return DeleteResponse.model_validate(training_service.delete_jobs(db, project_id=project_id))


@router.post("/training/jobs/delete", response_model=DeleteResponse)
def delete_training_jobs(payload: DeleteRequest, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(
        training_service.delete_jobs(
            db,
            ids=None if payload.clear_all else payload.ids,
            project_id=payload.project_id,
        )
    )


@router.get("/training/jobs/{job_id}", response_model=TrainingJobRead)
def get_training_job(job_id: str, db: Session = Depends(get_db)) -> TrainingJobRead:
    job = db.get(training_service_job_model(), job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    return training_job_read(job)


@router.delete("/training/jobs/{job_id}", response_model=DeleteResponse)
def delete_training_job(job_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(training_service.delete_jobs(db, ids=[job_id]))


@router.post("/training/jobs/{job_id}/cancel", response_model=TrainingJobRead)
def cancel_training_job(job_id: str, db: Session = Depends(get_db)) -> TrainingJobRead:
    try:
        job = training_service.cancel(db, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Training job not found") from exc
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
        project_id=job.project_id or DEFAULT_PROJECT_ID,
        comparison_id=job.comparison_id or job.id,
        model_id=job.model_id,
        dataset_key=job.dataset_key,
        status=job.status,
        limit=job.limit,
        metrics=job.metrics or {},
        artifacts=job.artifacts or {},
        progress=progress_from_artifacts(job.artifacts),
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def comparison_metric_summary(job) -> dict:
    metrics = job.metrics or {}
    return {
        "job_id": job.id,
        "model_id": job.model_id,
        "status": job.status,
        "samples": metrics.get("samples"),
        "accuracy": ((metrics.get("image") or {}).get("overall") or {}).get("accuracy"),
        "macro_f1": ((metrics.get("image") or {}).get("overall") or {}).get("macro_f1"),
        "pixel_dice": (metrics.get("pixel") or {}).get("dice"),
        "object_recall": (metrics.get("object") or {}).get("recall"),
    }


def training_job_read(job) -> TrainingJobRead:
    return TrainingJobRead(
        id=job.id,
        project_id=job.project_id or DEFAULT_PROJECT_ID,
        task_type=(job.parameters or {}).get("task_type", DEFAULT_TASK_TYPE),
        model_family=job.model_family,
        status=job.status,
        parameters=job.parameters or {},
        artifacts=job.artifacts or {},
        artifact_urls=(job.artifacts or {}).get("artifact_urls", {}),
        progress=progress_from_artifacts(job.artifacts),
        metrics=(job.artifacts or {}).get("metrics", {}),
        history=(job.artifacts or {}).get("history", []),
        curves=(job.artifacts or {}).get("curves", {}),
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
