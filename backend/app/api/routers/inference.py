from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from app.container import inference_executor, inference_service
from app.core.database import get_db
from app.core.defaults import DEFAULT_PROJECT_ID
from app.schemas import (
    DeleteRequest,
    DeleteResponse,
    InferenceJobRead,
    InferenceParameters,
    InferenceResult,
)
from app.services.job_runner import submit_job

router = APIRouter(prefix="/inference")


@router.post("", response_model=InferenceResult)
async def create_inference(
    model_id: str = Form(...),
    project_id: str = Form(DEFAULT_PROJECT_ID),
    confidence_threshold: float = Form(0.65),
    iou_threshold: float = Form(0.7),
    question: str | None = Form(default=None),
    max_length: int = Form(120),
    text_content: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
) -> InferenceResult:
    parameters = InferenceParameters(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        question=question,
        max_length=max_length,
    )
    return await inference_service.run(db, file, model_id, parameters, project_id, text_content)


@router.post("/jobs", response_model=InferenceJobRead)
async def create_inference_job(
    model_id: str = Form(...),
    project_id: str = Form(DEFAULT_PROJECT_ID),
    confidence_threshold: float = Form(0.65),
    iou_threshold: float = Form(0.7),
    question: str | None = Form(default=None),
    max_length: int = Form(120),
    text_content: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
) -> InferenceJobRead:
    parameters = InferenceParameters(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        question=question,
        max_length=max_length,
    )
    job = await inference_service.create_job(db, file, model_id, parameters, project_id, text_content)
    submit_job(inference_executor, inference_service.run_job, job.id)
    return inference_service.get_job(db, job.id)


@router.get("", response_model=list[InferenceResult])
def list_inference(
    limit: int = Query(default=25, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[InferenceResult]:
    return inference_service.list(db, limit=limit, offset=offset, project_id=project_id)


@router.delete("", response_model=DeleteResponse)
def clear_inference(
    project_id: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> DeleteResponse:
    return DeleteResponse.model_validate(inference_service.delete_runs(db, project_id=project_id))


@router.post("/delete", response_model=DeleteResponse)
def delete_inference_many(payload: DeleteRequest, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(
        inference_service.delete_runs(
            db,
            ids=None if payload.clear_all else payload.ids,
            project_id=payload.project_id,
        )
    )


@router.get("/{inference_id}", response_model=InferenceResult)
def get_inference(inference_id: str, db: Session = Depends(get_db)) -> InferenceResult:
    return inference_service.get(db, inference_id)


@router.delete("/{inference_id}", response_model=DeleteResponse)
def delete_inference(inference_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    return DeleteResponse.model_validate(inference_service.delete_runs(db, ids=[inference_id]))


@router.get("/jobs/{job_id}", response_model=InferenceJobRead)
def get_inference_job(job_id: str, db: Session = Depends(get_db)) -> InferenceJobRead:
    return inference_service.get_job(db, job_id)
