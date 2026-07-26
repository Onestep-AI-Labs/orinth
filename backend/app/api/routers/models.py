from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.container import export_service, model_upload_service, registry
from app.schemas import (
    DeleteResponse,
    ModelExportRequest,
    ModelExportStatus,
    ModelInfo,
    ModelUpdate,
    ModelUploadOption,
    ModelUploadResult,
)
from app.services.llm_export import ExportError
from app.services.model_upload import ModelUploadError

router = APIRouter()


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


@router.get("/models/upload-options", response_model=list[ModelUploadOption])
def model_upload_options() -> list[ModelUploadOption]:
    return model_upload_service.upload_options()


@router.post("/models/upload", response_model=ModelUploadResult)
async def upload_model(
    family: str = Form(...),
    name: str = Form(...),
    project_id: str | None = Form(default=None),
    task_type: str | None = Form(default=None),
    labels: str | None = Form(default=None),
    input_size: int | None = Form(default=None),
    base_model_id: str | None = Form(default=None),
    weights: UploadFile | None = File(default=None),
    model: UploadFile | None = File(default=None),
    unet: UploadFile | None = File(default=None),
    classifier: UploadFile | None = File(default=None),
    archive: UploadFile | None = File(default=None),
) -> ModelUploadResult:
    files = {
        key: value
        for key, value in {
            "weights": weights,
            "model": model,
            "unet": unet,
            "classifier": classifier,
            "archive": archive,
        }.items()
        if value is not None
    }
    try:
        return await model_upload_service.create_upload(
            family=family,
            name=name,
            files=files,
            project_id=project_id,
            task_type=task_type,
            labels=labels,
            input_size=input_size,
            base_model_id=base_model_id,
        )
    except ModelUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/models/{model_id}", response_model=ModelInfo)
def update_model(model_id: str, payload: ModelUpdate) -> ModelInfo:
    try:
        return registry.update_model(model_id, name=payload.name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Model not found or read-only") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/models/{model_id}", response_model=DeleteResponse)
def delete_model(model_id: str) -> DeleteResponse:
    try:
        registry.delete_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Model not found or read-only") from exc
    return DeleteResponse(deleted=1)


@router.get("/models/{model_id}/download")
def download_model(model_id: str) -> FileResponse:
    try:
        path, filename = registry.model_download(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Model not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FileResponse(path, filename=filename)


@router.post("/models/{model_id}/export", response_model=ModelExportStatus)
def create_model_export(model_id: str, payload: ModelExportRequest) -> ModelExportStatus:
    try:
        return export_service.create_export(model_id, payload.format)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Model not found") from exc
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models/{model_id}/exports", response_model=list[ModelExportStatus])
def list_model_exports(model_id: str) -> list[ModelExportStatus]:
    try:
        return export_service.list_exports(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Model not found") from exc


@router.get("/models/{model_id}/exports/{export_id}/download")
def download_model_export(model_id: str, export_id: str) -> FileResponse:
    try:
        path, filename = export_service.export_download(model_id, export_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Export not found") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FileResponse(path, filename=filename)
