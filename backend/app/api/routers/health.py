from fastapi import APIRouter, HTTPException, Query

from app.container import registry
from app.schemas import DeleteResponse, ModelInfo, ModelUpdate

router = APIRouter()


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
