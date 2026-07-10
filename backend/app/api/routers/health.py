from fastapi import APIRouter, Query

from app.container import registry
from app.schemas import ModelInfo

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
