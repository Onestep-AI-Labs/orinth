from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.container import recipe_service
from app.core.database import get_db
from app.db.models import Project
from app.schemas import (
    DeleteResponse,
    OpenRouterModelsResponse,
    RecipeCommitRequest,
    RecipeCommitResponse,
    RecipeCreate,
    RecipeGenerateRequest,
    RecipeRead,
    RecipeRecord,
    RecipeRecordCreate,
    RecipeRecordDeleteRequest,
    RecipeRecordPage,
    RecipeRecordUpdate,
)

router = APIRouter(prefix="/recipes")


def _require_llm_project(db: Session, project_id: str) -> None:
    """Block commit into a project that has not declared `llm_finetune`.

    Same rule the dataset create/import flows enforce (phase 10); the backend is
    the authority so a direct API call cannot bypass it.
    """
    project = db.get(Project, project_id)
    if project is None:
        return
    declared = list(project.task_types or [])
    if declared and "llm_finetune" not in declared:
        raise HTTPException(
            status_code=409,
            detail="Project does not allow 'llm_finetune' datasets. Add the task in project settings first.",
        )


@router.get("/openrouter/models", response_model=OpenRouterModelsResponse)
def list_openrouter_models() -> OpenRouterModelsResponse:
    return recipe_service.openrouter_models()


@router.post("", response_model=RecipeRead)
def create_recipe(payload: RecipeCreate) -> RecipeRead:
    return recipe_service.create_recipe(payload)


@router.get("", response_model=list[RecipeRead])
def list_recipes(project_id: str | None = Query(default=None)) -> list[RecipeRead]:
    return recipe_service.list_recipes(project_id)


@router.get("/{recipe_id}", response_model=RecipeRead)
def get_recipe(recipe_id: str) -> RecipeRead:
    return recipe_service.get_recipe(recipe_id)


@router.delete("/{recipe_id}", response_model=DeleteResponse)
def delete_recipe(recipe_id: str) -> DeleteResponse:
    recipe_service.delete_recipe(recipe_id)
    return DeleteResponse(deleted=1)


@router.post("/{recipe_id}/sources", response_model=RecipeRead)
async def add_recipe_sources(
    recipe_id: str, files: list[UploadFile] = File(...)
) -> RecipeRead:
    return await recipe_service.add_sources(recipe_id, files)


@router.delete("/{recipe_id}/sources/{source_id}", response_model=RecipeRead)
def delete_recipe_source(recipe_id: str, source_id: str) -> RecipeRead:
    return recipe_service.delete_source(recipe_id, source_id)


@router.post("/{recipe_id}/generate", response_model=RecipeRead)
def generate_recipe(recipe_id: str, payload: RecipeGenerateRequest) -> RecipeRead:
    return recipe_service.generate(recipe_id, payload)


@router.post("/{recipe_id}/cancel", response_model=RecipeRead)
def cancel_recipe(recipe_id: str) -> RecipeRead:
    return recipe_service.cancel(recipe_id)


@router.get("/{recipe_id}/records", response_model=RecipeRecordPage)
def list_recipe_records(
    recipe_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> RecipeRecordPage:
    return recipe_service.list_records(recipe_id, page, page_size)


@router.patch("/{recipe_id}/records/{index}", response_model=RecipeRecord)
def update_recipe_record(
    recipe_id: str, index: int, payload: RecipeRecordUpdate
) -> RecipeRecord:
    return recipe_service.update_record(recipe_id, index, payload)


@router.post("/{recipe_id}/records", response_model=RecipeRecord)
def add_recipe_record(recipe_id: str, payload: RecipeRecordCreate) -> RecipeRecord:
    return recipe_service.add_record(recipe_id, payload)


@router.post("/{recipe_id}/records/delete", response_model=RecipeRead)
def delete_recipe_records(
    recipe_id: str, payload: RecipeRecordDeleteRequest
) -> RecipeRead:
    return recipe_service.delete_records(recipe_id, payload.indices)


@router.post("/{recipe_id}/commit", response_model=RecipeCommitResponse)
def commit_recipe(
    recipe_id: str, payload: RecipeCommitRequest, db: Session = Depends(get_db)
) -> RecipeCommitResponse:
    recipe = recipe_service.get_recipe(recipe_id)
    _require_llm_project(db, recipe.project_id)
    return recipe_service.commit(recipe_id, payload)
