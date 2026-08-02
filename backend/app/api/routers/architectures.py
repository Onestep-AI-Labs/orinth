"""Architecture studio endpoints (phase 17).

Literal paths are declared before `/{architecture_id}` so FastAPI, which
matches in declaration order, does not read `node-catalog` as an id.
"""

import json

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import PlainTextResponse, Response
from sqlalchemy.orm import Session

from app.container import architecture_service
from app.core.database import get_db
from app.ml.architecture.catalog import CATEGORY_ORDER, node_catalog
from app.schemas import (
    Architecture,
    ArchitectureCode,
    ArchitectureCreate,
    ArchitectureGraph,
    ArchitectureSummary,
    ArchitectureTemplate,
    ArchitectureUpdate,
    ArchitectureValidation,
    DeleteResponse,
    NodeSpec,
)

router = APIRouter(prefix="/architectures")


@router.get("", response_model=list[ArchitectureSummary])
def list_architectures(
    project_id: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[ArchitectureSummary]:
    return architecture_service.list_architectures(db, project_id=project_id, task_type=task_type)


@router.post("", response_model=Architecture)
def create_architecture(
    payload: ArchitectureCreate, db: Session = Depends(get_db)
) -> Architecture:
    return architecture_service.create_architecture(db, payload)


@router.get("/node-catalog", response_model=list[NodeSpec])
def list_node_catalog(task_type: str | None = Query(default=None)) -> list[NodeSpec]:
    return node_catalog(task_type)


@router.get("/node-categories", response_model=list[str])
def list_node_categories() -> list[str]:
    """Palette category order, so the frontend does not hardcode a second copy."""

    return CATEGORY_ORDER


@router.get("/templates", response_model=list[ArchitectureTemplate])
def list_templates(task_type: str | None = Query(default=None)) -> list[ArchitectureTemplate]:
    return architecture_service.templates(task_type)


@router.post("/validate", response_model=ArchitectureValidation)
def validate_architecture(
    graph: ArchitectureGraph, num_classes: int | None = Query(default=None)
) -> ArchitectureValidation:
    """Analytic validation of an unsaved canvas. Stateless and fast — no TensorFlow."""

    return architecture_service.validate_graph(graph, num_classes)


@router.post("/import", response_model=Architecture)
async def import_architecture(
    file: UploadFile = File(...),
    project_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> Architecture:
    raw = await file.read()
    return architecture_service.import_payload(db, raw, project_id=project_id)


@router.get("/{architecture_id}", response_model=Architecture)
def get_architecture(architecture_id: str, db: Session = Depends(get_db)) -> Architecture:
    return architecture_service.get_architecture(db, architecture_id)


@router.put("/{architecture_id}", response_model=Architecture)
def update_architecture(
    architecture_id: str, payload: ArchitectureUpdate, db: Session = Depends(get_db)
) -> Architecture:
    return architecture_service.update_architecture(db, architecture_id, payload)


@router.delete("/{architecture_id}", response_model=DeleteResponse)
def delete_architecture(architecture_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    return architecture_service.delete_architecture(db, architecture_id)


@router.post("/{architecture_id}/duplicate", response_model=Architecture)
def duplicate_architecture(architecture_id: str, db: Session = Depends(get_db)) -> Architecture:
    return architecture_service.duplicate_architecture(db, architecture_id)


@router.get("/{architecture_id}/code", response_model=ArchitectureCode)
def architecture_code(
    architecture_id: str,
    num_classes: int = Query(default=2, ge=1, le=2_000_000),
    framework: str = Query(default="keras", pattern="^(keras|torch)$"),
    db: Session = Depends(get_db),
) -> ArchitectureCode:
    return architecture_service.generate_code(
        db, architecture_id, num_classes=num_classes, framework=framework
    )


@router.get("/{architecture_id}/code/download", response_class=PlainTextResponse)
def download_architecture_code(
    architecture_id: str,
    num_classes: int = Query(default=2, ge=1, le=2_000_000),
    framework: str = Query(default="keras", pattern="^(keras|torch)$"),
    db: Session = Depends(get_db),
) -> PlainTextResponse:
    result = architecture_service.generate_code(
        db, architecture_id, num_classes=num_classes, framework=framework
    )
    return PlainTextResponse(
        result.code,
        media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="{result.filename}"'},
    )


@router.get("/{architecture_id}/export")
def export_architecture(architecture_id: str, db: Session = Depends(get_db)) -> Response:
    payload = architecture_service.export_payload(db, architecture_id)
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{architecture_id}.json"'},
    )


@router.post("/{architecture_id}/validate", response_model=ArchitectureValidation)
def validate_saved_architecture(
    architecture_id: str,
    num_classes: int | None = Query(default=None),
    db: Session = Depends(get_db),
) -> ArchitectureValidation:
    architecture = architecture_service.get_architecture(db, architecture_id)
    return architecture_service.validate_graph(architecture.graph, num_classes)
