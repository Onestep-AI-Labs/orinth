"""CRUD and validation orchestration for the architecture studio (phase 17).

Holds no TensorFlow. Everything here is graph manipulation, persistence, and
delegation to the pure modules under `app.ml.architecture`; the authoritative
model build runs in a subprocess owned by the training layer.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.storage import Storage
from app.db.models import Architecture as ArchitectureRow
from app.ml.architecture.emit_keras import MODULE_FILENAME, EmitError, emit_module
from app.ml.architecture.emit_torch import TORCH_MODULE_FILENAME, emit_torch_module
from app.ml.architecture.graph import build_graph
from app.ml.architecture.layout import auto_layout
from app.ml.architecture.shapes import infer_shapes
from app.ml.architecture.templates import template, templates
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
)

# Bumped when the stored graph shape changes in a way older files need
# migrating for. Files from a newer schema are refused rather than guessed at.
CURRENT_SCHEMA_VERSION = 1


class ArchitectureService:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage

    # --- reads -------------------------------------------------------------

    def list_architectures(
        self, db: Session, project_id: str | None = None, task_type: str | None = None
    ) -> list[ArchitectureSummary]:
        statement = select(ArchitectureRow).order_by(ArchitectureRow.updated_at.desc())
        if project_id:
            statement = statement.where(ArchitectureRow.project_id == project_id)
        if task_type:
            statement = statement.where(ArchitectureRow.task_type == task_type)
        return [_summary(row) for row in db.scalars(statement).all()]

    def get_architecture(self, db: Session, architecture_id: str) -> Architecture:
        return _read(self._row(db, architecture_id))

    def templates(self, task_type: str | None = None) -> list[ArchitectureTemplate]:
        return templates(task_type)

    # --- writes ------------------------------------------------------------

    def create_architecture(self, db: Session, payload: ArchitectureCreate) -> Architecture:
        graph = payload.graph
        task_type = payload.task_type
        if payload.template_id:
            starter = template(payload.template_id)
            if starter is None:
                raise HTTPException(
                    status_code=404, detail=f"Unknown template {payload.template_id!r}"
                )
            graph = starter.graph
            task_type = payload.task_type or starter.task_type
        now = datetime.now(UTC).replace(tzinfo=None)
        row = ArchitectureRow(
            id=f"arch_{uuid4().hex[:12]}",
            project_id=payload.project_id,
            name=payload.name,
            description=payload.description,
            task_type=task_type,
            framework="keras",
            version=1,
            graph=(graph or ArchitectureGraph()).model_dump(),
            created_at=now,
            updated_at=now,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        self._snapshot(row)
        return _read(row)

    def update_architecture(
        self, db: Session, architecture_id: str, payload: ArchitectureUpdate
    ) -> Architecture:
        row = self._row(db, architecture_id)
        # A version the client did not read means another session saved in
        # between. Refuse rather than silently discarding their work; the
        # studio offers reload-or-overwrite on the 409.
        if payload.version is not None and payload.version != row.version:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"This architecture was saved elsewhere (now at version {row.version}, "
                    f"you have {payload.version}). Reload before saving."
                ),
            )
        if payload.name is not None:
            row.name = payload.name
        if payload.description is not None:
            row.description = payload.description
        if payload.task_type is not None:
            row.task_type = payload.task_type
        if payload.graph is not None:
            row.graph = payload.graph.model_dump()
            row.version += 1
        row.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()
        db.refresh(row)
        if payload.graph is not None:
            self._snapshot(row)
        return _read(row)

    def duplicate_architecture(self, db: Session, architecture_id: str) -> Architecture:
        source = self._row(db, architecture_id)
        return self.create_architecture(
            db,
            ArchitectureCreate(
                name=f"{source.name} copy",
                description=source.description,
                project_id=source.project_id,
                task_type=source.task_type,  # type: ignore[arg-type]
                graph=ArchitectureGraph.model_validate(source.graph or {}),
            ),
        )

    def delete_architecture(self, db: Session, architecture_id: str) -> DeleteResponse:
        row = self._row(db, architecture_id)
        db.delete(row)
        db.commit()
        directory = self._snapshot_dir(architecture_id)
        if directory.exists():
            for path in sorted(directory.iterdir()):
                path.unlink()
            directory.rmdir()
        return DeleteResponse(deleted=1)

    # --- validation and codegen -------------------------------------------

    def validate_graph(
        self, graph: ArchitectureGraph, num_classes: int | None = None
    ) -> ArchitectureValidation:
        """The fast analytic pass. Structural checks plus shape propagation."""

        resolved = build_graph(graph)
        shapes, params_estimate, shape_issues = infer_shapes(resolved, num_classes)
        issues = [*resolved.issues, *shape_issues]
        return ArchitectureValidation(
            ok=not any(issue.severity == "error" for issue in issues),
            issues=issues,
            node_shapes=shapes,
            total_params_estimate=params_estimate,
            layer_count=len(resolved.order),
        )

    def generate_code(
        self,
        db: Session,
        architecture_id: str,
        num_classes: int = 2,
        framework: str = "keras",
    ) -> ArchitectureCode:
        row = self._row(db, architecture_id)
        code = self.render_code(
            ArchitectureGraph.model_validate(row.graph or {}),
            name=row.name,
            architecture_id=row.id,
            version=row.version,
            num_classes=num_classes,
            framework=framework,
        )
        return ArchitectureCode(
            architecture_id=row.id,
            filename=TORCH_MODULE_FILENAME if framework == "torch" else MODULE_FILENAME,
            code=code,
            framework="torch" if framework == "torch" else "keras",
        )

    def render_code(
        self,
        graph: ArchitectureGraph,
        *,
        name: str,
        architecture_id: str = "",
        version: int = 1,
        num_classes: int = 2,
        framework: str = "keras",
    ) -> str:
        """Emit the module, turning a generation failure into a 400 with the reason.

        Both emitters read the same resolved graph and the same inferred
        shapes; only the rendering differs, which is what keeps a PyTorch
        export from drifting away from the Keras one.
        """

        resolved = build_graph(graph)
        shapes, _params, shape_issues = infer_shapes(resolved, num_classes)
        blocking = [
            issue
            for issue in (*resolved.issues, *shape_issues)
            if issue.severity == "error"
        ]
        if blocking:
            raise HTTPException(
                status_code=400,
                detail="Fix these before generating code: "
                + "; ".join(issue.message for issue in blocking),
            )
        emit = emit_torch_module if framework == "torch" else emit_module
        try:
            return emit(
                resolved,
                shapes,
                architecture_name=name,
                architecture_id=architecture_id,
                version=version,
                default_num_classes=num_classes,
            )
        except EmitError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    # --- import/export -----------------------------------------------------

    def export_payload(self, db: Session, architecture_id: str) -> dict[str, Any]:
        row = self._row(db, architecture_id)
        return {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "name": row.name,
            "description": row.description,
            "task_type": row.task_type,
            "framework": row.framework,
            "graph": row.graph,
        }

    def import_payload(
        self, db: Session, raw: bytes, project_id: str | None = None
    ) -> Architecture:
        """Create an architecture from an exported `.json` file."""

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HTTPException(
                status_code=400, detail=f"That file is not valid JSON: {error}"
            ) from error
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Expected a JSON object.")
        version = int(data.get("schema_version", CURRENT_SCHEMA_VERSION))
        if version > CURRENT_SCHEMA_VERSION:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"This file was exported by a newer version of the studio "
                    f"(schema {version}, this build reads {CURRENT_SCHEMA_VERSION})."
                ),
            )
        try:
            graph = ArchitectureGraph.model_validate(data.get("graph") or {})
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail=f"That file's graph is malformed: {error}"
            ) from error
        # An export written by hand, or one produced by a tool that does not
        # track canvas coordinates, still opens — it just gets laid out. Nodes
        # all sitting at the origin means "unpositioned", not "stacked".
        if graph.nodes and all(_unpositioned(node.position) for node in graph.nodes):
            graph = auto_layout(graph)
        return self.create_architecture(
            db,
            ArchitectureCreate(
                name=str(data.get("name") or "Imported architecture")[:120],
                description=data.get("description"),
                project_id=project_id,
                task_type=data.get("task_type") or "classification",
                graph=graph,
            ),
        )

    # --- internals ---------------------------------------------------------

    def _row(self, db: Session, architecture_id: str) -> ArchitectureRow:
        row = db.get(ArchitectureRow, architecture_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Architecture not found")
        return row

    def _snapshot_dir(self, architecture_id: str) -> Path:
        return self.storage.architectures / architecture_id

    def _snapshot(self, row: ArchitectureRow) -> None:
        """Write the saved graph to storage alongside the DB row.

        Cheap insurance: the DB holds only the current version, so without
        this a bad edit followed by a save is unrecoverable.
        """

        directory = self._snapshot_dir(row.id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"v{row.version}.json").write_text(
            json.dumps(
                {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "name": row.name,
                    "task_type": row.task_type,
                    "version": row.version,
                    "saved_at": row.updated_at.isoformat(),
                    "graph": row.graph,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _unpositioned(position: dict[str, float]) -> bool:
    return not position or (position.get("x", 0.0) == 0.0 and position.get("y", 0.0) == 0.0)


def _read(row: ArchitectureRow) -> Architecture:
    return Architecture(
        id=row.id,
        project_id=row.project_id,
        name=row.name,
        description=row.description,
        task_type=row.task_type,  # type: ignore[arg-type]
        framework="keras",
        version=row.version,
        graph=ArchitectureGraph.model_validate(row.graph or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _summary(row: ArchitectureRow) -> ArchitectureSummary:
    return ArchitectureSummary(
        id=row.id,
        project_id=row.project_id,
        name=row.name,
        description=row.description,
        task_type=row.task_type,  # type: ignore[arg-type]
        framework="keras",
        version=row.version,
        node_count=len((row.graph or {}).get("nodes") or []),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
