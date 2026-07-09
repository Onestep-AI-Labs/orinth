from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.storage import Storage
from app.db.models import InferenceRun
from app.ml.model_registry import ModelRegistry
from app.ml.predictors.base import image_level_from_detections
from app.schemas import InferenceParameters, InferenceResult
from app.services.image_utils import create_overlay


class InferenceService:
    def __init__(self, storage: Storage, registry: ModelRegistry) -> None:
        self.storage = storage
        self.registry = registry

    async def run(
        self,
        db: Session,
        file: UploadFile,
        model_id: str,
        parameters: InferenceParameters,
    ) -> InferenceResult:
        self.storage.ensure()
        upload_id, image_path = await self.storage.save_upload(file)
        inference_id = uuid4().hex

        try:
            predictor = self.registry.get_predictor(model_id)
            detections = predictor.predict(image_path, parameters)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        image_label = image_level_from_detections(detections)
        overlay_path = self.storage.overlay_path(inference_id)
        create_overlay(image_path, detections, overlay_path)

        result = InferenceResult(
            id=inference_id,
            model_id=model_id,
            image_level_label=image_label,
            detections=detections,
            overlay_url=self.storage.media_url(overlay_path),
            original_url=self.storage.media_url(image_path),
            parameters=parameters,
            created_at=datetime.utcnow(),
        )

        db.add(
            InferenceRun(
                id=inference_id,
                model_id=model_id,
                original_filename=file.filename,
                input_path=str(image_path),
                overlay_path=str(overlay_path),
                image_level_label=image_label,
                parameters=parameters.model_dump(),
                result=result.model_dump(mode="json"),
            )
        )
        db.commit()
        return result

    def get(self, db: Session, inference_id: str) -> InferenceResult:
        record = db.get(InferenceRun, inference_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Inference result not found")
        return InferenceResult.model_validate(record.result)

    def list(self, db: Session, limit: int = 25) -> list[InferenceResult]:
        records = db.scalars(
            select(InferenceRun).order_by(InferenceRun.created_at.desc()).limit(limit)
        ).all()
        return [InferenceResult.model_validate(record.result) for record in records]


def ensure_existing_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved
