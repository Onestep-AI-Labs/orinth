from __future__ import annotations

import builtins
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.defaults import DEFAULT_PROJECT_ID
from app.core.storage import Storage
from app.db.models import InferenceJob, InferenceRun
from app.ml.model_registry import ModelRegistry
from app.ml.predictors.base import image_level_from_detections
from app.schemas import InferenceJobRead, InferenceParameters, InferenceResult
from app.services.image_utils import create_overlay
from app.services.job_progress import append_log, make_progress, progress_from_artifacts


class InferenceService:
    def __init__(self, storage: Storage, registry: ModelRegistry) -> None:
        self.storage = storage
        self.registry = registry

    async def run(
        self,
        db: Session,
        file: UploadFile | None,
        model_id: str,
        parameters: InferenceParameters,
        project_id: str = DEFAULT_PROJECT_ID,
        text_content: str | None = None,
    ) -> InferenceResult:
        self.storage.ensure()
        started = perf_counter()
        spec = self.registry.get_spec(model_id)
        if spec.task_type in {"text_classification", "summarization", "question_answering"}:
            if not text_content or not text_content.strip():
                raise HTTPException(status_code=400, detail="Text content is required for NLP inference")
            upload_started = perf_counter()
            _upload_id, text_path = self.storage.save_text_input(text_content)
            timings = {"upload_ms": elapsed_ms(upload_started)}
            return self._predict_text_and_store(
                db,
                text_path=text_path,
                text_content=text_content,
                model_id=model_id,
                parameters=parameters,
                project_id=project_id,
                timings=timings,
                started=started,
            )
        if file is None:
            raise HTTPException(status_code=400, detail="Image file is required for vision inference")
        upload_started = perf_counter()
        _upload_id, image_path = await self.storage.save_upload(file)
        timings = {"upload_ms": elapsed_ms(upload_started)}
        return self._predict_and_store(
            db,
            image_path=image_path,
            original_filename=file.filename,
            model_id=model_id,
            parameters=parameters,
            project_id=project_id,
            timings=timings,
            started=started,
        )

    async def create_job(
        self,
        db: Session,
        file: UploadFile | None,
        model_id: str,
        parameters: InferenceParameters,
        project_id: str = DEFAULT_PROJECT_ID,
        text_content: str | None = None,
    ) -> InferenceJob:
        self.storage.ensure()
        spec = self.registry.get_spec(model_id)
        input_type = "text" if spec.task_type in {"text_classification", "summarization", "question_answering"} else "image"
        original_filename: str | None
        if input_type == "text":
            if not text_content or not text_content.strip():
                raise HTTPException(status_code=400, detail="Text content is required for NLP inference")
            _upload_id, input_path = self.storage.save_text_input(text_content)
            original_filename = "text-input.txt"
        else:
            if file is None:
                raise HTTPException(status_code=400, detail="Image file is required for vision inference")
            _upload_id, input_path = await self.storage.save_upload(file)
            original_filename = file.filename
        job_id = uuid4().hex
        artifacts = append_log({}, "Upload saved")
        artifacts["input_type"] = input_type
        artifacts["progress"] = make_progress(
            percent=10,
            current_step="Upload saved",
            processed=1,
            total=5,
            started_at=datetime.now(UTC).replace(tzinfo=None),
            logs=artifacts["logs"],
        )
        job = InferenceJob(
            id=job_id,
            project_id=project_id,
            model_id=model_id,
            status="queued",
            input_path=str(input_path),
            original_filename=original_filename,
            parameters=parameters.model_dump(),
            artifacts=artifacts,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    def run_job(self, job_id: str) -> None:
        db = SessionLocal()
        started = perf_counter()
        started_at = datetime.now(UTC).replace(tzinfo=None)
        try:
            job = db.get(InferenceJob, job_id)
            if job is None:
                return

            job.status = "running"
            artifacts = append_log(job.artifacts, "Loading model")
            artifacts["progress"] = make_progress(
                percent=20,
                processed=1,
                total=5,
                current_step="Loading model",
                started_at=started_at,
                logs=artifacts["logs"],
            )
            job.artifacts = artifacts
            job.updated_at = datetime.now(UTC).replace(tzinfo=None)
            db.commit()

            timings: dict[str, int] = {}
            spec = self.registry.get_spec(job.model_id)
            if spec.task_type in {"text_classification", "summarization", "question_answering"}:
                text_path = Path(job.input_path)
                result = self._predict_text_and_store(
                    db,
                    text_path=text_path,
                    text_content=text_path.read_text(encoding="utf-8", errors="replace"),
                    model_id=job.model_id,
                    parameters=InferenceParameters.model_validate(job.parameters),
                    project_id=job.project_id or DEFAULT_PROJECT_ID,
                    timings=timings,
                    started=started,
                    progress_callback=lambda step, percent: self._update_job_progress(
                        db, job_id, step, percent, started_at
                    ),
                )
            else:
                result = self._predict_and_store(
                    db,
                    image_path=Path(job.input_path),
                    original_filename=job.original_filename,
                    model_id=job.model_id,
                    parameters=InferenceParameters.model_validate(job.parameters),
                    project_id=job.project_id or DEFAULT_PROJECT_ID,
                    timings=timings,
                    started=started,
                    progress_callback=lambda step, percent: self._update_job_progress(
                        db, job_id, step, percent, started_at
                    ),
                )

            job = db.get(InferenceJob, job_id)
            if job is None:
                return
            artifacts = append_log(job.artifacts, "Inference completed")
            artifacts["progress"] = make_progress(
                percent=100,
                processed=5,
                total=5,
                current_step="Completed",
                started_at=started_at,
                finished_at=datetime.now(UTC).replace(tzinfo=None),
                logs=artifacts["logs"],
            )
            job.status = "completed"
            job.result = result.model_dump(mode="json")
            job.artifacts = artifacts
            job.updated_at = datetime.now(UTC).replace(tzinfo=None)
            db.commit()
        except Exception as exc:  # noqa: BLE001 - background jobs must persist errors
            job = db.get(InferenceJob, job_id)
            if job is not None:
                artifacts = append_log(job.artifacts, f"Failed: {exc}")
                artifacts["progress"] = make_progress(
                    percent=100,
                    current_step="Failed",
                    started_at=started_at,
                    finished_at=datetime.now(UTC).replace(tzinfo=None),
                    logs=artifacts["logs"],
                )
                job.status = "failed"
                job.error = str(exc)
                job.artifacts = artifacts
                job.updated_at = datetime.now(UTC).replace(tzinfo=None)
                db.commit()
        finally:
            db.close()

    def get_job(self, db: Session, job_id: str) -> InferenceJobRead:
        job = db.get(InferenceJob, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Inference job not found")
        return inference_job_read(job)

    def _predict_and_store(
        self,
        db: Session,
        *,
        image_path: Path,
        original_filename: str | None,
        model_id: str,
        parameters: InferenceParameters,
        project_id: str,
        timings: dict[str, int],
        started: float,
        progress_callback=None,
    ) -> InferenceResult:
        inference_id = uuid4().hex

        try:
            model_started = perf_counter()
            predictor = self.registry.get_predictor(model_id)
            timings["model_load_ms"] = elapsed_ms(model_started)
            if progress_callback:
                progress_callback("Running prediction", 50)
            prediction_started = perf_counter()
            class_scores = predictor.classify(image_path, parameters)
            detections = [] if class_scores else predictor.predict(image_path, parameters)
            timings["prediction_ms"] = elapsed_ms(prediction_started)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if progress_callback:
            progress_callback("Rendering overlay", 75)
        image_label = (
            max(class_scores.items(), key=lambda item: item[1])[0]
            if class_scores
            else image_level_from_detections(detections)
        )
        overlay_path = self.storage.overlay_path(inference_id)
        overlay_started = perf_counter()
        create_overlay(image_path, detections, overlay_path)
        timings["overlay_ms"] = elapsed_ms(overlay_started)
        timings["total_ms"] = elapsed_ms(started)

        if progress_callback:
            progress_callback("Persisting result", 90)

        result = InferenceResult(
            id=inference_id,
            project_id=project_id,
            model_id=model_id,
            image_level_label=image_label,
            detections=detections,
            class_scores=class_scores or {},
            overlay_url=self.storage.media_url(overlay_path),
            original_url=self.storage.media_url(image_path),
            parameters=parameters,
            created_at=datetime.now(UTC).replace(tzinfo=None),
            duration_ms=timings["total_ms"],
            timings=timings,
        )

        db.add(
            InferenceRun(
                id=inference_id,
                project_id=project_id,
                model_id=model_id,
                original_filename=original_filename,
                input_path=str(image_path),
                overlay_path=str(overlay_path),
                image_level_label=image_label,
                parameters=parameters.model_dump(),
                result=result.model_dump(mode="json"),
            )
        )
        db.commit()
        return result

    def _predict_text_and_store(
        self,
        db: Session,
        *,
        text_path: Path,
        text_content: str,
        model_id: str,
        parameters: InferenceParameters,
        project_id: str,
        timings: dict[str, int],
        started: float,
        progress_callback=None,
    ) -> InferenceResult:
        inference_id = uuid4().hex
        try:
            model_started = perf_counter()
            predictor = self.registry.get_predictor(model_id)
            timings["model_load_ms"] = elapsed_ms(model_started)
            if progress_callback:
                progress_callback("Running text prediction", 60)
            prediction_started = perf_counter()
            nlp_result = predictor.predict_text(text_content, parameters)
            timings["prediction_ms"] = elapsed_ms(prediction_started)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        label = (
            nlp_result.get("label")
            or nlp_result.get("summary")
            or nlp_result.get("answer")
            or "Text result"
        )
        raw_scores = nlp_result.get("scores")
        class_scores: dict[str, float] = raw_scores if isinstance(raw_scores, dict) else {}
        timings["total_ms"] = elapsed_ms(started)
        result = InferenceResult(
            id=inference_id,
            project_id=project_id,
            model_id=model_id,
            input_type="text",
            image_level_label=str(label),
            detections=[],
            class_scores=class_scores,
            overlay_url=None,
            original_url=self.storage.media_url(text_path),
            text_content=text_content,
            nlp_result=nlp_result,
            parameters=parameters,
            created_at=datetime.now(UTC).replace(tzinfo=None),
            duration_ms=timings["total_ms"],
            timings=timings,
        )
        db.add(
            InferenceRun(
                id=inference_id,
                project_id=project_id,
                model_id=model_id,
                original_filename=text_path.name,
                input_path=str(text_path),
                overlay_path=None,
                image_level_label=str(label)[:64],
                parameters=parameters.model_dump(),
                result=result.model_dump(mode="json"),
            )
        )
        db.commit()
        return result

    def _update_job_progress(
        self, db: Session, job_id: str, step: str, percent: float, started_at: datetime
    ) -> None:
        job = db.get(InferenceJob, job_id)
        if job is None:
            return
        artifacts = append_log(job.artifacts, step)
        artifacts["progress"] = make_progress(
            percent=percent,
            processed=max(1, int(percent // 20)),
            total=5,
            current_step=step,
            started_at=started_at,
            logs=artifacts["logs"],
        )
        job.artifacts = artifacts
        job.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()

    def get(self, db: Session, inference_id: str) -> InferenceResult:
        record = db.get(InferenceRun, inference_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Inference result not found")
        result = dict(record.result or {})
        result.setdefault("project_id", record.project_id or DEFAULT_PROJECT_ID)
        return InferenceResult.model_validate(result)

    def list(
        self, db: Session, limit: int = 25, project_id: str | None = None
    ) -> list[InferenceResult]:
        query = select(InferenceRun).order_by(InferenceRun.created_at.desc()).limit(limit)
        if project_id:
            query = (
                select(InferenceRun)
                .where(or_(InferenceRun.project_id == project_id, InferenceRun.project_id.is_(None)))
                .order_by(InferenceRun.created_at.desc())
                .limit(limit)
            )
        records = db.scalars(query).all()
        results = []
        for record in records:
            result = dict(record.result or {})
            result.setdefault("project_id", record.project_id or DEFAULT_PROJECT_ID)
            results.append(InferenceResult.model_validate(result))
        return results

    def delete_runs(
        self, db: Session, ids: builtins.list[str] | None = None, project_id: str | None = None
    ) -> dict:
        query = select(InferenceRun)
        if ids:
            query = query.where(InferenceRun.id.in_(ids))
        if project_id:
            query = query.where(or_(InferenceRun.project_id == project_id, InferenceRun.project_id.is_(None)))
        records = db.scalars(query).all()
        found = {record.id for record in records}
        deleted = 0
        for record in records:
            self.storage.delete_owned_path(record.input_path)
            self.storage.delete_owned_path(record.overlay_path)
            db.delete(record)
            deleted += 1
        db.commit()
        blocked: list[str] = []
        missing: list[str] = [item_id for item_id in (ids or []) if item_id not in found]
        return {
            "deleted": deleted,
            "blocked": blocked,
            "missing": missing,
        }


def ensure_existing_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def inference_job_read(job: InferenceJob) -> InferenceJobRead:
    result = None
    if job.result:
        payload = dict(job.result)
        payload.setdefault("project_id", job.project_id or DEFAULT_PROJECT_ID)
        result = InferenceResult.model_validate(payload)
    return InferenceJobRead(
        id=job.id,
        project_id=job.project_id or DEFAULT_PROJECT_ID,
        model_id=job.model_id,
        status=job.status,
        progress=progress_from_artifacts(job.artifacts),
        result=result,
        artifacts=job.artifacts or {},
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)
