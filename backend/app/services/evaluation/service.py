import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.database import SessionLocal
from app.core.defaults import DEFAULT_LABELS, DEFAULT_PROJECT_ID
from app.core.storage import Storage
from app.db.models import EvaluationJob
from app.ml.model_registry import ModelRegistry
from app.schemas import (
    Detection,
    EvaluationDatasetInfo,
    EvaluationJobBatchCreate,
    EvaluationJobCreate,
    EvaluationPerImageRow,
    InferenceParameters,
)
from app.services.datasets import DatasetService
from app.services.evaluation.helpers import (
    average_metric_rows,
    classification_roc_auc,
    polygon_bbox,
    rasterize_detections,
    rasterize_ground_truth,
)
from app.services.evaluation.types import ClassificationSample, EvaluationSample, GroundTruthObject
from app.services.job_progress import append_log, make_progress
from app.services.metrics import (
    bbox_iou,
    binary_mask_metrics,
    classification_metrics,
    image_label_from_class_ids,
    image_label_from_detections,
    qa_scores,
    rouge_scores,
)


class EvaluationService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        registry: ModelRegistry,
        dataset_service: DatasetService | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.registry = registry
        self.dataset_service = dataset_service

    def list_datasets(
        self,
        project_id: str | None = None,
        task_type: str | None = None,
    ) -> list[EvaluationDatasetInfo]:
        datasets_root = self.settings.datasets_path
        yolo_path = datasets_root / "dental dataset_yolov11_format" / "test"
        coco_path = datasets_root / "dental dataset_coco_format" / "test"
        yolo_labels = (
            self.dataset_service.labels_for_dataset_key("yolo_test")
            if self.dataset_service is not None
            else DEFAULT_LABELS
        )
        coco_labels = (
            self.dataset_service.labels_for_dataset_key("coco_test")
            if self.dataset_service is not None
            else DEFAULT_LABELS
        )
        datasets = [
            EvaluationDatasetInfo(
                key="yolo_test",
                name="YOLO Test Split",
                project_id=DEFAULT_PROJECT_ID,
                task_type="segmentation",
                format="yolo",
                split="test",
                available=(yolo_path / "images").exists() and (yolo_path / "labels").exists(),
                path=str(yolo_path),
                labels=yolo_labels,
            ),
            EvaluationDatasetInfo(
                key="coco_test",
                name="COCO Test Split",
                project_id=DEFAULT_PROJECT_ID,
                task_type="segmentation",
                format="coco",
                split="test",
                available=(coco_path / "_annotations.coco.json").exists(),
                path=str(coco_path),
                labels=coco_labels,
            ),
        ]
        if self.dataset_service is not None:
            for dataset in self.dataset_service.list_datasets():
                sample_nlp = dataset.id.startswith("sample_") and dataset.task_type in {
                    "text_classification",
                    "summarization",
                    "question_answering",
                }
                if (not dataset.editable and not sample_nlp) or dataset.format not in {"yolo", "image_folder", "text_folder", "jsonl", "csv"}:
                    continue
                if project_id and dataset.project_id != project_id and not sample_nlp:
                    continue
                for split, summary in dataset.splits.items():
                    if split != "test":
                        continue
                    if summary.image_count <= 0 and summary.text_count <= 0 and summary.item_count <= 0:
                        continue
                    datasets.append(
                        EvaluationDatasetInfo(
                            key=f"dataset:{dataset.id}:{split}",
                            name=f"{dataset.name} {split}",
                            project_id=dataset.project_id,
                            task_type=dataset.task_type,
                            format=dataset.format,
                            split=split,
                            available=True,
                            path=str(self.dataset_service.split_root(dataset.id, split)),
                            labels=dataset.labels,
                        )
                    )
        if project_id:
            datasets = [
                dataset
                for dataset in datasets
                if dataset.project_id == project_id or dataset.key.startswith("dataset:sample_")
            ]
        if task_type:
            normalized_task = "text_classification" if task_type == "text" else task_type
            datasets = [dataset for dataset in datasets if dataset.task_type == normalized_task]
        return datasets

    def create_job(self, db: Session, payload: EvaluationJobCreate) -> EvaluationJob:
        spec = self.registry.get_spec(payload.model_id)
        dataset = self._dataset_info(payload.dataset_key, payload.project_id)
        if dataset is None:
            raise KeyError(payload.dataset_key)
        self._ensure_compatible(spec.task_type, dataset.task_type)
        job_id = uuid4().hex
        job = EvaluationJob(
            id=job_id,
            project_id=payload.project_id,
            comparison_id=job_id,
            model_id=payload.model_id,
            dataset_key=payload.dataset_key,
            status="queued",
            limit=payload.limit,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    def create_jobs(self, db: Session, payload: EvaluationJobBatchCreate) -> list[EvaluationJob]:
        dataset = self._dataset_info(payload.dataset_key, payload.project_id)
        if dataset is None:
            raise KeyError(payload.dataset_key)
        for model_id in payload.model_ids:
            spec = self.registry.get_spec(model_id)
            self._ensure_compatible(spec.task_type, dataset.task_type)
        comparison_id = uuid4().hex
        jobs = [
            EvaluationJob(
                id=uuid4().hex,
                project_id=payload.project_id,
                comparison_id=comparison_id,
                model_id=model_id,
                dataset_key=payload.dataset_key,
                status="queued",
                limit=payload.limit,
            )
            for model_id in payload.model_ids
        ]
        db.add_all(jobs)
        db.commit()
        for job in jobs:
            db.refresh(job)
        return jobs

    def comparison_jobs(self, db: Session, job_id: str) -> list[EvaluationJob]:
        job = db.get(EvaluationJob, job_id)
        if job is None:
            raise KeyError(job_id)
        comparison_id = job.comparison_id or job.id
        query = (
            select(EvaluationJob)
            .where(EvaluationJob.comparison_id == comparison_id)
            .order_by(EvaluationJob.created_at.asc())
        )
        jobs = list(db.scalars(query).all())
        if not jobs:
            jobs = [job]
        return jobs

    def list_jobs(
        self, db: Session, limit: int = 25, project_id: str | None = None
    ) -> list[EvaluationJob]:
        query = select(EvaluationJob).order_by(EvaluationJob.created_at.desc()).limit(limit)
        if project_id:
            query = (
                select(EvaluationJob)
                .where(or_(EvaluationJob.project_id == project_id, EvaluationJob.project_id.is_(None)))
                .order_by(EvaluationJob.created_at.desc())
                .limit(limit)
            )
        return list(db.scalars(query).all())

    def delete_jobs(self, db: Session, ids: list[str] | None = None, project_id: str | None = None) -> dict:
        query = select(EvaluationJob)
        if ids:
            query = query.where(EvaluationJob.id.in_(ids))
        if project_id:
            query = query.where(or_(EvaluationJob.project_id == project_id, EvaluationJob.project_id.is_(None)))
        jobs = db.scalars(query).all()
        deleted = 0
        blocked = []
        found = {job.id for job in jobs}
        for job in jobs:
            if job.status not in {"completed", "failed", "canceled"}:
                blocked.append(job.id)
                continue
            artifacts = job.artifacts or {}
            for key in ("metrics", "per_image"):
                self.storage.delete_owned_path(artifacts.get(key))
            self.storage.delete_owned_path(self.storage.evaluations / job.id)
            db.delete(job)
            deleted += 1
        db.commit()
        return {
            "deleted": deleted,
            "blocked": blocked,
            "missing": [item_id for item_id in (ids or []) if item_id not in found],
        }

    def run_job(self, job_id: str) -> None:
        db = SessionLocal()
        try:
            job = db.get(EvaluationJob, job_id)
            if job is None:
                return
            started_at = datetime.now(UTC).replace(tzinfo=None)
            job.status = "running"
            artifacts = append_log(job.artifacts, "Evaluation started")
            artifacts["progress"] = make_progress(
                percent=1,
                current_step="Loading dataset",
                started_at=started_at,
                logs=artifacts["logs"],
            )
            job.artifacts = artifacts
            job.updated_at = datetime.now(UTC).replace(tzinfo=None)
            db.commit()

            metrics, artifacts = self._evaluate(db, job, started_at)
            job.status = "completed"
            job.metrics = metrics
            merged_artifacts = {**(job.artifacts or {}), **artifacts}
            merged_artifacts = append_log(merged_artifacts, "Evaluation completed")
            merged_artifacts["progress"] = make_progress(
                percent=100,
                processed=metrics["samples"],
                total=metrics["samples"],
                current_step="Completed",
                started_at=started_at,
                finished_at=datetime.now(UTC).replace(tzinfo=None),
                logs=merged_artifacts["logs"],
            )
            job.artifacts = merged_artifacts
            job.updated_at = datetime.now(UTC).replace(tzinfo=None)
            db.commit()
        except Exception as exc:  # noqa: BLE001 - background jobs must persist errors
            job = db.get(EvaluationJob, job_id)
            if job is not None:
                artifacts = append_log(job.artifacts, f"Failed: {exc}")
                artifacts["progress"] = make_progress(
                    percent=100,
                    current_step="Failed",
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

    def reconcile_stale_jobs(self) -> None:
        db = SessionLocal()
        try:
            jobs = db.scalars(
                select(EvaluationJob).where(EvaluationJob.status.in_(["queued", "running"]))
            ).all()
            for job in jobs:
                artifacts = append_log(job.artifacts, "Backend restarted before evaluation completed")
                artifacts["progress"] = make_progress(
                    percent=100,
                    current_step="Failed",
                    finished_at=datetime.now(UTC).replace(tzinfo=None),
                    logs=artifacts["logs"],
                )
                job.status = "failed"
                job.error = "Backend restarted before evaluation completed"
                job.artifacts = artifacts
                job.updated_at = datetime.now(UTC).replace(tzinfo=None)
            db.commit()
        finally:
            db.close()

    def per_image_rows(self, db: Session, job_id: str) -> list[EvaluationPerImageRow]:
        job = db.get(EvaluationJob, job_id)
        if job is None:
            raise KeyError(job_id)
        path = (job.artifacts or {}).get("per_image")
        if not path:
            return []
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        return [EvaluationPerImageRow.model_validate(row) for row in rows]

    def _evaluate(
        self, db: Session, job: EvaluationJob, started_at: datetime
    ) -> tuple[dict, dict]:
        spec = self.registry.get_spec(job.model_id)
        if spec.task_type == "classification":
            return self._evaluate_classification(db, job, started_at)
        if spec.task_type in {"text_classification", "summarization", "question_answering"}:
            return self._evaluate_nlp(db, job, started_at, spec.task_type)

        samples = self._load_samples(job.dataset_key)
        labels = (
            self.dataset_service.labels_for_dataset_key(job.dataset_key)
            if self.dataset_service is not None
            else DEFAULT_LABELS
        )
        normal_index = len(labels)
        if job.limit:
            samples = samples[: job.limit]
        if not samples:
            raise ValueError(f"No samples found for dataset: {job.dataset_key}")

        predictor = self.registry.get_predictor(job.model_id)
        parameters = InferenceParameters()
        pixel_metric_rows = []
        image_true: list[int] = []
        image_pred: list[int] = []
        object_confusion = np.zeros((len(labels) + 1, len(labels) + 1), dtype=int)
        object_totals = {"gt": 0, "pred": 0, "matched": 0}
        per_image = []

        total = len(samples)
        self._update_progress(
            db,
            job,
            percent=5,
            processed=0,
            total=total,
            step="Running predictions",
            started_at=started_at,
        )

        for index, sample in enumerate(samples, start=1):
            with Image.open(sample.image_path).convert("RGB") as image:
                width, height = image.size
            detections = predictor.predict(sample.image_path, parameters)
            gt_mask = rasterize_ground_truth(sample.objects, width, height)
            pred_mask = rasterize_detections(detections, width, height)
            pixel_metrics = binary_mask_metrics(gt_mask, pred_mask)
            pixel_metric_rows.append(pixel_metrics)

            object_summary = self._update_object_metrics(
                sample.objects, detections, object_confusion, object_totals
            )
            gt_label = image_label_from_class_ids(
                (obj.class_id for obj in sample.objects), normal_index
            )
            pred_label = image_label_from_detections(detections, normal_index)
            image_true.append(gt_label)
            image_pred.append(pred_label)
            per_image.append(
                {
                    "image": sample.image_path.name,
                    "ground_truth": gt_label,
                    "prediction": pred_label,
                    "detections": len(detections),
                    "objects": len(sample.objects),
                    "pixel": pixel_metrics,
                    "object": object_summary,
                }
            )
            self._update_progress(
                db,
                job,
                percent=5 + (90 * index / total),
                processed=index,
                total=total,
                step="Running predictions",
                current_item=sample.image_path.name,
                started_at=started_at,
                log_every=index == 1 or index == total or index % 5 == 0,
            )

        metrics = {
            "samples": len(samples),
            "pixel": average_metric_rows(pixel_metric_rows),
            "object": {
                "confusion_matrix": object_confusion.tolist(),
                "matched": object_totals["matched"],
                "ground_truth_objects": object_totals["gt"],
                "predicted_objects": object_totals["pred"],
                "precision": round(
                    object_totals["matched"] / (object_totals["pred"] + 1e-7), 6
                ),
                "recall": round(object_totals["matched"] / (object_totals["gt"] + 1e-7), 6),
                "iou_threshold": 0.2,
            },
            "image": classification_metrics(image_true, image_pred, labels),
            "labels": labels,
        }
        output_dir = self.storage.evaluations / job.id
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        per_image_path = output_dir / "per_image.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        per_image_path.write_text(json.dumps(per_image, indent=2), encoding="utf-8")
        return metrics, {"metrics": str(metrics_path), "per_image": str(per_image_path)}

    def _evaluate_nlp(
        self, db: Session, job: EvaluationJob, started_at: datetime, task_type: str
    ) -> tuple[dict, dict]:
        samples = self._load_nlp_samples(job.dataset_key, task_type)
        labels = (
            self.dataset_service.labels_for_dataset_key(job.dataset_key)
            if self.dataset_service is not None
            else []
        )
        if job.limit:
            samples = samples[: job.limit]
        if not samples:
            raise ValueError(f"No NLP samples found for dataset: {job.dataset_key}")
        predictor = self.registry.get_predictor(job.model_id)
        total = len(samples)
        per_image = []
        self._update_progress(
            db,
            job,
            percent=5,
            processed=0,
            total=total,
            step="Running text evaluation",
            started_at=started_at,
        )

        if task_type == "text_classification":
            y_true: list[int] = []
            y_pred: list[int] = []
            for index, sample in enumerate(samples, start=1):
                result = predictor.predict_text(sample["text"], InferenceParameters())
                label = result.get("label", "")
                prediction = labels.index(label) if label in labels else 0
                ground_truth = int(sample["class_id"])
                y_true.append(ground_truth)
                y_pred.append(prediction)
                per_image.append(
                    {
                        "image": sample["id"],
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "detections": 0,
                        "objects": 1,
                        "pixel": {},
                        "object": {},
                        "text_preview": sample["text"][:180],
                        "scores": result.get("scores", {}),
                    }
                )
                self._update_progress(
                    db,
                    job,
                    percent=5 + (90 * index / total),
                    processed=index,
                    total=total,
                    step="Running text classification",
                    current_item=sample["id"],
                    started_at=started_at,
                    log_every=index == 1 or index == total or index % 10 == 0,
                )
            metrics = {
                "samples": len(samples),
                "labels": labels,
                "text_classification": classification_metrics(y_true, y_pred, labels),
            }
        elif task_type == "summarization":
            rows = []
            for index, sample in enumerate(samples, start=1):
                result = predictor.predict_text(sample["text"], InferenceParameters())
                prediction_text = str(result.get("summary", ""))
                scores = rouge_scores(prediction_text, sample["summary"])
                rows.append(scores)
                per_image.append(
                    {
                        "image": sample["id"],
                        "ground_truth": 0,
                        "prediction": 0,
                        "detections": 0,
                        "objects": 1,
                        "pixel": {},
                        "object": {},
                        "text_preview": sample["text"][:180],
                        "reference_text": sample["summary"],
                        "prediction_text": prediction_text,
                        "scores": scores,
                    }
                )
                self._update_progress(
                    db,
                    job,
                    percent=5 + (90 * index / total),
                    processed=index,
                    total=total,
                    step="Running summarization",
                    current_item=sample["id"],
                    started_at=started_at,
                    log_every=index == 1 or index == total or index % 10 == 0,
                )
            metrics = {"samples": len(samples), "summarization": average_dicts(rows)}
        else:
            rows = []
            for index, sample in enumerate(samples, start=1):
                params = InferenceParameters(question=sample["question"])
                result = predictor.predict_text(sample["text"], params)
                prediction_text = str(result.get("answer", ""))
                scores = qa_scores(prediction_text, sample["answer"])
                rows.append(scores)
                per_image.append(
                    {
                        "image": sample["id"],
                        "ground_truth": 0,
                        "prediction": 0,
                        "detections": 0,
                        "objects": 1,
                        "pixel": {},
                        "object": {},
                        "text_preview": sample["text"][:180],
                        "reference_text": sample["answer"],
                        "prediction_text": prediction_text,
                        "scores": scores,
                    }
                )
                self._update_progress(
                    db,
                    job,
                    percent=5 + (90 * index / total),
                    processed=index,
                    total=total,
                    step="Running question answering",
                    current_item=sample["id"],
                    started_at=started_at,
                    log_every=index == 1 or index == total or index % 10 == 0,
                )
            metrics = {"samples": len(samples), "question_answering": average_dicts(rows)}

        output_dir = self.storage.evaluations / job.id
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        per_image_path = output_dir / "per_image.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        per_image_path.write_text(json.dumps(per_image, indent=2), encoding="utf-8")
        return metrics, {"metrics": str(metrics_path), "per_image": str(per_image_path)}

    def _evaluate_classification(
        self, db: Session, job: EvaluationJob, started_at: datetime
    ) -> tuple[dict, dict]:
        samples = self._load_classification_samples(job.dataset_key)
        labels = (
            self.dataset_service.labels_for_dataset_key(job.dataset_key)
            if self.dataset_service is not None
            else DEFAULT_LABELS
        )
        if job.limit:
            samples = samples[: job.limit]
        if not samples:
            raise ValueError(f"No classification samples found for dataset: {job.dataset_key}")

        predictor = self.registry.get_predictor(job.model_id)
        parameters = InferenceParameters()
        y_true: list[int] = []
        y_pred: list[int] = []
        score_rows: list[list[float]] = []
        per_image = []
        total = len(samples)
        self._update_progress(
            db,
            job,
            percent=5,
            processed=0,
            total=total,
            step="Running classification",
            started_at=started_at,
        )

        for index, sample in enumerate(samples, start=1):
            scores = predictor.classify(sample.image_path, parameters) or {}
            ordered_scores = [float(scores.get(label, 0.0)) for label in labels]
            prediction = int(np.argmax(ordered_scores)) if ordered_scores else 0
            y_true.append(sample.class_id)
            y_pred.append(prediction)
            score_rows.append(ordered_scores)
            per_image.append(
                {
                    "image": sample.image_path.name,
                    "ground_truth": sample.class_id,
                    "prediction": prediction,
                    "detections": 0,
                    "objects": 1,
                    "pixel": {},
                    "object": {},
                    "scores": scores,
                }
            )
            self._update_progress(
                db,
                job,
                percent=5 + (90 * index / total),
                processed=index,
                total=total,
                step="Running classification",
                current_item=sample.image_path.name,
                started_at=started_at,
                log_every=index == 1 or index == total or index % 5 == 0,
            )

        image_metrics = classification_metrics(y_true, y_pred, labels)
        metrics = {
            "samples": len(samples),
            "labels": labels,
            "image": image_metrics,
            "classification": {
                "scores": True,
                **classification_roc_auc(y_true, score_rows, labels),
            },
        }
        output_dir = self.storage.evaluations / job.id
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        per_image_path = output_dir / "per_image.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        per_image_path.write_text(json.dumps(per_image, indent=2), encoding="utf-8")
        return metrics, {"metrics": str(metrics_path), "per_image": str(per_image_path)}

    def _load_samples(self, dataset_key: str) -> list[EvaluationSample]:
        if dataset_key == "yolo_test":
            return self._load_yolo_samples()
        if dataset_key == "coco_test":
            return self._load_coco_samples()
        if dataset_key.startswith("dataset:") and self.dataset_service is not None:
            _, dataset_id, split = dataset_key.split(":", 2)
            return self._load_yolo_samples_from_split(self.dataset_service.yolo_split(dataset_id, split))
        raise KeyError(dataset_key)

    def _load_classification_samples(self, dataset_key: str) -> list[ClassificationSample]:
        if not dataset_key.startswith("dataset:") or self.dataset_service is None:
            raise KeyError(dataset_key)
        _, dataset_id, split = dataset_key.split(":", 2)
        split_root = self.dataset_service.split_root(dataset_id, split)
        image_dir = split_root / "images"
        annotation_dir = split_root / "annotations"
        samples: list[ClassificationSample] = []
        if not image_dir.exists():
            return samples
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            annotation_path = annotation_dir / f"{image_path.stem}.json"
            if not annotation_path.exists():
                continue
            try:
                payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rows = payload.get("annotations", [])
            if not rows:
                continue
            try:
                class_id = int(rows[0].get("class_id", 0))
            except (TypeError, ValueError):
                continue
            samples.append(ClassificationSample(image_path=image_path, class_id=class_id))
        return samples

    def _load_nlp_samples(self, dataset_key: str, task_type: str) -> list[dict]:
        if not dataset_key.startswith("dataset:") or self.dataset_service is None:
            raise KeyError(dataset_key)
        _, dataset_id, split = dataset_key.split(":", 2)
        split_root = self.dataset_service.split_root(dataset_id, split)
        text_dir = split_root / "texts"
        annotation_dir = split_root / "annotations"
        samples: list[dict] = []
        if not text_dir.exists():
            return samples
        for text_path in sorted(text_dir.glob("*.txt")):
            annotation_path = annotation_dir / f"{text_path.stem}.json"
            if not annotation_path.exists():
                continue
            try:
                payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            rows = payload.get("annotations", [])
            text = text_path.read_text(encoding="utf-8", errors="replace")
            if task_type == "text_classification":
                annotation = next((row for row in rows if row.get("kind") == "classification"), None)
                if annotation is None:
                    continue
                samples.append(
                    {
                        "id": text_path.name,
                        "text": text,
                        "class_id": int(annotation.get("class_id", 0)),
                    }
                )
            elif task_type == "summarization":
                annotation = next((row for row in rows if row.get("kind") == "summary"), None)
                summary = (annotation or {}).get("text") or (annotation or {}).get("answer")
                if not summary:
                    continue
                samples.append({"id": text_path.name, "text": text, "summary": str(summary)})
            else:
                for annotation in rows:
                    if annotation.get("kind") != "qa":
                        continue
                    question = annotation.get("question")
                    answer = annotation.get("answer") or annotation.get("text")
                    if question and answer:
                        samples.append(
                            {
                                "id": text_path.name,
                                "text": text,
                                "question": str(question),
                                "answer": str(answer),
                            }
                        )
        return samples

    def _dataset_info(
        self, dataset_key: str, project_id: str | None = None
    ) -> EvaluationDatasetInfo | None:
        return next(
            (dataset for dataset in self.list_datasets(project_id) if dataset.key == dataset_key),
            None,
        )

    def _ensure_compatible(self, model_task: str, dataset_task: str) -> None:
        if model_task == dataset_task:
            return
        vision_shape_tasks = {"object_detection", "segmentation"}
        if model_task in vision_shape_tasks and dataset_task in vision_shape_tasks:
            return
        raise ValueError(f"Model task {model_task} is not compatible with dataset task {dataset_task}")

    def _load_yolo_samples(self) -> list[EvaluationSample]:
        split = self.settings.datasets_path / "dental dataset_yolov11_format" / "test"
        return self._load_yolo_samples_from_split(split)

    def _load_yolo_samples_from_split(self, split: Path) -> list[EvaluationSample]:
        image_dir = split / "images"
        label_dir = split / "labels"
        samples = []
        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            label_path = label_dir / f"{image_path.stem}.txt"
            objects = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                    polygon = [
                        [coords[index] * width, coords[index + 1] * height]
                        for index in range(0, len(coords), 2)
                    ]
                    objects.append(GroundTruthObject(class_id, polygon, polygon_bbox(polygon)))
            samples.append(EvaluationSample(image_path, objects))
        return samples

    def _load_coco_samples(self) -> list[EvaluationSample]:
        split = self.settings.datasets_path / "dental dataset_coco_format" / "test"
        data = json.loads((split / "_annotations.coco.json").read_text(encoding="utf-8"))
        labels = self.dataset_service.labels_for_dataset_key("coco_test") if self.dataset_service else DEFAULT_LABELS
        category_map = {
            category["id"]: labels.index(category["name"])
            for category in data.get("categories", [])
            if category.get("name") in labels
        }
        annotations_by_image: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            if ann.get("category_id") in category_map:
                annotations_by_image.setdefault(ann["image_id"], []).append(ann)

        samples = []
        for image_info in data.get("images", []):
            objects = []
            for ann in annotations_by_image.get(image_info["id"], []):
                polygons = ann.get("segmentation") or []
                if polygons and isinstance(polygons[0], list):
                    points = polygons[0]
                    polygon = [
                        [float(points[index]), float(points[index + 1])]
                        for index in range(0, len(points), 2)
                    ]
                else:
                    x, y, w, h = [float(value) for value in ann["bbox"]]
                    polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                objects.append(
                    GroundTruthObject(
                        class_id=category_map[ann["category_id"]],
                        polygon=polygon,
                        bbox=polygon_bbox(polygon),
                    )
                )
            samples.append(EvaluationSample(split / image_info["file_name"], objects))
        return samples

    def _update_object_metrics(
        self,
        gt_objects: list[GroundTruthObject],
        detections: list[Detection],
        confusion: np.ndarray,
        totals: dict[str, int],
    ) -> dict[str, int]:
        matched_start = totals["matched"]
        background_index = confusion.shape[0] - 1
        totals["gt"] += len(gt_objects)
        totals["pred"] += len(detections)
        matched_pred: set[int] = set()

        for gt in gt_objects:
            best_iou = 0.2
            best_idx = -1
            for idx, detection in enumerate(detections):
                if idx in matched_pred:
                    continue
                det_box = (
                    detection.bbox.x,
                    detection.bbox.y,
                    detection.bbox.x + detection.bbox.width,
                    detection.bbox.y + detection.bbox.height,
                )
                score = bbox_iou(gt.bbox, det_box)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx >= 0:
                matched_pred.add(best_idx)
                totals["matched"] += 1
                pred_class = detections[best_idx].class_id
                if 0 <= gt.class_id < background_index and 0 <= pred_class < background_index:
                    confusion[gt.class_id, pred_class] += 1
                elif 0 <= gt.class_id < background_index:
                    confusion[gt.class_id, background_index] += 1
            else:
                if 0 <= gt.class_id < background_index:
                    confusion[gt.class_id, background_index] += 1

        for idx, detection in enumerate(detections):
            if idx not in matched_pred:
                if 0 <= detection.class_id < background_index:
                    confusion[background_index, detection.class_id] += 1

        matched = totals["matched"] - matched_start
        return {
            "matched": matched,
            "false_negatives": max(0, len(gt_objects) - matched),
            "false_positives": max(0, len(detections) - matched),
        }

    def _update_progress(
        self,
        db: Session,
        job: EvaluationJob,
        *,
        percent: float,
        processed: int,
        total: int,
        step: str,
        started_at: datetime,
        current_item: str | None = None,
        log_every: bool = True,
    ) -> None:
        artifacts = dict(job.artifacts or {})
        if log_every:
            artifacts = append_log(
                artifacts,
                f"{step}: {processed}/{total}" + (f" {current_item}" if current_item else ""),
            )
        artifacts["progress"] = make_progress(
            percent=percent,
            processed=processed,
            total=total,
            current_step=step,
            current_item=current_item,
            started_at=started_at,
            logs=artifacts.get("logs", []),
        )
        job.artifacts = artifacts
        job.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.commit()


def average_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)
        for key in keys
    }
