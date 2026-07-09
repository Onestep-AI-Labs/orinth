import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.database import SessionLocal
from app.core.storage import Storage
from app.db.models import EvaluationJob
from app.ml.model_registry import ModelRegistry
from app.schemas import Detection, EvaluationDatasetInfo, EvaluationJobCreate, InferenceParameters
from app.services.metrics import (
    bbox_iou,
    binary_mask_metrics,
    classification_metrics,
    image_label_from_class_ids,
    image_label_from_detections,
)


@dataclass
class GroundTruthObject:
    class_id: int
    polygon: list[list[float]]
    bbox: tuple[float, float, float, float]


@dataclass
class EvaluationSample:
    image_path: Path
    objects: list[GroundTruthObject]


class EvaluationService:
    def __init__(self, settings: Settings, storage: Storage, registry: ModelRegistry) -> None:
        self.settings = settings
        self.storage = storage
        self.registry = registry

    def list_datasets(self) -> list[EvaluationDatasetInfo]:
        datasets_root = self.settings.datasets_path
        yolo_path = datasets_root / "dental dataset_yolov11_format" / "test"
        coco_path = datasets_root / "dental dataset_coco_format" / "test"
        return [
            EvaluationDatasetInfo(
                key="yolo_test",
                name="YOLOv11 Test Split",
                format="yolo",
                split="test",
                available=(yolo_path / "images").exists() and (yolo_path / "labels").exists(),
                path=str(yolo_path),
            ),
            EvaluationDatasetInfo(
                key="coco_test",
                name="COCO Test Split",
                format="coco",
                split="test",
                available=(coco_path / "_annotations.coco.json").exists(),
                path=str(coco_path),
            ),
        ]

    def create_job(self, db: Session, payload: EvaluationJobCreate) -> EvaluationJob:
        self.registry.get_spec(payload.model_id)
        if payload.dataset_key not in {dataset.key for dataset in self.list_datasets()}:
            raise KeyError(payload.dataset_key)
        job = EvaluationJob(
            id=uuid4().hex,
            model_id=payload.model_id,
            dataset_key=payload.dataset_key,
            status="queued",
            limit=payload.limit,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    def list_jobs(self, db: Session, limit: int = 25) -> list[EvaluationJob]:
        return db.scalars(
            select(EvaluationJob).order_by(EvaluationJob.created_at.desc()).limit(limit)
        ).all()

    def run_job(self, job_id: str) -> None:
        db = SessionLocal()
        try:
            job = db.get(EvaluationJob, job_id)
            if job is None:
                return
            job.status = "running"
            job.updated_at = datetime.utcnow()
            db.commit()

            metrics, artifacts = self._evaluate(job)
            job.status = "completed"
            job.metrics = metrics
            job.artifacts = artifacts
            job.updated_at = datetime.utcnow()
            db.commit()
        except Exception as exc:  # noqa: BLE001 - background jobs must persist errors
            job = db.get(EvaluationJob, job_id)
            if job is not None:
                job.status = "failed"
                job.error = str(exc)
                job.updated_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

    def _evaluate(self, job: EvaluationJob) -> tuple[dict, dict]:
        samples = self._load_samples(job.dataset_key)
        if job.limit:
            samples = samples[: job.limit]
        if not samples:
            raise ValueError(f"No samples found for dataset: {job.dataset_key}")

        predictor = self.registry.get_predictor(job.model_id)
        parameters = InferenceParameters()
        pixel_metric_rows = []
        image_true: list[int] = []
        image_pred: list[int] = []
        object_confusion = np.zeros((3, 3), dtype=int)
        object_totals = {"gt": 0, "pred": 0, "matched": 0}
        per_image = []

        for sample in samples:
            with Image.open(sample.image_path).convert("RGB") as image:
                width, height = image.size
            detections = predictor.predict(sample.image_path, parameters)
            gt_mask = rasterize_ground_truth(sample.objects, width, height)
            pred_mask = rasterize_detections(detections, width, height)
            pixel_metric_rows.append(binary_mask_metrics(gt_mask, pred_mask))

            self._update_object_metrics(sample.objects, detections, object_confusion, object_totals)
            gt_label = image_label_from_class_ids(obj.class_id for obj in sample.objects)
            pred_label = image_label_from_detections(detections)
            image_true.append(gt_label)
            image_pred.append(pred_label)
            if len(per_image) < 100:
                per_image.append(
                    {
                        "image": sample.image_path.name,
                        "ground_truth": gt_label,
                        "prediction": pred_label,
                        "detections": len(detections),
                        "objects": len(sample.objects),
                    }
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
            "image": classification_metrics(image_true, image_pred),
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
        raise KeyError(dataset_key)

    def _load_yolo_samples(self) -> list[EvaluationSample]:
        split = self.settings.datasets_path / "dental dataset_yolov11_format" / "test"
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
        category_map = {1: 0, 2: 1}
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
    ) -> None:
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
                confusion[gt.class_id, detections[best_idx].class_id] += 1
            else:
                confusion[gt.class_id, 2] += 1

        for idx, detection in enumerate(detections):
            if idx not in matched_pred:
                confusion[2, detection.class_id] += 1


def polygon_bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def rasterize_ground_truth(objects: list[GroundTruthObject], width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for obj in objects:
        if len(obj.polygon) >= 3:
            draw.polygon([(x, y) for x, y in obj.polygon], fill=1)
    return np.asarray(mask, dtype=np.uint8)


def rasterize_detections(detections: list[Detection], width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for detection in detections:
        if len(detection.polygon) >= 3:
            draw.polygon([(x, y) for x, y in detection.polygon], fill=1)
    return np.asarray(mask, dtype=np.uint8)


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: round(float(np.mean([row[key] for row in rows])), 6) for key in keys}
