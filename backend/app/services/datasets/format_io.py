"""Dataset annotation/label format IO: JSON sidecars, YOLO labels, COCO, and NLP import.

This is a mixin consumed by ``DatasetService`` (see ``service.py``); it relies on
core helpers (``storage``, ``_is_nlp_task``, ``_clean_label``, ``_normalize_labels``,
``_new_dataset_id``, ``_default_format_for_task``, ``_create_layout``,
``_write_manifest``, ``summary``) provided by the other mixins composed onto the
facade class.
"""


import csv
import json
from ast import literal_eval
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import HTTPException

from app.core.storage import Storage
from app.schemas import (
    Box,
    DatasetAnnotation,
    DatasetImportRequest,
    DatasetSplitConfig,
    DatasetSummary,
)
from app.services.datasets.constants import SPLITS, TEXT_SUFFIXES
from app.services.datasets.types import DatasetLocation


class FormatIoMixin:
    """YOLO/COCO/NLP annotation and dataset import-export helpers."""

    storage: Storage


    def _annotations(
        self, location: DatasetLocation, split: str, image_path: Path, width: int, height: int
    ) -> list[DatasetAnnotation]:
        json_path = location.root / split / "annotations" / f"{image_path.stem}.json"
        if json_path.exists():
            return self._read_annotation_json(json_path, location.labels)
        if location.format == "yolo":
            return self._yolo_annotations(location, split, image_path, width, height)
        if location.format == "coco":
            return self._coco_annotations(location, split, image_path.name)
        return []

    def _read_annotation_json(self, path: Path, labels: list[str]) -> list[DatasetAnnotation]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        rows = payload.get("annotations", payload if isinstance(payload, list) else [])
        annotations = []
        for row in rows:
            try:
                class_id = int(row.get("class_id", 0))
            except (TypeError, ValueError):
                continue
            class_name = labels[class_id] if 0 <= class_id < len(labels) else row.get("class_name", str(class_id))
            bbox = row.get("bbox")
            annotations.append(
                DatasetAnnotation(
                    class_id=class_id,
                    class_name=class_name,
                    kind=row.get("kind", "polygon"),
                    bbox=Box.model_validate(bbox) if bbox else None,
                    polygon=row.get("polygon", []),
                    text=row.get("text") or row.get("summary"),
                    question=row.get("question"),
                    answer=row.get("answer"),
                )
            )
        return annotations

    def _write_annotation_json(self, path: Path, annotations: list[DatasetAnnotation]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"annotations": [annotation.model_dump(mode="json") for annotation in annotations]},
                indent=2,
            ),
            encoding="utf-8",
        )

    def _normalize_annotation(
        self, annotation: DatasetAnnotation, location: DatasetLocation, width: int, height: int
    ) -> DatasetAnnotation | None:
        if self._is_nlp_task(location.task_type):
            if location.task_type == "text_classification":
                return self._classification_annotation(
                    location,
                    annotation.class_id,
                    annotation.class_name or None,
                )
            if location.task_type == "summarization":
                summary = (annotation.text or annotation.answer or "").strip()
                if not summary:
                    return None
                return DatasetAnnotation(
                    class_id=0,
                    class_name=location.labels[0] if location.labels else "summary",
                    kind="summary",
                    text=summary,
                    answer=summary,
                    polygon=[],
                    bbox=None,
                )
            question = (annotation.question or "").strip()
            answer = (annotation.answer or annotation.text or "").strip()
            if not question or not answer:
                return None
            return DatasetAnnotation(
                class_id=0,
                class_name=location.labels[0] if location.labels else "answer",
                kind="qa",
                question=question,
                answer=answer,
                text=answer,
                polygon=[],
                bbox=None,
            )

        if annotation.class_id >= len(location.labels):
            return None
        class_name = location.labels[annotation.class_id]
        if location.task_type == "classification":
            return DatasetAnnotation(
                class_id=annotation.class_id,
                class_name=class_name,
                kind="classification",
                polygon=[],
                bbox=None,
            )

        if location.task_type == "object_detection":
            bbox = annotation.bbox or self._bbox_from_polygon(annotation.polygon)
            if bbox is None or bbox.width <= 0 or bbox.height <= 0:
                return None
            bbox = Box(
                x=min(max(bbox.x, 0.0), float(width)),
                y=min(max(bbox.y, 0.0), float(height)),
                width=min(max(bbox.width, 0.0), float(width)),
                height=min(max(bbox.height, 0.0), float(height)),
            )
            return DatasetAnnotation(
                class_id=annotation.class_id,
                class_name=class_name,
                kind="box",
                bbox=bbox,
                polygon=annotation.polygon,
            )

        points = [
            [
                min(max(point[0], 0.0), float(width)),
                min(max(point[1], 0.0), float(height)),
            ]
            for point in annotation.polygon
        ]
        if len(points) < 3:
            return None
        return DatasetAnnotation(
            class_id=annotation.class_id,
            class_name=class_name,
            kind="polygon",
            bbox=annotation.bbox,
            polygon=points,
        )

    def _classification_annotation(
        self, location: DatasetLocation, class_id: int | None, class_name: str | None
    ) -> DatasetAnnotation:
        resolved_id = self._resolve_label_id(location, class_id, class_name)
        return DatasetAnnotation(
            class_id=resolved_id,
            class_name=location.labels[resolved_id],
            kind="classification",
            bbox=None,
            polygon=[],
        )

    def _resolve_label_id(
        self, location: DatasetLocation, class_id: int | None, class_name: str | None
    ) -> int:
        if class_id is not None:
            if 0 <= class_id < len(location.labels):
                return class_id
            raise HTTPException(status_code=400, detail="Label id is outside dataset labels")
        if class_name:
            cleaned = self._clean_label(class_name)
            if cleaned in location.labels:
                return location.labels.index(cleaned)
            raise HTTPException(status_code=400, detail="Label name is not in this dataset")
        raise HTTPException(status_code=400, detail="Classification label is required")

    def _bbox_from_polygon(self, polygon: list[list[float]]) -> Box | None:
        if len(polygon) < 2:
            return None
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return Box(x=min(xs), y=min(ys), width=max(xs) - min(xs), height=max(ys) - min(ys))

    def _write_yolo_label(
        self,
        location: DatasetLocation,
        split: str,
        stem: str,
        annotations: list[DatasetAnnotation],
        width: int,
        height: int,
    ) -> None:
        label_path = location.root / split / "labels" / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if location.format != "yolo" or location.task_type == "classification":
            label_path.write_text("", encoding="utf-8")
            return
        lines = []
        for annotation in annotations:
            if annotation.kind == "box" and annotation.bbox:
                bbox = annotation.bbox
                cx = (bbox.x + bbox.width / 2.0) / max(width, 1)
                cy = (bbox.y + bbox.height / 2.0) / max(height, 1)
                bw = bbox.width / max(width, 1)
                bh = bbox.height / max(height, 1)
                lines.append(f"{annotation.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            elif annotation.kind == "polygon" and len(annotation.polygon) >= 3:
                points = [
                    (
                        min(max(point[0], 0.0), float(width)) / max(width, 1),
                        min(max(point[1], 0.0), float(height)) / max(height, 1),
                    )
                    for point in annotation.polygon
                ]
                coords = " ".join(f"{value:.6f}" for point in points for value in point)
                lines.append(f"{annotation.class_id} {coords}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _yolo_annotations(
        self, location: DatasetLocation, split: str, image_path: Path, width: int, height: int
    ) -> list[DatasetAnnotation]:
        label_path = location.root / split / "labels" / f"{image_path.stem}.txt"
        annotations: list[DatasetAnnotation] = []
        if not label_path.exists():
            return annotations
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                continue
            class_name = location.labels[class_id] if class_id < len(location.labels) else str(class_id)
            if len(coords) == 4:
                cx, cy, bw, bh = coords
                bbox = Box(
                    x=(cx - bw / 2.0) * width,
                    y=(cy - bh / 2.0) * height,
                    width=bw * width,
                    height=bh * height,
                )
                annotations.append(
                    DatasetAnnotation(class_id=class_id, class_name=class_name, kind="box", bbox=bbox)
                )
            elif len(coords) >= 6:
                polygon = [
                    [coords[index] * width, coords[index + 1] * height]
                    for index in range(0, len(coords) - 1, 2)
                ]
                annotations.append(
                    DatasetAnnotation(
                        class_id=class_id,
                        class_name=class_name,
                        kind="polygon",
                        polygon=polygon,
                    )
                )
        return annotations

    def _coco_annotations(
        self, location: DatasetLocation, split: str, filename: str
    ) -> list[DatasetAnnotation]:
        data = self._coco_data(location, split)
        if not data:
            return []
        image_info = next(
            (item for item in data.get("images", []) if item.get("file_name") == filename), None
        )
        if not image_info:
            return []
        category_map = {
            category["id"]: index
            for index, category in enumerate(data.get("categories", []))
            if category.get("name") in location.labels
        }
        annotations = []
        for ann in data.get("annotations", []):
            if ann.get("image_id") != image_info.get("id") or ann.get("category_id") not in category_map:
                continue
            class_id = category_map[ann["category_id"]]
            polygons = ann.get("segmentation") or []
            if polygons and isinstance(polygons[0], list):
                points = polygons[0]
                polygon = [
                    [float(points[index]), float(points[index + 1])]
                    for index in range(0, len(points) - 1, 2)
                ]
                annotations.append(
                    DatasetAnnotation(
                        class_id=class_id,
                        class_name=location.labels[class_id],
                        kind="polygon",
                        polygon=polygon,
                    )
                )
            else:
                x, y, w, h = [float(value) for value in ann.get("bbox", [0, 0, 0, 0])]
                annotations.append(
                    DatasetAnnotation(
                        class_id=class_id,
                        class_name=location.labels[class_id],
                        kind="box",
                        bbox=Box(x=x, y=y, width=w, height=h),
                    )
                )
        return annotations

    def _coco_data(self, location: DatasetLocation, split: str) -> dict | None:
        path = location.root / split / "_annotations.coco.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _labels_from_coco(self, split_root: Path) -> list[str]:
        path = split_root / "_annotations.coco.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [category["name"] for category in data.get("categories", []) if category.get("name")]

    def _labels_from_yolo_yaml(self, root: Path) -> list[str]:
        yaml_path = root / "data.yaml"
        if not yaml_path.exists():
            return []
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                value = stripped.split(":", 1)[1].strip()
                try:
                    parsed = literal_eval(value)
                except (SyntaxError, ValueError):
                    return []
                if isinstance(parsed, dict):
                    return [str(parsed[key]) for key in sorted(parsed)]
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
        return []

    def _write_data_yaml(self, root: Path, labels: list[str]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        names = "[" + ", ".join(repr(label) for label in labels) + "]"
        (root / "data.yaml").write_text(
            "\n".join(
                [
                    "train: train/images",
                    "val: valid/images",
                    "test: test/images",
                    "",
                    f"nc: {len(labels)}",
                    f"names: {names}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _is_yolo_root(self, root: Path) -> bool:
        return any((root / split / "images").exists() for split in SPLITS)

    def _text_upload_rows(self, raw: str, suffix: str) -> list[dict]:
        if suffix == ".jsonl":
            rows = []
            for line in raw.splitlines():
                if not line.strip():
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
            return rows
        if suffix == ".csv":
            return [dict(row) for row in csv.DictReader(raw.splitlines())]
        return [{"text": raw}]

    def _text_from_upload_row(self, row: dict) -> str:
        return self._string_value(row, "text", "content", "source", "document", "context", "body")

    def _string_value(self, row: dict, *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return ""

    def _annotation_from_text_row(
        self,
        location: DatasetLocation,
        row: dict,
        *,
        class_id: int | None = None,
        class_name: str | None = None,
    ) -> DatasetAnnotation | None:
        if location.task_type == "text_classification":
            raw_class_id = class_id
            if raw_class_id is None and self._string_value(row, "class_id"):
                try:
                    raw_class_id = int(self._string_value(row, "class_id"))
                except ValueError:
                    raw_class_id = None
            raw_class_name = class_name or self._string_value(row, "label", "class", "class_name")
            if raw_class_id is None and not raw_class_name:
                return None
            return self._classification_annotation(location, raw_class_id, raw_class_name)
        if location.task_type == "summarization":
            summary = self._string_value(row, "summary", "reference_summary", "target", "answer")
            if not summary:
                return None
            return DatasetAnnotation(
                class_id=0,
                class_name=location.labels[0],
                kind="summary",
                text=summary,
                answer=summary,
            )
        question = self._string_value(row, "question", "query")
        answer = self._string_value(row, "answer", "answers", "target")
        if not question or not answer:
            return None
        return DatasetAnnotation(
            class_id=0,
            class_name=location.labels[0],
            kind="qa",
            question=question,
            answer=answer,
            text=answer,
        )

    def _write_text_item(
        self,
        location: DatasetLocation,
        split: str,
        filename: str,
        text: str,
        annotation: DatasetAnnotation | None,
    ) -> Path:
        safe_stem = Path(filename).stem.replace(" ", "-")[:72] or "sample"
        target_name = f"{safe_stem}-{uuid4().hex[:8]}.txt"
        text_dir = location.root / split / "texts"
        text_dir.mkdir(parents=True, exist_ok=True)
        path = text_dir / target_name
        path.write_text(text, encoding="utf-8")
        self._write_annotation_json(
            location.root / split / "annotations" / f"{path.stem}.json",
            [annotation] if annotation else [],
        )
        (location.root / split / "labels" / f"{path.stem}.txt").parent.mkdir(parents=True, exist_ok=True)
        (location.root / split / "labels" / f"{path.stem}.txt").write_text("", encoding="utf-8")
        return path

    def _import_nlp_dataset(
        self, source: Path, payload: DatasetImportRequest, task_type: str
    ) -> DatasetSummary:
        labels = self._normalize_labels(payload.labels or self._default_labels_for_task(task_type), task_type)
        dataset_id = self._new_dataset_id(payload.name or source.stem)
        root = self.storage.datasets / dataset_id
        format_name = self._default_format_for_task(task_type, payload.format)
        self._create_layout(root, format_name, task_type, labels)
        location = DatasetLocation(
            id=dataset_id,
            project_id=payload.project_id,
            name=payload.name or source.stem,
            task_type=task_type,
            format=format_name,
            source="editable",
            root=root,
            editable=True,
            labels=labels,
            metadata={},
        )
        if source.is_dir():
            for text_file in sorted(source.rglob("*.txt")):
                text = text_file.read_text(encoding="utf-8", errors="replace")
                annotation = self._annotation_from_text_row(location, {"text": text})
                self._write_text_item(location, "unassigned", text_file.name, text, annotation)
        elif source.suffix.lower() in TEXT_SUFFIXES:
            rows = self._text_upload_rows(source.read_text(encoding="utf-8", errors="replace"), source.suffix.lower())
            for index, row in enumerate(rows, start=1):
                text = self._text_from_upload_row(row)
                annotation = self._annotation_from_text_row(location, row)
                self._write_text_item(location, "unassigned", f"{source.stem}-{index:04d}.txt", text, annotation)
        else:
            raise HTTPException(status_code=400, detail="Only .txt, .csv, .jsonl, or folders are supported for NLP import")
        self._write_manifest(
            root,
            dataset_id=dataset_id,
            name=payload.name or source.stem,
            format_name=format_name,
            project_id=payload.project_id,
            task_type=task_type,
            labels=labels,
            metadata={
                "imported_from": str(source),
                "split_config": DatasetSplitConfig().model_dump(mode="json"),
            },
        )
        return self.summary(dataset_id)

    if TYPE_CHECKING:
        # Provided by sibling mixins on the composed DatasetService facade.
        def _is_nlp_task(self, task_type: str) -> bool:
            raise NotImplementedError

        def _clean_label(self, label: str) -> str:
            raise NotImplementedError

        def _normalize_labels(self, labels: list[str], task_type: str = ...) -> list[str]:
            raise NotImplementedError

        def _new_dataset_id(self, name: str) -> str:
            raise NotImplementedError

        def _default_format_for_task(self, task_type: str, format_name: str) -> str:
            raise NotImplementedError

        def _create_layout(self, root: Path, format_name: str, task_type: str, labels: list[str]) -> None:
            raise NotImplementedError

        def _write_manifest(
            self,
            root: Path,
            *,
            dataset_id: str,
            name: str,
            format_name: str,
            project_id: str,
            task_type: str,
            labels: list[str],
            metadata: dict,
        ) -> None:
            raise NotImplementedError

        def summary(self, dataset_id: str) -> DatasetSummary:
            raise NotImplementedError

        def _default_labels_for_task(self, task_type: str) -> list[str]:
            raise NotImplementedError

