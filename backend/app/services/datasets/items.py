"""Dataset item and label CRUD: uploads, listing, per-item detail, and quick labels.

This is a mixin consumed by ``DatasetService`` (see ``service.py``); it relies on
core/format-IO helpers (``_location``, ``_editable_location``, ``_is_nlp_task``,
``_annotations``, ``_write_annotation_json``, ``_classification_annotation``,
``_normalize_annotation``, ``_write_yolo_label``)
provided by the other mixins composed onto the facade class.
"""


from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.schemas import (
    DatasetAnnotation,
    DatasetAnnotationSave,
    DatasetItemBatchUploadResponse,
    DatasetItemBulkLabelUpdate,
    DatasetItemBulkLabelUpdateResponse,
    DatasetItemDeleteRequest,
    DatasetItemDetail,
    DatasetItemLabelUpdate,
    DatasetItemPage,
    DatasetItemSummary,
    DatasetSplitSummary,
    DatasetSummary,
    DeleteResponse,
)
from app.services.datasets.constants import IMAGE_SUFFIXES, SPLITS, TEXT_SUFFIXES
from app.services.datasets.types import DatasetLocation


class ItemsMixin:
    """Per-item CRUD: uploads, listing, detail, quick labels, and label management."""


    def add_label(self, dataset_id: str, name: str) -> DatasetSummary:
        location = self._editable_location(dataset_id)
        labels = location.labels.copy()
        normalized = self._clean_label(name)
        if normalized in labels:
            raise HTTPException(status_code=409, detail="Label already exists")
        labels.append(normalized)
        self._update_manifest(location, labels=labels)
        self._write_data_yaml(location.root, labels)
        return self.summary(dataset_id)

    def rename_label(self, dataset_id: str, label_index: int, name: str) -> DatasetSummary:
        location = self._editable_location(dataset_id)
        labels = location.labels.copy()
        if label_index < 0 or label_index >= len(labels):
            raise HTTPException(status_code=404, detail="Label not found")
        normalized = self._clean_label(name)
        if normalized in labels and labels[label_index] != normalized:
            raise HTTPException(status_code=409, detail="Label already exists")
        labels[label_index] = normalized
        self._update_manifest(location, labels=labels)
        self._write_data_yaml(location.root, labels)
        return self.summary(dataset_id)

    def delete_label(self, dataset_id: str, label_index: int, force: bool = False) -> DatasetSummary:
        location = self._editable_location(dataset_id)
        labels = location.labels.copy()
        if label_index < 0 or label_index >= len(labels):
            raise HTTPException(status_code=404, detail="Label not found")
        if len(labels) <= 1:
            raise HTTPException(status_code=409, detail="Datasets must keep at least one label")

        used = self._label_is_used(location, label_index)
        if used and not force:
            raise HTTPException(status_code=409, detail="Label is used by annotations")

        labels.pop(label_index)
        if used:
            self._rewrite_annotations_after_label_delete(location, label_index, labels)
        self._update_manifest(location, labels=labels)
        self._write_data_yaml(location.root, labels)
        return self.summary(dataset_id)

    async def upload_image(
        self,
        dataset_id: str,
        split: str,
        file: UploadFile,
        class_id: int | None = None,
        class_name: str | None = None,
    ) -> DatasetItemDetail:
        location = self._editable_location(dataset_id)
        self._validate_split(split)
        if self._is_nlp_task(location.task_type):
            items = await self._save_uploaded_text(location, split, file, class_id, class_name)
            if not items:
                raise HTTPException(status_code=400, detail="No text rows were found")
            return items[0]
        return await self._save_uploaded_image(location, split, file, class_id, class_name)

    async def upload_images(
        self,
        dataset_id: str,
        split: str,
        files: list[UploadFile],
        class_id: int | None = None,
        class_name: str | None = None,
    ) -> DatasetItemBatchUploadResponse:
        location = self._editable_location(dataset_id)
        self._validate_split(split)
        uploaded = []
        errors = []
        for file in files:
            try:
                if self._is_nlp_task(location.task_type):
                    uploaded.extend(
                        await self._save_uploaded_text(location, split, file, class_id, class_name)
                    )
                else:
                    uploaded.append(await self._save_uploaded_image(location, split, file, class_id, class_name))
            except HTTPException as exc:
                errors.append(
                    {
                        "filename": file.filename or "item",
                        "error": str(exc.detail),
                    }
                )
        return DatasetItemBatchUploadResponse(uploaded=uploaded, errors=errors)

    def delete_items(self, dataset_id: str, payload: DatasetItemDeleteRequest) -> DeleteResponse:
        location = self._editable_location(dataset_id)
        self._validate_split(payload.split)
        deleted = 0
        missing = []
        for item_id in payload.ids:
            try:
                item_path = self._item_path(location, payload.split, item_id)
            except HTTPException:
                missing.append(item_id)
                continue
            stem = item_path.stem
            item_path.unlink(missing_ok=True)
            (location.root / payload.split / "annotations" / f"{stem}.json").unlink(missing_ok=True)
            (location.root / payload.split / "labels" / f"{stem}.txt").unlink(missing_ok=True)
            deleted += 1
        return DeleteResponse(deleted=deleted, missing=missing)

    def bulk_set_item_labels(
        self, dataset_id: str, split: str, payload: DatasetItemBulkLabelUpdate
    ) -> DatasetItemBulkLabelUpdateResponse:
        location = self._editable_location(dataset_id)
        if location.task_type not in {"classification", "text_classification"}:
            raise HTTPException(status_code=409, detail="Bulk labels are only for classification datasets")
        self._validate_split(split)
        annotation = self._classification_annotation(location, payload.class_id, payload.class_name)
        updated = 0
        missing = []
        items: list[DatasetItemSummary] = []
        for item_id in payload.ids:
            try:
                item_path = self._item_path(location, split, item_id)
            except HTTPException:
                missing.append(item_id)
                continue
            self._write_annotation_json(
                location.root / split / "annotations" / f"{item_path.stem}.json",
                [annotation],
            )
            (location.root / split / "labels" / f"{item_path.stem}.txt").write_text(
                "",
                encoding="utf-8",
            )
            updated += 1
            items.append(self.item_detail(dataset_id, split, item_path.name))
        return DatasetItemBulkLabelUpdateResponse(updated=updated, missing=missing, items=items)

    async def _save_uploaded_image(
        self,
        location: DatasetLocation,
        split: str,
        file: UploadFile,
        class_id: int | None = None,
        class_name: str | None = None,
    ) -> DatasetItemDetail:
        suffix = Path(file.filename or "image.jpg").suffix.lower() or ".jpg"
        if suffix not in IMAGE_SUFFIXES:
            raise HTTPException(status_code=400, detail="Unsupported image file type")

        image_dir = location.root / split / "images"
        annotation_dir = location.root / split / "annotations"
        label_dir = location.root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        annotation_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(file.filename or "image").stem.replace(" ", "-")[:80] or "image"
        filename = f"{stem}-{uuid4().hex[:8]}{suffix}"
        path = image_dir / filename
        path.write_bytes(await file.read())
        annotations = []
        if location.task_type == "classification" and (class_id is not None or class_name):
            annotations = [self._classification_annotation(location, class_id, class_name)]
        self._write_annotation_json(annotation_dir / f"{path.stem}.json", annotations)
        (label_dir / f"{path.stem}.txt").write_text("", encoding="utf-8")
        return self.item_detail(location.id, split, filename)

    async def _save_uploaded_text(
        self,
        location: DatasetLocation,
        split: str,
        file: UploadFile,
        class_id: int | None = None,
        class_name: str | None = None,
    ) -> list[DatasetItemDetail]:
        suffix = Path(file.filename or "sample.txt").suffix.lower() or ".txt"
        if suffix not in TEXT_SUFFIXES:
            raise HTTPException(status_code=400, detail="Unsupported text file type")
        raw = (await file.read()).decode("utf-8", errors="replace")
        rows = self._text_upload_rows(raw, suffix)
        if not rows:
            raise HTTPException(status_code=400, detail="No text rows were found")
        uploaded = []
        for row_index, row in enumerate(rows, start=1):
            text = self._text_from_upload_row(row)
            if not text.strip():
                continue
            row_class_id = class_id
            row_class_name = class_name or self._string_value(row, "label", "class", "class_name")
            annotation = self._annotation_from_text_row(
                location,
                row,
                class_id=row_class_id,
                class_name=row_class_name,
            )
            stem = Path(file.filename or "sample").stem.replace(" ", "-")[:72] or "sample"
            filename = (
                f"{stem}-{uuid4().hex[:8]}.txt"
                if len(rows) == 1
                else f"{stem}-{row_index:04d}-{uuid4().hex[:8]}.txt"
            )
            text_dir = location.root / split / "texts"
            annotation_dir = location.root / split / "annotations"
            label_dir = location.root / split / "labels"
            text_dir.mkdir(parents=True, exist_ok=True)
            annotation_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            path = text_dir / filename
            path.write_text(text, encoding="utf-8")
            self._write_annotation_json(annotation_dir / f"{path.stem}.json", [annotation] if annotation else [])
            (label_dir / f"{path.stem}.txt").write_text("", encoding="utf-8")
            uploaded.append(self.item_detail(location.id, split, filename))
        return uploaded

    def list_items(
        self,
        dataset_id: str,
        split: str,
        class_filter: str | None = None,
        unlabeled: bool = False,
        limit: int = 200,
        offset: int = 0,
    ) -> list[DatasetItemSummary]:
        return self.list_items_page(dataset_id, split, class_filter, unlabeled, limit, offset).items

    def list_items_page(
        self,
        dataset_id: str,
        split: str,
        class_filter: str | None = None,
        unlabeled: bool = False,
        limit: int = 200,
        offset: int = 0,
    ) -> DatasetItemPage:
        splits = list(SPLITS) if split == "all" else [split]
        for split_name in splits:
            self._validate_split(split_name)
        location = self._location(dataset_id)
        items = []
        for split_name in splits:
            for item_path in self._item_paths(location, split_name):
                detail = self._item_from_path(location, split_name, item_path, include_annotations=True)
                if unlabeled and detail.is_labeled:
                    continue
                if class_filter and class_filter not in detail.classes:
                    continue
                items.append(
                    DatasetItemSummary(
                        id=detail.id,
                        dataset_id=detail.dataset_id,
                        split=detail.split,
                        filename=detail.filename,
                        media_type=detail.media_type,
                        image_url=detail.image_url,
                        text_url=detail.text_url,
                        text_preview=detail.text_preview,
                        width=detail.width,
                        height=detail.height,
                        annotation_count=detail.annotation_count,
                        classes=detail.classes,
                        class_id=detail.class_id,
                        label=detail.label,
                        is_labeled=detail.is_labeled,
                        annotations=detail.annotations,
                    )
                )
        return DatasetItemPage(
            items=items[offset : offset + limit],
            total=len(items),
            limit=limit,
            offset=offset,
        )

    def item_detail(self, dataset_id: str, split: str, item_id: str) -> DatasetItemDetail:
        self._validate_split(split)
        location = self._location(dataset_id)
        item_path = self._item_path(location, split, item_id)
        return self._item_from_path(location, split, item_path, include_annotations=True)

    def image_path(self, dataset_id: str, split: str, item_id: str) -> Path:
        self._validate_split(split)
        return self._image_path(self._location(dataset_id), split, item_id)

    def text_path(self, dataset_id: str, split: str, item_id: str) -> Path:
        self._validate_split(split)
        location = self._location(dataset_id)
        if not self._is_nlp_task(location.task_type):
            raise HTTPException(status_code=409, detail="Dataset item is not text")
        return self._text_path(location, split, item_id)

    def save_annotations(
        self, dataset_id: str, split: str, item_id: str, payload: DatasetAnnotationSave
    ) -> DatasetItemDetail:
        location = self._editable_location(dataset_id)
        self._validate_split(split)
        item_path = self._item_path(location, split, item_id)
        if self._is_nlp_task(location.task_type):
            raw_annotations = [
                self._normalize_annotation(annotation, location, 0, 0)
                for annotation in payload.annotations
            ]
            annotations: list[DatasetAnnotation] = [
                annotation for annotation in raw_annotations if annotation is not None
            ]
            annotation_path = location.root / split / "annotations" / f"{item_path.stem}.json"
            self._write_annotation_json(annotation_path, annotations)
            return self.item_detail(dataset_id, split, item_path.name)

        image_path = item_path
        with Image.open(image_path) as image:
            width, height = image.size

        raw_annotations = [
            self._normalize_annotation(annotation, location, width, height)
            for annotation in payload.annotations
        ]
        annotations = [annotation for annotation in raw_annotations if annotation is not None]
        annotation_path = location.root / split / "annotations" / f"{image_path.stem}.json"
        self._write_annotation_json(annotation_path, annotations)
        self._write_yolo_label(location, split, image_path.stem, annotations, width, height)
        return self.item_detail(dataset_id, split, image_path.name)

    def set_item_label(
        self, dataset_id: str, split: str, item_id: str, payload: DatasetItemLabelUpdate
    ) -> DatasetItemDetail:
        location = self._editable_location(dataset_id)
        if location.task_type not in {"classification", "text_classification"}:
            raise HTTPException(status_code=409, detail="Quick labels are only for classification datasets")
        self._validate_split(split)
        item_path = self._item_path(location, split, item_id)
        annotation = self._classification_annotation(location, payload.class_id, payload.class_name)
        self._write_annotation_json(
            location.root / split / "annotations" / f"{item_path.stem}.json",
            [annotation],
        )
        return self.item_detail(dataset_id, split, item_path.name)

    def _item_exists(self, location: DatasetLocation, split: str, item_id: str) -> bool:
        try:
            self._item_path(location, split, item_id)
        except HTTPException:
            return False
        return True

    def _split_summary(self, location: DatasetLocation, split: str) -> DatasetSplitSummary:
        items = self._item_paths(location, split)
        annotation_count = 0
        for item_path in items:
            annotation_count += len(self._annotations(location, split, item_path, 1, 1))
        text_count = len(items) if self._is_nlp_task(location.task_type) else 0
        image_count = len(items) if not self._is_nlp_task(location.task_type) else 0
        return DatasetSplitSummary(
            split=split,  # type: ignore[arg-type]
            image_count=image_count,
            text_count=text_count,
            item_count=len(items),
            annotation_count=annotation_count,
        )

    def _item_from_path(
        self, location: DatasetLocation, split: str, image_path: Path, include_annotations: bool
    ) -> DatasetItemDetail:
        if self._is_nlp_task(location.task_type):
            return self._text_item_from_path(location, split, image_path, include_annotations)
        with Image.open(image_path) as image:
            width, height = image.size
        annotations = self._annotations(location, split, image_path, width, height)
        classes = sorted({annotation.class_name for annotation in annotations})
        primary = annotations[0] if annotations else None
        return DatasetItemDetail(
            id=image_path.name,
            dataset_id=location.id,
            split=split,  # type: ignore[arg-type]
            filename=image_path.name,
            image_url=f"/api/datasets/{location.id}/items/{split}/{quote(image_path.name)}/image",
            width=width,
            height=height,
            annotation_count=len(annotations),
            classes=classes,
            class_id=primary.class_id if primary else None,
            label=primary.class_name if primary else None,
            is_labeled=primary is not None,
            annotations=annotations if include_annotations else [],
        )

    def _text_item_from_path(
        self, location: DatasetLocation, split: str, text_path: Path, include_annotations: bool
    ) -> DatasetItemDetail:
        text = text_path.read_text(encoding="utf-8", errors="replace")
        annotations = self._annotations(location, split, text_path, 0, 0)
        classes = sorted({annotation.class_name for annotation in annotations if annotation.class_name})
        primary = annotations[0] if annotations else None
        label = primary.class_name if primary and primary.kind == "classification" else None
        return DatasetItemDetail(
            id=text_path.name,
            dataset_id=location.id,
            split=split,  # type: ignore[arg-type]
            filename=text_path.name,
            media_type="text",
            image_url="",
            text_url=f"/api/datasets/{location.id}/items/{split}/{quote(text_path.name)}/text",
            text_preview=self._text_preview(text),
            width=0,
            height=0,
            annotation_count=len(annotations),
            classes=classes,
            class_id=primary.class_id if primary and primary.kind == "classification" else None,
            label=label,
            is_labeled=self._is_text_labeled(location.task_type, annotations),
            annotations=annotations if include_annotations else [],
            text_content=text if include_annotations else None,
        )

    def _image_paths(self, location: DatasetLocation, split: str) -> list[Path]:
        if location.format == "coco":
            image_dir = location.root / split
        else:
            image_dir = location.root / split / "images"
        if not image_dir.exists():
            return []
        return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    def _text_paths(self, location: DatasetLocation, split: str) -> list[Path]:
        text_dir = location.root / split / "texts"
        if not text_dir.exists():
            return []
        return sorted(path for path in text_dir.iterdir() if path.suffix.lower() == ".txt")

    def _image_path(self, location: DatasetLocation, split: str, item_id: str) -> Path:
        filename = Path(item_id).name
        for image_path in self._image_paths(location, split):
            if image_path.name == filename:
                return image_path
        raise HTTPException(status_code=404, detail="Dataset image not found")

    def _text_path(self, location: DatasetLocation, split: str, item_id: str) -> Path:
        filename = Path(item_id).name
        for text_path in self._text_paths(location, split):
            if text_path.name == filename:
                return text_path
        raise HTTPException(status_code=404, detail="Dataset text item not found")

    def _item_paths(self, location: DatasetLocation, split: str) -> list[Path]:
        return self._text_paths(location, split) if self._is_nlp_task(location.task_type) else self._image_paths(location, split)

    def _item_path(self, location: DatasetLocation, split: str, item_id: str) -> Path:
        return self._text_path(location, split, item_id) if self._is_nlp_task(location.task_type) else self._image_path(location, split, item_id)

    def _label_is_used(self, location: DatasetLocation, label_index: int) -> bool:
        for split in SPLITS:
            for item_path in self._item_paths(location, split):
                width, height = 1, 1
                if not self._is_nlp_task(location.task_type):
                    with Image.open(item_path) as image:
                        width, height = image.size
                for annotation in self._annotations(location, split, item_path, width, height):
                    if annotation.class_id == label_index:
                        return True
        return False

    def _rewrite_annotations_after_label_delete(
        self, location: DatasetLocation, deleted_index: int, labels: list[str]
    ) -> None:
        for split in SPLITS:
            for image_path in self._item_paths(location, split):
                width, height = 1, 1
                if not self._is_nlp_task(location.task_type):
                    with Image.open(image_path) as image:
                        width, height = image.size
                annotations = []
                for annotation in self._annotations(location, split, image_path, width, height):
                    if annotation.class_id == deleted_index:
                        continue
                    if annotation.class_id > deleted_index:
                        annotation.class_id -= 1
                    annotation.class_name = (
                        labels[annotation.class_id]
                        if annotation.class_id < len(labels)
                        else annotation.class_name
                    )
                    annotations.append(annotation)
                self._write_annotation_json(
                    location.root / split / "annotations" / f"{image_path.stem}.json",
                    annotations,
                )
                if not self._is_nlp_task(location.task_type):
                    self._write_yolo_label(location, split, image_path.stem, annotations, width, height)

    def _text_preview(self, text: str, limit: int = 180) -> str:
        compact = " ".join(text.split())
        return compact[:limit] + ("..." if len(compact) > limit else "")

    def _is_text_labeled(self, task_type: str, annotations: list[DatasetAnnotation]) -> bool:
        if task_type == "text_classification":
            return any(annotation.kind == "classification" for annotation in annotations)
        if task_type == "summarization":
            return any(annotation.kind == "summary" and (annotation.text or annotation.answer) for annotation in annotations)
        if task_type == "question_answering":
            return any(annotation.kind == "qa" and annotation.question and annotation.answer for annotation in annotations)
        return False

    if TYPE_CHECKING:
        # Provided by sibling mixins on the composed DatasetService facade.
        def _location(self, dataset_id: str) -> DatasetLocation:
            raise NotImplementedError

        def _editable_location(self, dataset_id: str) -> DatasetLocation:
            raise NotImplementedError

        def _is_nlp_task(self, task_type: str) -> bool:
            raise NotImplementedError

        def _clean_label(self, label: str) -> str:
            raise NotImplementedError

        def _validate_split(self, split: str) -> None:
            raise NotImplementedError

        def summary(self, dataset_id: str) -> DatasetSummary:
            raise NotImplementedError

        def _annotations(
            self, location: DatasetLocation, split: str, image_path: Path, width: int, height: int
        ) -> list[DatasetAnnotation]:
            raise NotImplementedError

        def _write_annotation_json(self, path: Path, annotations: list[DatasetAnnotation]) -> None:
            raise NotImplementedError

        def _classification_annotation(
            self, location: DatasetLocation, class_id: int | None, class_name: str | None
        ) -> DatasetAnnotation:
            raise NotImplementedError

        def _normalize_annotation(
            self, annotation: DatasetAnnotation, location: DatasetLocation, width: int, height: int
        ) -> DatasetAnnotation | None:
            raise NotImplementedError

        def _write_yolo_label(
            self,
            location: DatasetLocation,
            split: str,
            stem: str,
            annotations: list[DatasetAnnotation],
            width: int,
            height: int,
        ) -> None:
            raise NotImplementedError

        def _annotation_from_text_row(
            self,
            location: DatasetLocation,
            row: dict,
            *,
            class_id: int | None = None,
            class_name: str | None = None,
        ) -> DatasetAnnotation | None:
            raise NotImplementedError

        def _string_value(self, row: dict, *keys: str) -> str:
            raise NotImplementedError

        def _text_upload_rows(self, raw: str, suffix: str) -> list[dict]:
            raise NotImplementedError

        def _text_from_upload_row(self, row: dict) -> str:
            raise NotImplementedError

        def _update_manifest(self, location: DatasetLocation, **updates) -> None:
            raise NotImplementedError

        def _write_data_yaml(self, root: Path, labels: list[str]) -> None:
            raise NotImplementedError

