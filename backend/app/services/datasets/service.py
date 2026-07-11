import ast
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from statistics import mean
from urllib.parse import quote
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image

from app.core.config import Settings
from app.core.defaults import DEFAULT_LABELS, DEFAULT_PROJECT_ID, DEFAULT_TASK_TYPE
from app.core.storage import Storage
from app.services.datasets.constants import (
    ALLOWED_PREPROCESS_TRANSFORMS,
    IMAGE_SUFFIXES,
    IMAGE_TASK_TYPES,
    PREPROCESS_PRESETS,
    SPLITS,
    TRAINING_SPLITS,
)
from app.services.datasets.types import DatasetLocation
from app.schemas import (
    Box,
    DatasetAnnotation,
    DatasetAnnotationSave,
    DatasetCreate,
    DatasetEdaSummary,
    DatasetImportRequest,
    DatasetItemBatchUploadResponse,
    DatasetItemBulkLabelUpdate,
    DatasetItemBulkLabelUpdateResponse,
    DatasetItemDeleteRequest,
    DatasetItemDetail,
    DatasetItemLabelUpdate,
    DatasetItemMoveRequest,
    DatasetItemMoveResponse,
    DatasetItemPage,
    DatasetItemSummary,
    DatasetProcessRequest,
    DatasetProcessResponse,
    DatasetPreprocessConfig,
    DatasetPreprocessPreview,
    DatasetSplitConfig,
    DatasetSplitSummary,
    DatasetSummary,
    DatasetUpdate,
    DatasetVersionCreate,
    DatasetVersionSummary,
    DeleteResponse,
)

class DatasetService:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage

    def list_datasets(self, project_id: str | None = None) -> list[DatasetSummary]:
        summaries = [self.summary(location.id) for location in self._locations()]
        if project_id:
            summaries = [dataset for dataset in summaries if dataset.project_id == project_id]
        return summaries

    def summary(self, dataset_id: str) -> DatasetSummary:
        location = self._location(dataset_id)
        return DatasetSummary(
            id=location.id,
            project_id=location.project_id,
            name=location.name,
            task_type=location.task_type,  # type: ignore[arg-type]
            format=location.format,  # type: ignore[arg-type]
            source=location.source,  # type: ignore[arg-type]
            editable=location.editable,
            path=str(location.root),
            labels=location.labels,
            classes=location.labels,
            splits={split: self._split_summary(location, split) for split in SPLITS},
            metadata=location.metadata,
        )

    def create_dataset(self, payload: DatasetCreate) -> DatasetSummary:
        if payload.task_type not in IMAGE_TASK_TYPES:
            raise HTTPException(status_code=400, detail="Only image task datasets are supported")
        labels = self._normalize_labels(payload.labels)
        dataset_id = self._new_dataset_id(payload.name)
        root = self.storage.datasets / dataset_id
        self._create_layout(root, payload.format, payload.task_type, labels)
        self._write_manifest(
            root,
            dataset_id=dataset_id,
            name=payload.name,
            format_name=payload.format,
            project_id=payload.project_id,
            task_type=payload.task_type,
            labels=labels,
            metadata={
                "created_from": "empty",
                "split_config": DatasetSplitConfig().model_dump(mode="json"),
            },
        )
        return self.summary(dataset_id)

    def update_dataset(self, dataset_id: str, payload: DatasetUpdate) -> DatasetSummary:
        location = self._editable_location(dataset_id)
        metadata = dict(location.metadata or {})
        updates: dict = {}
        if payload.name is not None:
            updates["name"] = payload.name
        if payload.metadata is not None:
            metadata.update(payload.metadata)
        if payload.preprocess is not None:
            metadata["preprocess"] = self._normalize_preprocess_config(payload.preprocess).model_dump(
                mode="json"
            )
        if payload.metadata is not None or payload.preprocess is not None:
            updates["metadata"] = metadata
        if updates:
            self._update_manifest(location, **updates)
        return self.summary(dataset_id)

    def import_dataset(self, payload: DatasetImportRequest) -> DatasetSummary:
        source = Path(payload.path).expanduser().resolve()
        if not source.exists():
            raise HTTPException(status_code=404, detail=f"Dataset path not found: {source}")
        if not self._is_yolo_root(source):
            raise HTTPException(status_code=400, detail="Only YOLO dataset imports are supported")

        labels = self._normalize_labels(payload.labels or self._labels_from_yolo_yaml(source) or DEFAULT_LABELS)
        dataset_id = self._new_dataset_id(payload.name or source.name)
        root = self.storage.datasets / dataset_id
        shutil.copytree(source, root)
        self._create_layout(root, payload.format, payload.task_type, labels)
        self._write_manifest(
            root,
            dataset_id=dataset_id,
            name=payload.name or source.name,
            format_name=payload.format,
            project_id=payload.project_id,
            task_type=payload.task_type,
            labels=labels,
            metadata={"imported_from": str(source)},
        )
        return self.summary(dataset_id)

    def clone_dataset(
        self, dataset_id: str, name: str | None = None, project_id: str | None = None
    ) -> DatasetSummary:
        source = self._location(dataset_id)
        if source.format not in {"yolo", "image_folder", "image_manifest"}:
            raise HTTPException(status_code=400, detail="Only image datasets can be cloned")

        new_name = name or f"{source.name} Copy"
        new_id = self._new_dataset_id(new_name)
        root = self.storage.datasets / new_id
        shutil.copytree(source.root, root)
        self._create_layout(root, source.format, source.task_type, source.labels)
        self._write_manifest(
            root,
            dataset_id=new_id,
            name=new_name,
            format_name=source.format,
            project_id=project_id or source.project_id,
            task_type=source.task_type,
            labels=source.labels,
            metadata={"cloned_from": source.id, "source_path": str(source.root)},
        )
        return self.summary(new_id)

    def delete_dataset(self, dataset_id: str) -> None:
        location = self._location(dataset_id)
        if not location.editable:
            raise HTTPException(status_code=409, detail="Reference datasets are read-only")
        self.storage.delete_owned_path(location.root)

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
                uploaded.append(await self._save_uploaded_image(location, split, file, class_id, class_name))
            except HTTPException as exc:
                errors.append(
                    {
                        "filename": file.filename or "image",
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
                image_path = self._image_path(location, payload.split, item_id)
            except HTTPException:
                missing.append(item_id)
                continue
            stem = image_path.stem
            image_path.unlink(missing_ok=True)
            (location.root / payload.split / "annotations" / f"{stem}.json").unlink(missing_ok=True)
            (location.root / payload.split / "labels" / f"{stem}.txt").unlink(missing_ok=True)
            deleted += 1
        return DeleteResponse(deleted=deleted, missing=missing)

    def move_items(self, dataset_id: str, payload: DatasetItemMoveRequest) -> DatasetItemMoveResponse:
        location = self._editable_location(dataset_id)
        self._validate_split(payload.source_split)
        self._validate_split(payload.target_split)
        if payload.source_split == payload.target_split:
            return DatasetItemMoveResponse(
                moved=0,
                items=[
                    self.item_detail(dataset_id, payload.source_split, item_id)
                    for item_id in payload.ids
                    if self._item_exists(location, payload.source_split, item_id)
                ],
            )

        moved = 0
        missing = []
        items = []
        for item_id in payload.ids:
            try:
                new_name = self._move_item(location, payload.source_split, payload.target_split, item_id)
            except HTTPException:
                missing.append(item_id)
                continue
            moved += 1
            items.append(self.item_detail(dataset_id, payload.target_split, new_name))
        return DatasetItemMoveResponse(moved=moved, missing=missing, items=items)

    def process_dataset(self, dataset_id: str, payload: DatasetProcessRequest) -> DatasetProcessResponse:
        location = self._editable_location(dataset_id)
        split_config = self._normalize_split_config(payload.split)
        metadata = dict(location.metadata or {})
        if payload.preprocess is not None:
            metadata["preprocess"] = self._normalize_preprocess_config(payload.preprocess).model_dump(
                mode="json"
            )
        metadata["split_config"] = split_config.model_dump(mode="json")
        metadata["processed_at"] = datetime.utcnow().isoformat()
        self._update_manifest(location, metadata=metadata)
        location = self._location(dataset_id)

        source_splits = list(SPLITS if split_config.resplit_all else ("unassigned",))
        candidates = []
        for source_split in source_splits:
            for image_path in self._image_paths(location, source_split):
                detail = self._item_from_path(location, source_split, image_path, include_annotations=True)
                candidates.append((source_split, detail))

        grouped: dict[str, list[tuple[str, DatasetItemDetail]]] = {}
        if split_config.stratify:
            for source_split, item in candidates:
                label = item.label or "__unlabeled__"
                grouped.setdefault(label, []).append((source_split, item))
        else:
            grouped["__all__"] = candidates

        rng = random.Random(split_config.seed)
        moved = {split: 0 for split in TRAINING_SPLITS}
        for rows in grouped.values():
            rng.shuffle(rows)
            targets = self._targets_for_count(len(rows), split_config)
            for (source_split, item), target_split in zip(rows, targets):
                if source_split == target_split:
                    continue
                new_name = self._move_item(location, source_split, target_split, item.id)
                moved[target_split] += 1
                if new_name != item.id:
                    item.id = new_name

        return DatasetProcessResponse(
            dataset=self.summary(dataset_id),
            moved=moved,
            split_config=split_config,
        )

    def bulk_set_item_labels(
        self, dataset_id: str, split: str, payload: DatasetItemBulkLabelUpdate
    ) -> DatasetItemBulkLabelUpdateResponse:
        location = self._editable_location(dataset_id)
        if location.task_type != "classification":
            raise HTTPException(status_code=409, detail="Bulk labels are only for classification datasets")
        self._validate_split(split)
        annotation = self._classification_annotation(location, payload.class_id, payload.class_name)
        updated = 0
        missing = []
        items = []
        for item_id in payload.ids:
            try:
                image_path = self._image_path(location, split, item_id)
            except HTTPException:
                missing.append(item_id)
                continue
            self._write_annotation_json(
                location.root / split / "annotations" / f"{image_path.stem}.json",
                [annotation],
            )
            (location.root / split / "labels" / f"{image_path.stem}.txt").write_text(
                "",
                encoding="utf-8",
            )
            updated += 1
            items.append(self.item_detail(dataset_id, split, image_path.name))
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
            for image_path in self._image_paths(location, split_name):
                detail = self._item_from_path(location, split_name, image_path, include_annotations=True)
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
                        image_url=detail.image_url,
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
        image_path = self._image_path(location, split, item_id)
        return self._item_from_path(location, split, image_path, include_annotations=True)

    def image_path(self, dataset_id: str, split: str, item_id: str) -> Path:
        self._validate_split(split)
        return self._image_path(self._location(dataset_id), split, item_id)

    def save_annotations(
        self, dataset_id: str, split: str, item_id: str, payload: DatasetAnnotationSave
    ) -> DatasetItemDetail:
        location = self._editable_location(dataset_id)
        self._validate_split(split)
        image_path = self._image_path(location, split, item_id)
        with Image.open(image_path) as image:
            width, height = image.size

        annotations = [
            self._normalize_annotation(annotation, location, width, height)
            for annotation in payload.annotations
        ]
        annotations = [annotation for annotation in annotations if annotation is not None]
        annotation_path = location.root / split / "annotations" / f"{image_path.stem}.json"
        self._write_annotation_json(annotation_path, annotations)
        self._write_yolo_label(location, split, image_path.stem, annotations, width, height)
        return self.item_detail(dataset_id, split, image_path.name)

    def set_item_label(
        self, dataset_id: str, split: str, item_id: str, payload: DatasetItemLabelUpdate
    ) -> DatasetItemDetail:
        location = self._editable_location(dataset_id)
        if location.task_type != "classification":
            raise HTTPException(status_code=409, detail="Quick labels are only for classification datasets")
        self._validate_split(split)
        image_path = self._image_path(location, split, item_id)
        annotation = self._classification_annotation(location, payload.class_id, payload.class_name)
        self._write_annotation_json(
            location.root / split / "annotations" / f"{image_path.stem}.json",
            [annotation],
        )
        return self.item_detail(dataset_id, split, image_path.name)

    def preprocess_preview(
        self,
        dataset_id: str,
        split: str,
        item_id: str,
        config: DatasetPreprocessConfig | None = None,
    ) -> DatasetPreprocessPreview:
        self._validate_split(split)
        location = self._location(dataset_id)
        image_path = self._image_path(location, split, item_id)
        preprocess = self._normalize_preprocess_config(
            config or (location.metadata or {}).get("preprocess") or DatasetPreprocessConfig()
        )
        image, _annotations = self._preprocessed_image_and_annotations(location, split, image_path, preprocess)
        preview_dir = self.storage.previews / dataset_id
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{image_path.stem}-{uuid4().hex[:8]}.jpg"
        image.save(preview_path, quality=92)
        return DatasetPreprocessPreview(
            dataset_id=dataset_id,
            split=split,  # type: ignore[arg-type]
            item_id=image_path.name,
            image_url=self.storage.media_url(preview_path),
            config=preprocess,
        )

    def list_versions(self, dataset_id: str) -> list[DatasetVersionSummary]:
        self._location(dataset_id)
        versions_root = self.storage.dataset_versions / dataset_id
        if not versions_root.exists():
            return []
        versions = []
        for manifest_path in sorted(versions_root.glob("*/manifest.json")):
            try:
                versions.append(self._version_summary_from_manifest(dataset_id, manifest_path))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return sorted(versions, key=lambda version: version.created_at, reverse=True)

    def create_version(
        self, dataset_id: str, payload: DatasetVersionCreate
    ) -> DatasetVersionSummary:
        location = self._location(dataset_id)
        splits = self._normalize_splits(payload.splits)
        augmentation_splits = self._normalize_splits(payload.augmentation_splits)
        config = self._normalize_preprocess_config(
            payload.config or (location.metadata or {}).get("preprocess") or DatasetPreprocessConfig()
        )
        version_id = uuid4().hex
        version_root = self.storage.dataset_versions / dataset_id / version_id
        version_name = payload.name or f"{location.name} version {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        self._create_layout(version_root, location.format, location.task_type, location.labels)

        image_count = 0
        generated_count = 0
        prepared_location = DatasetLocation(
            id=location.id,
            project_id=location.project_id,
            name=version_name,
            task_type=location.task_type,
            format=location.format,
            source=location.source,
            root=version_root,
            editable=True,
            labels=location.labels,
            metadata=location.metadata,
        )
        for split in splits:
            for image_path in self._image_paths(location, split):
                image, annotations = self._preprocessed_image_and_annotations(
                    location, split, image_path, config
                )
                output_image = version_root / split / "images" / image_path.name
                output_image.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_image)
                self._write_annotation_json(
                    version_root / split / "annotations" / f"{image_path.stem}.json",
                    annotations,
                )
                self._write_yolo_label(
                    prepared_location,
                    split,
                    image_path.stem,
                    annotations,
                    image.width,
                    image.height,
                )
                image_count += 1

                if (
                    not config.enabled
                    or config.augmentation_mode != "materialize"
                    or split not in augmentation_splits
                ):
                    continue
                for copy_index in range(config.copies_per_image):
                    aug_image, aug_annotations = self._preprocessed_image_and_annotations(
                        location, split, image_path, config
                    )
                    aug_stem = f"{image_path.stem}-aug-{copy_index + 1:02d}"
                    aug_path = version_root / split / "images" / f"{aug_stem}{image_path.suffix}"
                    aug_image.save(aug_path)
                    self._write_annotation_json(
                        version_root / split / "annotations" / f"{aug_stem}.json",
                        aug_annotations,
                    )
                    self._write_yolo_label(
                        prepared_location,
                        split,
                        aug_stem,
                        aug_annotations,
                        aug_image.width,
                        aug_image.height,
                    )
                    image_count += 1
                    generated_count += 1

        created_at = datetime.utcnow()
        metadata = {
            "dataset_id": dataset_id,
            "version_id": version_id,
            "version_name": version_name,
            "created_at": created_at.isoformat(),
            "preprocess": config.model_dump(mode="json"),
            "selected_splits": splits,
            "augmentation_splits": augmentation_splits,
            "image_count": image_count,
            "generated_count": generated_count,
            "source_path": str(location.root),
        }
        self._write_manifest(
            version_root,
            dataset_id=version_id,
            name=version_name,
            format_name=location.format,
            project_id=location.project_id,
            task_type=location.task_type,
            labels=location.labels,
            metadata=metadata,
        )
        if location.format == "yolo":
            self._write_data_yaml(version_root, location.labels)
        return self._version_summary_from_manifest(dataset_id, version_root / "manifest.json")

    def eda_summary(self, dataset_id: str, split: str) -> DatasetEdaSummary:
        splits = list(SPLITS) if split == "all" else [split]
        for split_name in splits:
            self._validate_split(split_name)
        location = self._location(dataset_id)
        class_counts = {label: 0 for label in location.labels}
        widths = []
        heights = []
        aspect_ratios = []
        image_count = 0
        annotation_count = 0
        unlabeled_count = 0
        missing_annotation_count = 0
        for split_name in splits:
            for image_path in self._image_paths(location, split_name):
                with Image.open(image_path) as image:
                    width, height = image.size
                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height if height else 0)
                image_count += 1
                annotations = self._annotations(location, split_name, image_path, width, height)
                if not annotations:
                    unlabeled_count += 1
                    missing_annotation_count += 1
                annotation_count += len(annotations)
                seen_labels = {annotation.class_name for annotation in annotations}
                for label in seen_labels:
                    class_counts[label] = class_counts.get(label, 0) + 1

        nonzero_counts = [count for count in class_counts.values() if count > 0]
        warnings = []
        if unlabeled_count:
            warnings.append(f"{unlabeled_count} images do not have labels or annotations")
        if len(nonzero_counts) >= 2 and max(nonzero_counts) / max(min(nonzero_counts), 1) >= 3:
            warnings.append("Class distribution is imbalanced")
        if image_count == 0:
            warnings.append("No images found in this split")

        return DatasetEdaSummary(
            dataset_id=dataset_id,
            split=split,
            split_counts={name: len(self._image_paths(location, name)) for name in SPLITS},
            class_counts=class_counts,
            unlabeled_count=unlabeled_count,
            missing_annotation_count=missing_annotation_count,
            image_count=image_count,
            annotation_count=annotation_count,
            image_size={
                "min_width": min(widths) if widths else None,
                "max_width": max(widths) if widths else None,
                "mean_width": round(mean(widths), 2) if widths else None,
                "min_height": min(heights) if heights else None,
                "max_height": max(heights) if heights else None,
                "mean_height": round(mean(heights), 2) if heights else None,
            },
            aspect_ratio={
                "min": round(min(aspect_ratios), 3) if aspect_ratios else None,
                "max": round(max(aspect_ratios), 3) if aspect_ratios else None,
                "mean": round(mean(aspect_ratios), 3) if aspect_ratios else None,
            },
            warnings=warnings,
        )

    def dataset_root(self, dataset_id: str) -> Path:
        return self._location(dataset_id).root

    def training_root(self, dataset_id: str) -> Path:
        location = self._location(dataset_id)
        if location.format != "yolo":
            raise ValueError("YOLO training requires a YOLO dataset")
        if not (location.root / "train" / "images").exists() or not (
            location.root / "valid" / "images"
        ).exists():
            raise FileNotFoundError("YOLO dataset must include train and valid image splits")
        self._write_data_yaml(location.root, location.labels)
        return location.root

    def prepared_training_root(self, dataset_id: str, output_root: Path) -> Path:
        location = self._location(dataset_id)
        preprocess = self._normalize_preprocess_config(
            (location.metadata or {}).get("preprocess") or DatasetPreprocessConfig()
        )
        if not preprocess.enabled:
            return self.training_root(dataset_id) if location.format == "yolo" else location.root

        if output_root.exists():
            shutil.rmtree(output_root)
        self._create_layout(output_root, location.format, location.task_type, location.labels)
        self._write_manifest(
            output_root,
            dataset_id=location.id,
            name=location.name,
            format_name=location.format,
            project_id=location.project_id,
            task_type=location.task_type,
            labels=location.labels,
            metadata={**location.metadata, "prepared_from": str(location.root)},
        )

        prepared_location = DatasetLocation(
            id=location.id,
            project_id=location.project_id,
            name=location.name,
            task_type=location.task_type,
            format=location.format,
            source=location.source,
            root=output_root,
            editable=True,
            labels=location.labels,
            metadata=location.metadata,
        )
        for split in TRAINING_SPLITS:
            for image_path in self._image_paths(location, split):
                image, annotations = self._preprocessed_image_and_annotations(
                    location, split, image_path, preprocess
                )
                output_image = output_root / split / "images" / image_path.name
                output_image.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_image)
                self._write_annotation_json(
                    output_root / split / "annotations" / f"{image_path.stem}.json",
                    annotations,
                )
                self._write_yolo_label(
                    prepared_location,
                    split,
                    image_path.stem,
                    annotations,
                    image.width,
                    image.height,
                )
                if preprocess.enabled and preprocess.augmentation_mode == "materialize" and split == "train":
                    for copy_index in range(preprocess.copies_per_image):
                        aug_image, aug_annotations = self._preprocessed_image_and_annotations(
                            location, split, image_path, preprocess
                        )
                        aug_stem = f"{image_path.stem}-aug-{copy_index + 1:02d}"
                        output_aug = output_root / split / "images" / f"{aug_stem}{image_path.suffix}"
                        aug_image.save(output_aug)
                        self._write_annotation_json(
                            output_root / split / "annotations" / f"{aug_stem}.json",
                            aug_annotations,
                        )
                        self._write_yolo_label(
                            prepared_location,
                            split,
                            aug_stem,
                            aug_annotations,
                            aug_image.width,
                            aug_image.height,
                        )
        if location.format == "yolo":
            self._write_data_yaml(output_root, location.labels)
        return output_root

    def yolo_split(self, dataset_id: str, split: str) -> Path:
        self._validate_split(split)
        if split == "unassigned":
            raise ValueError("Unassigned split is not available for YOLO evaluation")
        location = self._location(dataset_id)
        if location.format != "yolo":
            raise ValueError("YOLO split requested for a non-YOLO dataset")
        return location.root / split

    def split_root(self, dataset_id: str, split: str) -> Path:
        self._validate_split(split)
        return self._location(dataset_id).root / split

    def labels_for_dataset_key(self, dataset_key: str) -> list[str]:
        if dataset_key == "yolo_test":
            return self._labels_from_yolo_yaml(self.settings.datasets_path / "dental dataset_yolov11_format") or DEFAULT_LABELS
        if dataset_key == "coco_test":
            return self._labels_from_coco(self.settings.datasets_path / "dental dataset_coco_format" / "test") or DEFAULT_LABELS
        if dataset_key.startswith("dataset:"):
            _, dataset_id, _split = dataset_key.split(":", 2)
            return self._location(dataset_id).labels
        return DEFAULT_LABELS

    def task_for_dataset_key(self, dataset_key: str) -> str:
        if dataset_key in {"yolo_test", "coco_test"}:
            return "segmentation"
        if dataset_key.startswith("dataset:"):
            _, dataset_id, _split = dataset_key.split(":", 2)
            return self._location(dataset_id).task_type
        return DEFAULT_TASK_TYPE

    def _item_exists(self, location: DatasetLocation, split: str, item_id: str) -> bool:
        try:
            self._image_path(location, split, item_id)
        except HTTPException:
            return False
        return True

    def _move_item(
        self,
        location: DatasetLocation,
        source_split: str,
        target_split: str,
        item_id: str,
    ) -> str:
        image_path = self._image_path(location, source_split, item_id)
        source_stem = image_path.stem
        target_image_dir = location.root / target_split / "images"
        target_annotation_dir = location.root / target_split / "annotations"
        target_label_dir = location.root / target_split / "labels"
        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_annotation_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

        target_name = image_path.name
        if (target_image_dir / target_name).exists():
            target_name = f"{image_path.stem}-{uuid4().hex[:8]}{image_path.suffix}"
        target_stem = Path(target_name).stem

        shutil.move(str(image_path), target_image_dir / target_name)
        source_annotation = location.root / source_split / "annotations" / f"{source_stem}.json"
        if source_annotation.exists():
            shutil.move(str(source_annotation), target_annotation_dir / f"{target_stem}.json")
        source_label = location.root / source_split / "labels" / f"{source_stem}.txt"
        if source_label.exists():
            shutil.move(str(source_label), target_label_dir / f"{target_stem}.txt")
        return target_name

    def _normalize_split_config(self, config: DatasetSplitConfig | dict) -> DatasetSplitConfig:
        if isinstance(config, dict):
            config = DatasetSplitConfig.model_validate(config)
        total = config.train + config.valid + config.test
        if total <= 0:
            raise HTTPException(status_code=400, detail="Dataset split proportions must be greater than zero")
        return DatasetSplitConfig(
            train=config.train / total,
            valid=config.valid / total,
            test=config.test / total,
            seed=config.seed,
            stratify=config.stratify,
            resplit_all=config.resplit_all,
        )

    def _targets_for_count(self, count: int, config: DatasetSplitConfig) -> list[str]:
        if count <= 0:
            return []
        ratios = {"train": config.train, "valid": config.valid, "test": config.test}
        counts = {split: int(count * ratio) for split, ratio in ratios.items()}
        remaining = count - sum(counts.values())
        remainders = sorted(
            ((count * ratio) - counts[split], split) for split, ratio in ratios.items()
        )
        for _remainder, split in reversed(remainders[-remaining:] if remaining else []):
            counts[split] += 1
        targets = []
        for split in TRAINING_SPLITS:
            targets.extend([split] * counts[split])
        return targets[:count]

    def _locations(self) -> list[DatasetLocation]:
        locations = [
            DatasetLocation(
                id="reference_yolo",
                project_id=DEFAULT_PROJECT_ID,
                name="YOLO Reference Dataset",
                task_type="segmentation",
                format="yolo",
                source="reference",
                root=self.settings.datasets_path / "dental dataset_yolov11_format",
                editable=False,
                labels=self._labels_from_yolo_yaml(
                    self.settings.datasets_path / "dental dataset_yolov11_format"
                )
                or DEFAULT_LABELS,
                metadata=self._reference_metadata("dental dataset_yolov11_format"),
            ),
            DatasetLocation(
                id="reference_coco",
                project_id=DEFAULT_PROJECT_ID,
                name="COCO Reference Dataset",
                task_type="segmentation",
                format="coco",
                source="reference",
                root=self.settings.datasets_path / "dental dataset_coco_format",
                editable=False,
                labels=self._labels_from_coco(
                    self.settings.datasets_path / "dental dataset_coco_format" / "test"
                )
                or DEFAULT_LABELS,
                metadata=self._reference_metadata("dental dataset_coco_format"),
            ),
        ]
        if self.storage.datasets.exists():
            for manifest_path in sorted(self.storage.datasets.glob("*/manifest.json")):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, KeyError):
                    continue
                labels = self._normalize_labels(manifest.get("labels") or manifest.get("classes") or DEFAULT_LABELS)
                locations.append(
                    DatasetLocation(
                        id=manifest["id"],
                        project_id=manifest.get("project_id", DEFAULT_PROJECT_ID),
                        name=manifest["name"],
                        task_type=manifest.get("task_type", DEFAULT_TASK_TYPE),
                        format=manifest.get("format", "yolo"),
                        source="editable",
                        root=manifest_path.parent,
                        editable=True,
                        labels=labels,
                        metadata=manifest.get("metadata", {}),
                    )
                )
        return locations

    def _location(self, dataset_id: str) -> DatasetLocation:
        for location in self._locations():
            if location.id == dataset_id:
                return location
        raise HTTPException(status_code=404, detail="Dataset not found")

    def _editable_location(self, dataset_id: str) -> DatasetLocation:
        location = self._location(dataset_id)
        if not location.editable:
            raise HTTPException(status_code=409, detail="Reference datasets are read-only")
        return location

    def _split_summary(self, location: DatasetLocation, split: str) -> DatasetSplitSummary:
        images = self._image_paths(location, split)
        annotation_count = 0
        for image_path in images:
            annotation_count += len(self._annotations(location, split, image_path, 1, 1))
        return DatasetSplitSummary(
            split=split,  # type: ignore[arg-type]
            image_count=len(images),
            annotation_count=annotation_count,
        )

    def _item_from_path(
        self, location: DatasetLocation, split: str, image_path: Path, include_annotations: bool
    ) -> DatasetItemDetail:
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

    def _image_paths(self, location: DatasetLocation, split: str) -> list[Path]:
        if location.format == "coco":
            image_dir = location.root / split
        else:
            image_dir = location.root / split / "images"
        if not image_dir.exists():
            return []
        return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    def _image_path(self, location: DatasetLocation, split: str, item_id: str) -> Path:
        filename = Path(item_id).name
        for image_path in self._image_paths(location, split):
            if image_path.name == filename:
                return image_path
        raise HTTPException(status_code=404, detail="Dataset image not found")

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

    def _normalize_preprocess_config(
        self, config: DatasetPreprocessConfig | dict | None
    ) -> DatasetPreprocessConfig:
        if config is None:
            config = DatasetPreprocessConfig()
        if isinstance(config, dict):
            config = DatasetPreprocessConfig.model_validate(config)
        if not config.enabled:
            return DatasetPreprocessConfig(
                enabled=False,
                preset=config.preset,
                resize_width=None,
                resize_height=None,
                normalize=False,
                transforms=[],
                augmentation_mode=config.augmentation_mode,
                copies_per_image=config.copies_per_image,
            )
        transforms = list(config.transforms or [])
        if config.preset != "none" and not transforms:
            transforms = PREPROCESS_PRESETS[config.preset].copy()
        unknown = sorted(set(transforms) - ALLOWED_PREPROCESS_TRANSFORMS)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported preprocess transforms: {', '.join(unknown)}",
            )
        return DatasetPreprocessConfig(
            enabled=config.enabled,
            preset=config.preset,
            resize_width=config.resize_width,
            resize_height=config.resize_height,
            normalize=config.normalize,
            transforms=transforms,
            augmentation_mode=config.augmentation_mode,
            copies_per_image=config.copies_per_image,
        )

    def _preprocessed_image_and_annotations(
        self,
        location: DatasetLocation,
        split: str,
        image_path: Path,
        config: DatasetPreprocessConfig,
    ) -> tuple[Image.Image, list[DatasetAnnotation]]:
        with Image.open(image_path).convert("RGB") as source_image:
            width, height = source_image.size
            annotations = self._annotations(location, split, image_path, width, height)
            transform = self._albumentations_transform(config)
            image_array = transform(image=self._image_to_array(source_image))["image"]
            image = Image.fromarray(self._array_to_uint8(image_array))
        return image, self._transform_annotations(annotations, config, width, height, image.width, image.height)

    def _albumentations_transform(self, config: DatasetPreprocessConfig):
        import os  # noqa: PLC0415

        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        import albumentations as A  # noqa: PLC0415

        transforms = []
        if config.resize_width and config.resize_height:
            transforms.append(A.Resize(height=config.resize_height, width=config.resize_width, p=1.0))
        for name in config.transforms:
            if name == "horizontal_flip":
                transforms.append(A.HorizontalFlip(p=1.0))
            elif name == "vertical_flip":
                transforms.append(A.VerticalFlip(p=1.0))
            elif name == "brightness_contrast":
                transforms.append(A.RandomBrightnessContrast(p=1.0))
            elif name == "gaussian_blur":
                transforms.append(A.GaussianBlur(p=1.0))
        # Keras and YOLO runners already perform numeric normalization, so persisted
        # previews/prepared copies stay uint8 images.
        return A.Compose(transforms)

    def _image_to_array(self, image: Image.Image):
        import numpy as np  # noqa: PLC0415

        return np.asarray(image)

    def _array_to_uint8(self, array):
        import numpy as np  # noqa: PLC0415

        if array.dtype == np.uint8:
            return array
        return np.clip(array, 0, 255).astype(np.uint8)

    def _transform_annotations(
        self,
        annotations: list[DatasetAnnotation],
        config: DatasetPreprocessConfig,
        original_width: int,
        original_height: int,
        output_width: int,
        output_height: int,
    ) -> list[DatasetAnnotation]:
        x_scale = output_width / max(original_width, 1)
        y_scale = output_height / max(original_height, 1)
        hflip = "horizontal_flip" in config.transforms
        vflip = "vertical_flip" in config.transforms
        rows = []
        for annotation in annotations:
            bbox = annotation.bbox
            polygon = [[point[0] * x_scale, point[1] * y_scale] for point in annotation.polygon]
            if bbox is not None:
                bbox = Box(
                    x=bbox.x * x_scale,
                    y=bbox.y * y_scale,
                    width=bbox.width * x_scale,
                    height=bbox.height * y_scale,
                )
            if hflip:
                polygon = [[output_width - point[0], point[1]] for point in polygon]
                if bbox is not None:
                    bbox = Box(
                        x=max(0.0, output_width - bbox.x - bbox.width),
                        y=bbox.y,
                        width=bbox.width,
                        height=bbox.height,
                    )
            if vflip:
                polygon = [[point[0], output_height - point[1]] for point in polygon]
                if bbox is not None:
                    bbox = Box(
                        x=bbox.x,
                        y=max(0.0, output_height - bbox.y - bbox.height),
                        width=bbox.width,
                        height=bbox.height,
                    )
            rows.append(
                DatasetAnnotation(
                    class_id=annotation.class_id,
                    class_name=annotation.class_name,
                    kind=annotation.kind,
                    bbox=bbox,
                    polygon=polygon,
                )
            )
        return rows

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
        annotations = []
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
                    parsed = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    return []
                if isinstance(parsed, dict):
                    return [str(parsed[key]) for key in sorted(parsed)]
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
        return []

    def _reference_metadata(self, dirname: str) -> dict:
        root = self.settings.datasets_path / dirname
        yaml_path = root / "data.yaml"
        metadata = {}
        if yaml_path.exists():
            for line in yaml_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("workspace:"):
                    metadata["workspace"] = line.split(":", 1)[1].strip()
                if line.strip().startswith("project:"):
                    metadata["source_project"] = line.split(":", 1)[1].strip()
                if line.strip().startswith("version:"):
                    metadata["version"] = line.split(":", 1)[1].strip()
        return metadata

    def _create_layout(self, root: Path, format_name: str, task_type: str, labels: list[str]) -> None:
        for split in SPLITS:
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "annotations").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
        if format_name == "yolo":
            self._write_data_yaml(root, labels)

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
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "id": dataset_id,
                    "project_id": project_id,
                    "name": name,
                    "task_type": task_type,
                    "format": format_name,
                    "labels": labels,
                    "metadata": metadata,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _update_manifest(self, location: DatasetLocation, **updates) -> None:
        manifest_path = location.root / "manifest.json"
        manifest = {
            "id": location.id,
            "project_id": location.project_id,
            "name": location.name,
            "task_type": location.task_type,
            "format": location.format,
            "labels": location.labels,
            "metadata": location.metadata,
        }
        if manifest_path.exists():
            manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest.update(updates)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _label_is_used(self, location: DatasetLocation, label_index: int) -> bool:
        for split in SPLITS:
            for image_path in self._image_paths(location, split):
                with Image.open(image_path) as image:
                    width, height = image.size
                for annotation in self._annotations(location, split, image_path, width, height):
                    if annotation.class_id == label_index:
                        return True
        return False

    def _rewrite_annotations_after_label_delete(
        self, location: DatasetLocation, deleted_index: int, labels: list[str]
    ) -> None:
        for split in SPLITS:
            for image_path in self._image_paths(location, split):
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
                self._write_yolo_label(location, split, image_path.stem, annotations, width, height)

    def _new_dataset_id(self, name: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
        slug = "-".join(part for part in slug.split("-") if part)[:48] or "dataset"
        return f"{slug}-{uuid4().hex[:8]}"

    def _is_yolo_root(self, root: Path) -> bool:
        return any((root / split / "images").exists() for split in SPLITS)

    def _normalize_labels(self, labels: list[str]) -> list[str]:
        normalized = []
        for label in labels:
            cleaned = self._clean_label(label)
            if cleaned not in normalized:
                normalized.append(cleaned)
        return normalized or DEFAULT_LABELS.copy()

    def _clean_label(self, label: str) -> str:
        cleaned = " ".join(label.strip().split())
        if not cleaned:
            raise HTTPException(status_code=400, detail="Label cannot be empty")
        return cleaned[:80]

    def _validate_split(self, split: str) -> None:
        if split not in SPLITS:
            raise HTTPException(status_code=400, detail="Unknown dataset split")

    def _normalize_splits(self, splits: list[str]) -> list[str]:
        normalized = []
        for split in splits or list(SPLITS):
            self._validate_split(split)
            if split not in normalized:
                normalized.append(split)
        return normalized or ["train"]

    def _version_summary_from_manifest(
        self, dataset_id: str, manifest_path: Path
    ) -> DatasetVersionSummary:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest.get("metadata", {})
        config = self._normalize_preprocess_config(metadata.get("preprocess"))
        created_at_raw = metadata.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_raw)
            if isinstance(created_at_raw, str)
            else datetime.fromtimestamp(manifest_path.stat().st_mtime)
        )
        root = manifest_path.parent
        location = DatasetLocation(
            id=manifest["id"],
            project_id=manifest.get("project_id", DEFAULT_PROJECT_ID),
            name=manifest.get("name", root.name),
            task_type=manifest.get("task_type", DEFAULT_TASK_TYPE),
            format=manifest.get("format", "yolo"),
            source="editable",
            root=root,
            editable=True,
            labels=self._normalize_labels(manifest.get("labels") or DEFAULT_LABELS),
            metadata=metadata,
        )
        return DatasetVersionSummary(
            id=metadata.get("version_id", manifest["id"]),
            dataset_id=dataset_id,
            name=metadata.get("version_name", manifest.get("name", root.name)),
            path=str(root),
            image_count=int(metadata.get("image_count", 0)),
            generated_count=int(metadata.get("generated_count", 0)),
            splits={split: self._split_summary(location, split) for split in SPLITS},
            config=config,
            created_at=created_at,
        )
