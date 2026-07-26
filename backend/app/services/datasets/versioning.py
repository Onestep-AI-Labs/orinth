"""Dataset item moves, split/process orchestration, and version snapshots.

This is a mixin consumed by ``DatasetService`` (see ``service.py``); it relies on
core/format-IO/preprocess helpers (``_location``, ``_editable_location``,
``_update_manifest``, ``_create_layout``, ``_write_manifest``, ``_write_data_yaml``,
``_write_annotation_json``, ``_write_yolo_label``, ``_normalize_preprocess_config``,
``_preprocessed_image_and_annotations``, ``_process_text``, ``_augment_text``,
``summary``, ``storage``) provided by the other mixins composed onto the facade
class.
"""


import json
import random
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from fastapi import HTTPException

from app.core.defaults import DEFAULT_PROJECT_ID, DEFAULT_TASK_TYPE
from app.core.storage import Storage
from app.schemas import (
    DatasetAnnotation,
    DatasetItemDetail,
    DatasetItemMoveRequest,
    DatasetItemMoveResponse,
    DatasetItemSummary,
    DatasetPreprocessConfig,
    DatasetProcessRequest,
    DatasetProcessResponse,
    DatasetSplitConfig,
    DatasetSplitSummary,
    DatasetSummary,
    DatasetVersionCreate,
    DatasetVersionSummary,
)
from app.services.datasets.constants import SPLITS, TRAINING_SPLITS
from app.services.datasets.types import DatasetLocation


class VersioningMixin:
    """Item moves, train/valid/test split processing, and dataset version snapshots."""

    storage: Storage


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
        items: list[DatasetItemSummary] = []
        for item_id in payload.ids:
            try:
                new_name = self._move_item(location, payload.source_split, payload.target_split, item_id)
            except HTTPException:
                missing.append(item_id)
                continue
            moved += 1
            items.append(self.item_detail(dataset_id, payload.target_split, new_name))
        if moved and self._is_llm_task(location.task_type):
            self._regenerate_records_jsonl(location, payload.source_split)
            self._regenerate_records_jsonl(location, payload.target_split)
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
        metadata["processed_at"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
        self._update_manifest(location, metadata=metadata)
        location = self._location(dataset_id)

        source_splits = list(SPLITS if split_config.resplit_all else ("unassigned",))
        is_llm = self._is_llm_task(location.task_type)
        candidates: list[tuple[str, Any]] = []
        for source_split in source_splits:
            for item_path in self._item_paths(location, source_split):
                # LLM records carry no label, so splitting never needs to parse the
                # record body — reading every file just to compute an excerpt made
                # splitting a large imported dataset time out. Use a light stub.
                if is_llm:
                    candidates.append((source_split, SimpleNamespace(id=item_path.name, label=None)))
                else:
                    detail = self._item_from_path(location, source_split, item_path, include_annotations=True)
                    candidates.append((source_split, detail))

        grouped: dict[str, list[tuple[str, Any]]] = {}
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
            for (source_split, item), target_split in zip(rows, targets, strict=True):
                if source_split == target_split:
                    continue
                new_name = self._move_item(location, source_split, target_split, item.id)
                moved[target_split] += 1
                if new_name != item.id:
                    item.id = new_name

        if self._is_llm_task(location.task_type):
            self._regenerate_all_records_jsonl(location)

        return DatasetProcessResponse(
            dataset=self.summary(dataset_id),
            moved=moved,
            split_config=split_config,
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
        version_name = payload.name or f"{location.name} version {datetime.now(UTC).replace(tzinfo=None).strftime('%Y-%m-%d %H:%M')}"
        self._create_layout(version_root, location.format, location.task_type, location.labels)
        if self._is_nlp_task(location.task_type):
            return self._create_text_version(
                location,
                dataset_id=dataset_id,
                version_id=version_id,
                version_root=version_root,
                version_name=version_name,
                splits=splits,
                augmentation_splits=augmentation_splits,
                config=config,
            )

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

        created_at = datetime.now(UTC).replace(tzinfo=None)
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

    def _move_item(
        self,
        location: DatasetLocation,
        source_split: str,
        target_split: str,
        item_id: str,
    ) -> str:
        item_path = self._item_path(location, source_split, item_id)
        source_stem = item_path.stem
        media_dir = self._media_dir(location.task_type)
        target_image_dir = location.root / target_split / media_dir
        target_annotation_dir = location.root / target_split / "annotations"
        target_label_dir = location.root / target_split / "labels"
        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_annotation_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

        target_name = item_path.name
        if (target_image_dir / target_name).exists():
            target_name = f"{item_path.stem}-{uuid4().hex[:8]}{item_path.suffix}"
        target_stem = Path(target_name).stem

        shutil.move(str(item_path), target_image_dir / target_name)
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

    def _normalize_splits(self, splits: Sequence[str]) -> list[str]:
        normalized = []
        for split in splits or list(SPLITS):
            self._validate_split(split)
            if split not in normalized:
                normalized.append(split)
        return normalized or ["train"]

    def _create_text_version(
        self,
        location: DatasetLocation,
        *,
        dataset_id: str,
        version_id: str,
        version_root: Path,
        version_name: str,
        splits: list[str],
        augmentation_splits: list[str],
        config: DatasetPreprocessConfig,
    ) -> DatasetVersionSummary:
        text_count = 0
        generated_count = 0
        for split in splits:
            for text_path in self._text_paths(location, split):
                text = self._process_text(text_path.read_text(encoding="utf-8"), config)
                annotations = self._annotations(location, split, text_path, 0, 0)
                output = version_root / split / "texts" / text_path.name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(text, encoding="utf-8")
                self._write_annotation_json(version_root / split / "annotations" / f"{text_path.stem}.json", annotations)
                text_count += 1
                if (
                    not config.enabled
                    or config.augmentation_mode != "materialize"
                    or split not in augmentation_splits
                ):
                    continue
                for copy_index in range(config.copies_per_image):
                    aug_stem = f"{text_path.stem}-aug-{copy_index + 1:02d}"
                    aug_text = self._augment_text(text, copy_index)
                    (version_root / split / "texts" / f"{aug_stem}.txt").write_text(aug_text, encoding="utf-8")
                    self._write_annotation_json(version_root / split / "annotations" / f"{aug_stem}.json", annotations)
                    text_count += 1
                    generated_count += 1
        created_at = datetime.now(UTC).replace(tzinfo=None)
        metadata = {
            "dataset_id": dataset_id,
            "version_id": version_id,
            "version_name": version_name,
            "created_at": created_at.isoformat(),
            "preprocess": config.model_dump(mode="json"),
            "selected_splits": splits,
            "augmentation_splits": augmentation_splits,
            "text_count": text_count,
            "item_count": text_count,
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
        return self._version_summary_from_manifest(dataset_id, version_root / "manifest.json")

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
            else datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=UTC)
        )
        root = manifest_path.parent
        task_type = self._normalize_task_type(manifest.get("task_type", DEFAULT_TASK_TYPE))
        location = DatasetLocation(
            id=manifest["id"],
            project_id=manifest.get("project_id", DEFAULT_PROJECT_ID),
            name=manifest.get("name", root.name),
            task_type=task_type,
            format=manifest.get("format", "yolo"),
            source="editable",
            root=root,
            editable=True,
            labels=self._normalize_labels(
                manifest.get("labels") or self._default_labels_for_task(task_type),
                task_type,
            ),
            metadata=metadata,
        )
        return DatasetVersionSummary(
            id=metadata.get("version_id", manifest["id"]),
            dataset_id=dataset_id,
            name=metadata.get("version_name", manifest.get("name", root.name)),
            path=str(root),
            image_count=int(metadata.get("image_count", 0)),
            text_count=int(metadata.get("text_count", 0)),
            item_count=int(metadata.get("item_count", metadata.get("image_count", metadata.get("text_count", 0)))),
            generated_count=int(metadata.get("generated_count", 0)),
            splits={split: self._split_summary(location, split) for split in SPLITS},
            config=config,
            created_at=created_at,
        )

    if TYPE_CHECKING:
        # Provided by sibling mixins on the composed DatasetService facade.
        def _location(self, dataset_id: str) -> DatasetLocation:
            raise NotImplementedError

        def _editable_location(self, dataset_id: str) -> DatasetLocation:
            raise NotImplementedError

        def _is_nlp_task(self, task_type: str) -> bool:
            raise NotImplementedError

        def _is_llm_task(self, task_type: str) -> bool:
            raise NotImplementedError

        def _media_dir(self, task_type: str) -> str:
            raise NotImplementedError

        def _regenerate_records_jsonl(self, location: DatasetLocation, split: str) -> None:
            raise NotImplementedError

        def _regenerate_all_records_jsonl(self, location: DatasetLocation) -> None:
            raise NotImplementedError

        def _normalize_task_type(self, task_type: str) -> str:
            raise NotImplementedError

        def _normalize_labels(self, labels: list[str], task_type: str = ...) -> list[str]:
            raise NotImplementedError

        def _default_labels_for_task(self, task_type: str) -> list[str]:
            raise NotImplementedError

        def _item_exists(self, location: DatasetLocation, split: str, item_id: str) -> bool:
            raise NotImplementedError

        def _item_path(self, location: DatasetLocation, split: str, item_id: str) -> Path:
            raise NotImplementedError

        def _item_paths(self, location: DatasetLocation, split: str) -> list[Path]:
            raise NotImplementedError

        def _item_from_path(
            self, location: DatasetLocation, split: str, image_path: Path, include_annotations: bool
        ) -> DatasetItemDetail:
            raise NotImplementedError

        def _image_paths(self, location: DatasetLocation, split: str) -> list[Path]:
            raise NotImplementedError

        def _text_paths(self, location: DatasetLocation, split: str) -> list[Path]:
            raise NotImplementedError

        def _split_summary(self, location: DatasetLocation, split: str) -> DatasetSplitSummary:
            raise NotImplementedError

        def _annotations(
            self, location: DatasetLocation, split: str, image_path: Path, width: int, height: int
        ) -> list[DatasetAnnotation]:
            raise NotImplementedError

        def _write_annotation_json(self, path: Path, annotations: list[DatasetAnnotation]) -> None:
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

        def _write_data_yaml(self, root: Path, labels: list[str]) -> None:
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

        def _update_manifest(self, location: DatasetLocation, **updates) -> None:
            raise NotImplementedError

        def _normalize_preprocess_config(
            self, config: DatasetPreprocessConfig | dict | None
        ) -> DatasetPreprocessConfig:
            raise NotImplementedError

        def _preprocessed_image_and_annotations(
            self, location: DatasetLocation, split: str, image_path: Path, config: DatasetPreprocessConfig
        ) -> tuple:
            raise NotImplementedError

        def _process_text(self, text: str, config: DatasetPreprocessConfig) -> str:
            raise NotImplementedError

        def _augment_text(self, text: str, copy_index: int) -> str:
            raise NotImplementedError

        def item_detail(self, dataset_id: str, split: str, item_id: str) -> DatasetItemDetail:
            raise NotImplementedError

        def summary(self, dataset_id: str) -> DatasetSummary:
            raise NotImplementedError

        def _validate_split(self, split: str) -> None:
            raise NotImplementedError

