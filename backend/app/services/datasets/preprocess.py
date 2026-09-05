"""Dataset preprocessing, augmentation, EDA summaries, and prepared-training roots.

This is a mixin consumed by ``DatasetService`` (see ``service.py``); it relies on
core/format-IO helpers (``_location``, ``_validate_split``, ``_is_nlp_task``,
``_annotations``, ``_write_annotation_json``, ``_write_yolo_label``,
``_create_layout``, ``_write_manifest``, ``_write_data_yaml``, ``training_root``,
``storage``) provided by the other mixins composed onto the facade class.
"""


import random
import re
import shutil
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import HTTPException
from PIL import Image

from app.core.storage import Storage
from app.schemas import (
    Box,
    DatasetAnnotation,
    DatasetEdaSummary,
    DatasetPreprocessConfig,
    DatasetPreprocessPreview,
)
from app.services.datasets.constants import (
    ALLOWED_PREPROCESS_TRANSFORMS,
    PREPROCESS_PRESETS,
    SPLITS,
    TRAINING_SPLITS,
)
from app.services.datasets.types import DatasetLocation


def eda_sample(paths: list, limit: int) -> tuple[list, int]:
    """`limit` items spread evenly across `paths`, plus how many were skipped.

    Striding rather than slicing is the whole point. `stanfordnlp/imdb` stores
    its train split sorted by label — the first half is `neg`, the second `pos` —
    so a head sample of any size reports one class and a perfectly balanced
    dataset looks degenerate. Every real ordering (by class, by source file, by
    date) has the same property, so the sample has to walk the whole split.

    Returns `(sample, sampled_count)` where `sampled_count` is `0` when nothing
    was skipped, which is how callers say "these numbers are exact".
    """
    total = len(paths)
    if limit <= 0 or total <= limit:
        return paths, 0
    stride = total / limit
    sample = [paths[int(index * stride)] for index in range(limit)]
    return sample, len(sample)


class PreprocessMixin:
    """Preprocess config normalization, image/text transforms, EDA, and prepared roots."""

    storage: Storage

    def _eda_limit(self) -> int:
        return int(getattr(self.settings, "eda_sample_items", 0) or 0)

    @staticmethod
    def _sampling_warning(sampled: int, total: int) -> list[str]:
        """Deliberately empty: sampling is not a warning.

        A dataset large enough to be sampled has nothing wrong with it, and
        putting the notice in `warnings` made a healthy 100,000-row dataset
        render as a problem. `sampled_items` carries the fact instead, and the
        studio states it as a caption. Kept as a seam so a future scan that
        samples for a *bad* reason has somewhere to say so.
        """
        del sampled, total
        return []

    def eda_summary(self, dataset_id: str, split: str) -> DatasetEdaSummary:
        splits = list(SPLITS) if split == "all" else [split]
        for split_name in splits:
            self._validate_split(split_name)
        location = self._location(dataset_id)
        if self._is_llm_task(location.task_type):
            return self._llm_eda_summary(location, dataset_id, split, splits)
        if self._is_nlp_task(location.task_type):
            return self._text_eda_summary(location, dataset_id, split, splits)
        class_counts = {label: 0 for label in location.labels}
        widths = []
        heights = []
        aspect_ratios = []
        image_count = 0
        annotation_count = 0
        unlabeled_count = 0
        missing_annotation_count = 0
        split_totals = {name: len(self._image_paths(location, name)) for name in SPLITS}
        scanned = [
            (split_name, path)
            for split_name in splits
            for path in self._image_paths(location, split_name)
        ]
        sample, sampled = eda_sample(scanned, self._eda_limit())
        for split_name, image_path in sample:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except FileNotFoundError:
                # See `_text_eda_summary`: the listing is a snapshot of a
                # directory a prep run may be rebuilding underneath it.
                continue
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
        warnings.extend(self._sampling_warning(sampled, len(scanned)))

        return DatasetEdaSummary(
            dataset_id=dataset_id,
            split=split,
            split_counts=split_totals,
            class_counts=class_counts,
            unlabeled_count=unlabeled_count,
            missing_annotation_count=missing_annotation_count,
            image_count=image_count,
            annotation_count=annotation_count,
            sampled_items=sampled,
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

    def prepared_training_root(self, dataset_id: str, output_root: Path) -> Path:
        location = self._location(dataset_id)
        preprocess = self._normalize_preprocess_config(
            (location.metadata or {}).get("preprocess") or DatasetPreprocessConfig()
        )
        if self._is_nlp_task(location.task_type):
            if not preprocess.enabled:
                return location.root
            return self._prepared_text_root(location, output_root, preprocess)
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

    def preprocess_preview(
        self,
        dataset_id: str,
        split: str,
        item_id: str,
        config: DatasetPreprocessConfig | None = None,
    ) -> DatasetPreprocessPreview:
        self._validate_split(split)
        location = self._location(dataset_id)
        if self._is_nlp_task(location.task_type):
            text_path = self._text_path(location, split, item_id)
            preprocess = self._normalize_preprocess_config(
                config or (location.metadata or {}).get("preprocess") or DatasetPreprocessConfig()
            )
            text = text_path.read_text(encoding="utf-8")
            processed = self._process_text(text, preprocess)
            preview = processed[:800] + ("..." if len(processed) > 800 else "")
            return DatasetPreprocessPreview(
                dataset_id=dataset_id,
                split=split,  # type: ignore[arg-type]
                item_id=text_path.name,
                media_type="text",
                image_url="",
                text_preview=preview,
                config=preprocess,
            )
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

    def _process_text(self, text: str, config: DatasetPreprocessConfig) -> str:
        if not config.enabled:
            return text
        tokens = text
        if "lowercase" in config.transforms:
            tokens = tokens.lower()
        if "remove_punctuation" in config.transforms:
            tokens = re.sub(r"[^\w\s]", " ", tokens)
        if "remove_stopwords" in config.transforms:
            stopwords = {
                "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
                "to", "was", "were", "will", "with",
            }
            tokens = " ".join(word for word in tokens.split() if word.lower() not in stopwords)
        if "normalize_whitespace" in config.transforms:
            tokens = " ".join(tokens.split())
        return tokens

    def _augment_text(self, text: str, copy_index: int) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        rng = random.Random(copy_index + len(text))
        synonyms = {
            "good": "positive",
            "great": "excellent",
            "bad": "negative",
            "small": "compact",
            "large": "big",
            "fast": "quick",
            "slow": "delayed",
        }
        if copy_index % 3 == 0:
            candidates = [idx for idx, word in enumerate(words) if word.lower().strip(".,;:!?") in synonyms]
            if candidates:
                idx = rng.choice(candidates)
                clean = words[idx].lower().strip(".,;:!?")
                words[idx] = synonyms[clean]
        elif copy_index % 3 == 1 and len(words) > 3:
            left = rng.randrange(0, len(words) - 1)
            words[left], words[left + 1] = words[left + 1], words[left]
        elif len(words) > 5:
            words.pop(rng.randrange(0, len(words)))
        return " ".join(words)

    def _prepared_text_root(
        self,
        location: DatasetLocation,
        output_root: Path,
        preprocess: DatasetPreprocessConfig,
    ) -> Path:
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
        for split in TRAINING_SPLITS:
            for text_path in self._text_paths(location, split):
                text = self._process_text(text_path.read_text(encoding="utf-8"), preprocess)
                annotations = self._annotations(location, split, text_path, 0, 0)
                output = output_root / split / "texts" / text_path.name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(text, encoding="utf-8")
                self._write_annotation_json(output_root / split / "annotations" / f"{text_path.stem}.json", annotations)
                if preprocess.augmentation_mode == "materialize" and split == "train":
                    for copy_index in range(preprocess.copies_per_image):
                        aug_stem = f"{text_path.stem}-aug-{copy_index + 1:02d}"
                        (output_root / split / "texts" / f"{aug_stem}.txt").write_text(
                            self._augment_text(text, copy_index),
                            encoding="utf-8",
                        )
                        self._write_annotation_json(output_root / split / "annotations" / f"{aug_stem}.json", annotations)
        return output_root

    def _text_eda_summary(
        self, location: DatasetLocation, dataset_id: str, split: str, splits: list[str]
    ) -> DatasetEdaSummary:
        class_counts = {label: 0 for label in location.labels}
        lengths = []
        token_counts = []
        annotation_count = 0
        unlabeled_count = 0
        missing_annotation_count = 0
        item_count = 0
        split_totals = {name: len(self._text_paths(location, name)) for name in SPLITS}
        scanned = [
            (split_name, path)
            for split_name in splits
            for path in self._text_paths(location, split_name)
        ]
        sample, sampled = eda_sample(scanned, self._eda_limit())
        for split_name, text_path in sample:
            try:
                text = text_path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                # Listed a moment ago, gone now: a prep run rebuilding the
                # splits from staging deleted it between the two. Summarize
                # what is still there rather than failing the whole request.
                continue
            lengths.append(len(text))
            token_counts.append(len(text.split()))
            item_count += 1
            annotations = self._annotations(location, split_name, text_path, 0, 0)
            annotation_count += len(annotations)
            if not self._is_text_labeled(location.task_type, annotations):
                unlabeled_count += 1
                missing_annotation_count += 1
            for annotation in annotations:
                if annotation.class_name:
                    class_counts[annotation.class_name] = class_counts.get(annotation.class_name, 0) + 1
        warnings = []
        if unlabeled_count:
            warnings.append(f"{unlabeled_count} text items are missing task annotations")
        nonzero_counts = [count for count in class_counts.values() if count > 0]
        if len(nonzero_counts) >= 2 and max(nonzero_counts) / max(min(nonzero_counts), 1) >= 3:
            warnings.append("Class distribution is imbalanced")
        if item_count == 0:
            warnings.append("No text items found in this split")
        warnings.extend(self._sampling_warning(sampled, len(scanned)))
        return DatasetEdaSummary(
            dataset_id=dataset_id,
            split=split,
            split_counts=split_totals,
            class_counts=class_counts,
            unlabeled_count=unlabeled_count,
            missing_annotation_count=missing_annotation_count,
            image_count=item_count,
            text_count=item_count,
            item_count=item_count,
            annotation_count=annotation_count,
            sampled_items=sampled,
            image_size={},
            aspect_ratio={},
            text_length={
                "min_chars": min(lengths) if lengths else None,
                "max_chars": max(lengths) if lengths else None,
                "mean_chars": round(mean(lengths), 2) if lengths else None,
                "min_tokens": min(token_counts) if token_counts else None,
                "max_tokens": max(token_counts) if token_counts else None,
                "mean_tokens": round(mean(token_counts), 2) if token_counts else None,
            },
            warnings=warnings,
        )

    if TYPE_CHECKING:
        # Provided by sibling mixins on the composed DatasetService facade.
        def _location(self, dataset_id: str) -> DatasetLocation:
            raise NotImplementedError

        def _validate_split(self, split: str) -> None:
            raise NotImplementedError

        def _is_nlp_task(self, task_type: str) -> bool:
            raise NotImplementedError

        def _is_llm_task(self, task_type: str) -> bool:
            raise NotImplementedError

        def _llm_eda_summary(
            self, location: DatasetLocation, dataset_id: str, split: str, splits: list[str]
        ) -> DatasetEdaSummary:
            raise NotImplementedError

        def _image_paths(self, location: DatasetLocation, split: str) -> list[Path]:
            raise NotImplementedError

        def _text_paths(self, location: DatasetLocation, split: str) -> list[Path]:
            raise NotImplementedError

        def _image_path(self, location: DatasetLocation, split: str, item_id: str) -> Path:
            raise NotImplementedError

        def _text_path(self, location: DatasetLocation, split: str, item_id: str) -> Path:
            raise NotImplementedError

        def _is_text_labeled(self, task_type: str, annotations: list[DatasetAnnotation]) -> bool:
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

        def training_root(self, dataset_id: str) -> Path:
            raise NotImplementedError

