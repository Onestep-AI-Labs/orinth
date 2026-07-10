from io import BytesIO
from pathlib import Path

import anyio
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.core.config import Settings
from app.core.storage import Storage
from app.schemas import (
    DatasetAnnotation,
    DatasetAnnotationSave,
    DatasetCreate,
    DatasetImportRequest,
    DatasetItemBulkLabelUpdate,
    DatasetItemDeleteRequest,
    DatasetItemLabelUpdate,
    DatasetItemMoveRequest,
    DatasetProcessRequest,
    DatasetSplitConfig,
    DatasetPreprocessConfig,
    DatasetVersionCreate,
    DatasetUpdate,
)
from app.services.datasets import DatasetService


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )


def write_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def upload_file(name: str = "sample.jpg", size: tuple[int, int] = (100, 80)) -> UploadFile:
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="JPEG")
    buffer.seek(0)
    return UploadFile(file=buffer, filename=name)


def write_yolo_dataset(root: Path) -> None:
    for split in ("train", "valid", "test"):
        write_image(root / split / "images" / f"{split}.jpg")
        label_dir = root / split / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / f"{split}.txt").write_text(
            "0 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000\n",
            encoding="utf-8",
        )


def test_dataset_service_creates_and_saves_yolo_annotations(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)

    dataset = service.create_dataset(DatasetCreate(name="Editable Dental"))
    image_path = storage.datasets / dataset.id / "train" / "images" / "sample.jpg"
    write_image(image_path)

    detail = service.save_annotations(
        dataset.id,
        "train",
        "sample.jpg",
        DatasetAnnotationSave(
            annotations=[
                DatasetAnnotation(
                    class_id=1,
                    class_name="kista",
                    polygon=[[10, 10], [50, 10], [50, 50], [10, 50]],
                )
            ]
        ),
    )

    assert detail.annotation_count == 1
    assert detail.annotations[0].class_name == "kista"
    label_text = (storage.datasets / dataset.id / "train" / "labels" / "sample.txt").read_text(
        encoding="utf-8"
    )
    assert label_text.startswith("1 ")


def test_reference_dataset_annotations_are_read_only(tmp_path: Path):
    settings = make_settings(tmp_path)
    reference_root = settings.datasets_path / "dental dataset_yolov11_format"
    write_yolo_dataset(reference_root)
    service = DatasetService(settings, Storage(settings))

    with pytest.raises(HTTPException) as exc:
        service.save_annotations(
            "reference_yolo",
            "train",
            "train.jpg",
            DatasetAnnotationSave(annotations=[]),
        )

    assert exc.value.status_code == 409


def test_dataset_import_copies_yolo_layout_to_storage(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    source = tmp_path / "source-yolo"
    write_yolo_dataset(source)
    service = DatasetService(settings, storage)

    dataset = service.import_dataset(DatasetImportRequest(path=str(source), name="Imported"))

    assert dataset.editable is True
    assert dataset.splits["train"].image_count == 1
    assert (storage.datasets / dataset.id / "train" / "images" / "train.jpg").exists()


def test_dataset_labels_are_manifest_driven_and_can_exceed_two_classes(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)

    dataset = service.create_dataset(
        DatasetCreate(name="Wildlife", labels=["cat", "dog", "horse"], task_type="segmentation")
    )
    image_path = storage.datasets / dataset.id / "train" / "images" / "sample.jpg"
    write_image(image_path)

    detail = service.save_annotations(
        dataset.id,
        "train",
        "sample.jpg",
        DatasetAnnotationSave(
            annotations=[
                DatasetAnnotation(
                    class_id=2,
                    class_name="horse",
                    kind="polygon",
                    polygon=[[10, 10], [50, 10], [50, 50], [10, 50]],
                )
            ]
        ),
    )

    assert detail.annotations[0].class_name == "horse"
    assert service.summary(dataset.id).labels == ["cat", "dog", "horse"]


def test_classification_dataset_stores_label_only_annotations(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)
    dataset = service.create_dataset(
        DatasetCreate(name="Classifier", task_type="classification", format="image_folder", labels=["ok", "bad"])
    )
    image_path = storage.datasets / dataset.id / "train" / "images" / "sample.jpg"
    write_image(image_path)

    detail = service.save_annotations(
        dataset.id,
        "train",
        "sample.jpg",
        DatasetAnnotationSave(
            annotations=[
                DatasetAnnotation(class_id=1, class_name="bad", kind="classification")
            ]
        ),
    )

    assert detail.annotations[0].kind == "classification"
    assert detail.annotations[0].polygon == []


def test_classification_upload_and_label_patch(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)
    dataset = service.create_dataset(
        DatasetCreate(
            name="Classifier",
            task_type="classification",
            format="image_folder",
            labels=["ok", "bad"],
        )
    )

    detail = anyio.run(
        service.upload_image,
        dataset.id,
        "train",
        upload_file(),
        1,
        None,
    )
    assert detail.annotations[0].class_name == "bad"

    updated = service.set_item_label(
        dataset.id,
        "train",
        detail.id,
        DatasetItemLabelUpdate(class_name="ok"),
    )

    assert updated.annotations[0].class_name == "ok"


def test_batch_upload_delete_and_bulk_label_edit(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)
    dataset = service.create_dataset(
        DatasetCreate(
            name="Batch Classifier",
            task_type="classification",
            format="image_folder",
            labels=["paper", "plastic"],
        )
    )

    result = anyio.run(
        service.upload_images,
        dataset.id,
        "train",
        [upload_file("one.jpg"), upload_file("two.jpg")],
        0,
        None,
    )

    assert len(result.uploaded) == 2
    assert not result.errors
    assert {item.label for item in result.uploaded} == {"paper"}

    relabeled = service.bulk_set_item_labels(
        dataset.id,
        "train",
        DatasetItemBulkLabelUpdate(ids=[item.id for item in result.uploaded], class_name="plastic"),
    )

    assert relabeled.updated == 2
    assert not relabeled.missing
    assert {item.annotations[0].class_name for item in relabeled.items} == {"plastic"}

    deleted = service.delete_items(
        dataset.id,
        DatasetItemDeleteRequest(split="train", ids=[result.uploaded[0].id, "missing.jpg"]),
    )

    assert deleted.deleted == 1
    assert deleted.missing == ["missing.jpg"]
    assert not (storage.datasets / dataset.id / "train" / "images" / result.uploaded[0].id).exists()
    assert not (
        storage.datasets / dataset.id / "train" / "annotations" / f"{Path(result.uploaded[0].id).stem}.json"
    ).exists()


def test_dataset_process_splits_unassigned_and_move_items(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)
    dataset = service.create_dataset(
        DatasetCreate(
            name="Inbox Classifier",
            task_type="classification",
            format="image_folder",
            labels=["paper", "plastic"],
        )
    )
    uploaded = anyio.run(
        service.upload_images,
        dataset.id,
        "unassigned",
        [upload_file(f"sample-{index}.jpg") for index in range(10)],
        0,
        None,
    )

    processed = service.process_dataset(
        dataset.id,
        DatasetProcessRequest(split=DatasetSplitConfig(train=0.7, valid=0.2, test=0.1, seed=42)),
    )

    assert len(uploaded.uploaded) == 10
    assert processed.dataset.splits["unassigned"].image_count == 0
    assert processed.dataset.splits["train"].image_count == 7
    assert processed.dataset.splits["valid"].image_count == 2
    assert processed.dataset.splits["test"].image_count == 1

    train_item = service.list_items(dataset.id, "train", limit=1)[0]
    moved = service.move_items(
        dataset.id,
        DatasetItemMoveRequest(
            source_split="train",
            target_split="test",
            ids=[train_item.id],
        ),
    )

    assert moved.moved == 1
    assert service.summary(dataset.id).splits["test"].image_count == 2


def test_reference_dataset_rejects_item_mutations(tmp_path: Path):
    settings = make_settings(tmp_path)
    reference_root = settings.datasets_path / "dental dataset_yolov11_format"
    write_yolo_dataset(reference_root)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)

    with pytest.raises(HTTPException) as exc:
        anyio.run(service.upload_images, "reference_yolo", "train", [upload_file()])
    assert exc.value.status_code == 409

    with pytest.raises(HTTPException) as exc:
        service.delete_items(
            "reference_yolo",
            DatasetItemDeleteRequest(split="train", ids=["train.jpg"]),
        )
    assert exc.value.status_code == 409

    with pytest.raises(HTTPException) as exc:
        service.bulk_set_item_labels(
            "reference_yolo",
            "train",
            DatasetItemBulkLabelUpdate(ids=["train.jpg"], class_id=0),
        )
    assert exc.value.status_code == 409


def test_dataset_update_and_preprocess_validation(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)
    dataset = service.create_dataset(DatasetCreate(name="Editable"))

    updated = service.update_dataset(
        dataset.id,
        DatasetUpdate(
            name="Editable Updated",
            preprocess=DatasetPreprocessConfig(
                enabled=True,
                resize_width=64,
                resize_height=64,
                transforms=["horizontal_flip"],
            ),
        ),
    )

    assert updated.name == "Editable Updated"
    assert updated.metadata["preprocess"]["enabled"] is True
    assert updated.metadata["preprocess"]["transforms"] == ["horizontal_flip"]

    with pytest.raises(HTTPException) as exc:
        service.update_dataset(
            dataset.id,
            DatasetUpdate(
                preprocess=DatasetPreprocessConfig(
                    enabled=True,
                    transforms=["unknown_transform"],
                )
            ),
        )
    assert exc.value.status_code == 400


def test_preprocess_preview_and_prepared_training_copy_do_not_modify_source(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)
    dataset = service.create_dataset(
        DatasetCreate(
            name="Classifier",
            task_type="classification",
            format="image_folder",
            labels=["ok", "bad"],
        )
    )
    image_path = storage.datasets / dataset.id / "train" / "images" / "sample.jpg"
    write_image(image_path, (100, 80))
    service.set_item_label(
        dataset.id,
        "train",
        "sample.jpg",
        DatasetItemLabelUpdate(class_id=1),
    )
    config = DatasetPreprocessConfig(
        enabled=True,
        resize_width=64,
        resize_height=64,
        transforms=["horizontal_flip"],
    )

    preview = service.preprocess_preview(dataset.id, "train", "sample.jpg", config)
    preview_path = storage.root / preview.image_url.removeprefix("/media/")
    assert preview_path.exists()

    service.update_dataset(dataset.id, DatasetUpdate(preprocess=config))
    prepared = service.prepared_training_root(dataset.id, tmp_path / "prepared")

    with Image.open(image_path) as original:
        assert original.size == (100, 80)
    with Image.open(prepared / "train" / "images" / "sample.jpg") as transformed:
        assert transformed.size == (64, 64)


def test_materialized_version_creates_augmented_artifacts_without_touching_source(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)
    dataset = service.create_dataset(
        DatasetCreate(
            name="Versioned Classifier",
            task_type="classification",
            format="image_folder",
            labels=["ok", "bad"],
        )
    )
    detail = anyio.run(service.upload_image, dataset.id, "train", upload_file("sample.jpg"), 1, None)
    original_path = storage.datasets / dataset.id / "train" / "images" / detail.id

    version = service.create_version(
        dataset.id,
        DatasetVersionCreate(
            name="augmented",
            splits=["train"],
            augmentation_splits=["train"],
            config=DatasetPreprocessConfig(
                enabled=True,
                resize_width=64,
                resize_height=64,
                transforms=["horizontal_flip"],
                augmentation_mode="materialize",
                copies_per_image=4,
            ),
        ),
    )

    assert original_path.exists()
    assert version.image_count == 5
    assert version.generated_count == 4
    version_root = storage.dataset_versions / dataset.id / version.id
    assert len(list((version_root / "train" / "images").glob("*.jpg"))) == 5
    with Image.open(original_path) as original:
        assert original.size == (100, 80)


def test_random_augmentation_mode_does_not_generate_extra_files(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)
    dataset = service.create_dataset(
        DatasetCreate(
            name="Random Classifier",
            task_type="classification",
            format="image_folder",
            labels=["ok", "bad"],
        )
    )
    anyio.run(service.upload_image, dataset.id, "train", upload_file("sample.jpg"), 1, None)

    version = service.create_version(
        dataset.id,
        DatasetVersionCreate(
            name="random",
            splits=["train"],
            augmentation_splits=["train"],
            config=DatasetPreprocessConfig(
                enabled=True,
                transforms=["brightness_contrast"],
                augmentation_mode="random",
                copies_per_image=4,
            ),
        ),
    )

    assert version.image_count == 1
    assert version.generated_count == 0
    version_root = storage.dataset_versions / dataset.id / version.id
    assert len(list((version_root / "train" / "images").glob("*.jpg"))) == 1


def test_eda_summary_reports_counts_and_warnings(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)
    dataset = service.create_dataset(
        DatasetCreate(
            name="EDA Classifier",
            task_type="classification",
            format="image_folder",
            labels=["paper", "plastic"],
        )
    )
    anyio.run(service.upload_image, dataset.id, "train", upload_file("paper.jpg"), 0, None)
    anyio.run(service.upload_image, dataset.id, "train", upload_file("unlabeled.jpg"), None, None)

    eda = service.eda_summary(dataset.id, "train")

    assert eda.image_count == 2
    assert eda.annotation_count == 1
    assert eda.class_counts["paper"] == 1
    assert eda.class_counts["plastic"] == 0
    assert eda.unlabeled_count == 1
    assert eda.split_counts["train"] == 2
    assert eda.image_size["mean_width"] == 100
    assert eda.warnings


def test_dataset_delete_removes_editable_storage_and_blocks_reference(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)
    dataset = service.create_dataset(DatasetCreate(name="Delete Me"))
    dataset_root = storage.datasets / dataset.id

    service.delete_dataset(dataset.id)

    assert not dataset_root.exists()
    with pytest.raises(HTTPException) as exc:
        service.delete_dataset("reference_yolo")
    assert exc.value.status_code == 409


def test_label_delete_blocks_used_annotations(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = DatasetService(settings, storage)
    dataset = service.create_dataset(DatasetCreate(name="Used Label", labels=["a", "b"]))
    image_path = storage.datasets / dataset.id / "train" / "images" / "sample.jpg"
    write_image(image_path)
    service.save_annotations(
        dataset.id,
        "train",
        "sample.jpg",
        DatasetAnnotationSave(
            annotations=[
                DatasetAnnotation(
                    class_id=1,
                    class_name="b",
                    kind="polygon",
                    polygon=[[10, 10], [50, 10], [50, 50], [10, 50]],
                )
            ]
        ),
    )

    with pytest.raises(HTTPException) as exc:
        service.delete_label(dataset.id, 1)

    assert exc.value.status_code == 409
