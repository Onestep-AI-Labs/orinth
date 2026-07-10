from datetime import datetime
from pathlib import Path

from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.core.database import Base
from app.core.storage import Storage
from app.db.models import EvaluationJob, TrainingJob
from app.ml.model_registry import ModelRegistry
from app.schemas import (
    DatasetAnnotation,
    DatasetAnnotationSave,
    DatasetCreate,
    EvaluationJobBatchCreate,
    EvaluationJobCreate,
    ModelAssetPrepareRequest,
    TrainingJobCreate,
)
from app.services.evaluation import EvaluationService
from app.services.datasets import DatasetService
from app.services.job_progress import append_log, make_progress, progress_from_artifacts
from app.services.training import (
    clean_log_line,
    keras_application_kwargs,
    keras_application_name,
    parse_training_history,
    parse_yolo_results,
    TrainingService,
    ultralytics_initial_weights,
)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )


def test_job_progress_round_trips_from_artifacts():
    artifacts = append_log({}, "started")
    artifacts["progress"] = make_progress(
        percent=25,
        processed=1,
        total=4,
        current_step="Running",
        started_at=datetime.utcnow(),
        logs=artifacts["logs"],
    )

    progress = progress_from_artifacts(artifacts)

    assert progress.percent == 25
    assert progress.processed == 1
    assert progress.total == 4
    assert progress.logs == ["started"]


def test_parse_yolo_results_returns_latest_epoch_metrics(tmp_path: Path):
    results = tmp_path / "results.csv"
    results.write_text(
        "epoch,train/box_loss,metrics/mAP50(B)\n"
        "1,0.9,0.5\n"
        "2,0.7,0.75\n",
        encoding="utf-8",
    )

    metrics = parse_yolo_results(results)

    assert metrics["epoch"] == 2
    assert metrics["train/box_loss"] == 0.7
    assert metrics["metrics/mAP50(B)"] == 0.75


def test_parse_training_history_returns_all_epochs(tmp_path: Path):
    results = tmp_path / "results.csv"
    results.write_text(
        "epoch,accuracy,val_accuracy\n"
        "1,0.5,0.4\n"
        "2,0.8,0.7\n",
        encoding="utf-8",
    )

    history = parse_training_history(results)

    assert [row["epoch"] for row in history] == [1, 2]
    assert history[-1]["val_accuracy"] == 0.7


def test_training_command_includes_observability_parameters(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    dataset_root = settings.datasets_path / "dental dataset_yolov11_format"
    (dataset_root / "train" / "images").mkdir(parents=True)
    (dataset_root / "valid" / "images").mkdir(parents=True)
    service = TrainingService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )
    payload = TrainingJobCreate(
        model_family="yolo",
        epochs=5,
        image_size=512,
        batch_size=8,
        cache="disk",
        workers=2,
        patience=12,
        device="cpu",
    )
    job = TrainingJob(
        id="job-id",
        model_family="yolo",
        status="queued",
        parameters=payload.model_dump(),
    )

    command = service._command_for_job(job, tmp_path / "run")

    assert "--cache" in command
    assert "disk" in command
    assert "--workers" in command
    assert "2" in command
    assert "--patience" in command
    assert "12" in command
    assert "--device" in command
    assert "cpu" in command


def test_yolo26_option_uses_ultralytics_weights(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    dataset_root = settings.datasets_path / "dental dataset_yolov11_format"
    (dataset_root / "train" / "images").mkdir(parents=True)
    (dataset_root / "valid" / "images").mkdir(parents=True)
    service = TrainingService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )
    payload = TrainingJobCreate(
        model_option_id="ultralytics_yolo26_segment",
        model_family="yolo",
        task_type="segmentation",
        epochs=5,
        image_size=640,
        batch_size=8,
    )
    job = TrainingJob(
        id="job-id",
        model_family="yolo",
        status="queued",
        parameters=payload.model_dump(),
    )

    command = service._command_for_job(job, tmp_path / "run")

    assert "ultralytics_yolo26_segment" in {option.id for option in service.model_options("segmentation")}
    assert ultralytics_initial_weights("ultralytics_yolo26_segment", settings) == "yolo26n-seg.pt"
    assert "--initial-weights" in command
    assert "yolo26n-seg.pt" in command


def test_training_options_gate_transformer_and_generate_keras_command(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = TrainingService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )

    hf_option = service.option_by_id("hf_vit_base")
    assert hf_option.runnable is False
    assert service.prepare_model_asset(ModelAssetPrepareRequest(option_id="hf_vit_base")).status == "gated"

    payload = TrainingJobCreate(
        model_option_id="keras_efficientnet_b0",
        model_family="keras_classification",
        task_type="classification",
        dataset_id="reference_yolo",
        epochs=3,
        image_size=224,
        batch_size=4,
        optimizer="adam",
        learning_rate=0.001,
    )
    job = TrainingJob(
        id="keras-job",
        model_family="keras_classification",
        status="queued",
        parameters=payload.model_dump(),
    )

    command = service._command_for_job(job, tmp_path / "run")

    assert "app.training.runners.keras_classification_train" in command
    assert "--base-model" in command
    assert "EfficientNetB0" in command
    assert "--application-kwargs" in command


def test_training_registration_copies_stable_yolo_artifact(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = TrainingService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )
    run_dir = storage.training_runs / "job-id"
    best_model = run_dir / "weights" / "best.pt"
    best_model.parent.mkdir(parents=True, exist_ok=True)
    best_model.write_bytes(b"weights")
    job = TrainingJob(
        id="abcdef123456",
        project_id="project-one",
        model_family="yolo",
        status="completed",
        parameters={
            "task_type": "segmentation",
            "dataset_id": "reference_yolo",
            "image_size": 512,
        },
    )

    model_id = service._register_training_model(job, run_dir, best_model, {"metrics/mAP50(B)": 0.8})
    model = service.registry.get_spec(model_id)

    assert model.available is True
    assert model.project_id == "project-one"
    assert model.paths["weights"].is_relative_to(storage.trained_models)


def test_ultralytics_catalog_is_task_filtered_and_gates_unvalidated_families(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = TrainingService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )

    detection_options = {option.id: option for option in service.model_options("object_detection")}
    segmentation_options = {option.id: option for option in service.model_options("segmentation")}
    classification_options = {option.id for option in service.model_options("classification")}

    assert "ultralytics_yolo26_detect" in detection_options
    assert "ultralytics_rt_detr" in detection_options
    assert "ultralytics_yolo26_segment" in segmentation_options
    assert "ultralytics_sam3" in segmentation_options
    assert segmentation_options["ultralytics_sam3"].runnable is False
    assert "keras_mobilenet_v2" in classification_options
    assert "ultralytics_yolo26_detect" not in classification_options


def test_keras_application_catalog_includes_official_examples(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    service = TrainingService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )

    option_ids = {option.id for option in service.model_options()}

    assert "keras_mobilenet_v2" in option_ids
    assert "keras_efficientnet_b7" in option_ids
    assert "keras_convnext_tiny" in option_ids
    assert keras_application_name("keras_mobilenet_v2") == "MobileNetV2"
    assert keras_application_kwargs("keras_mobilenet_v2") == {"alpha": 1.0}
    assert keras_application_kwargs("keras_efficientnet_b7") == {"name": "efficientnetb7"}


def test_clean_log_line_removes_ansi_sequences():
    assert clean_log_line("\x1b[34mtrain:\x1b[0m 1/5\r") == "train: 1/5"


def test_evaluation_batch_creates_one_job_per_model(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    service = EvaluationService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )

    jobs = service.create_jobs(
        db,
        EvaluationJobBatchCreate(
            model_ids=["yolo_11_best", "unet_inception"],
            dataset_key="yolo_test",
            limit=2,
        ),
    )

    assert [job.model_id for job in jobs] == ["yolo_11_best", "unet_inception"]
    assert {job.limit for job in jobs} == {2}
    assert len({job.comparison_id for job in jobs}) == 1
    assert db.query(EvaluationJob).count() == 2
    db.close()


def test_evaluation_single_job_has_stable_comparison_id(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    engine = create_engine(f"sqlite:///{tmp_path / 'single.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    service = EvaluationService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )

    job = service.create_job(db, EvaluationJobCreate(model_id="yolo_11_best", dataset_key="yolo_test"))
    comparison = service.comparison_jobs(db, job.id)

    assert job.comparison_id == job.id
    assert [item.id for item in comparison] == [job.id]
    db.close()


def test_evaluation_comparison_returns_sibling_jobs(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    engine = create_engine(f"sqlite:///{tmp_path / 'comparison.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    service = EvaluationService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )
    jobs = service.create_jobs(
        db,
        EvaluationJobBatchCreate(
            model_ids=["yolo_11_best", "unet_inception"],
            dataset_key="yolo_test",
        ),
    )

    comparison = service.comparison_jobs(db, jobs[0].id)

    assert {job.id for job in comparison} == {job.id for job in jobs}
    db.close()


def test_evaluation_dataset_discovery_only_lists_editable_test_split(tmp_path: Path):
    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    dataset_service = DatasetService(settings, storage)
    dataset = dataset_service.create_dataset(
        DatasetCreate(
            name="Split Eval",
            task_type="segmentation",
            format="yolo",
            labels=["lesion"],
        )
    )
    for split in ("train", "valid", "test"):
        image_path = storage.datasets / dataset.id / split / "images" / f"{split}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), "white").save(image_path)

    service = EvaluationService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        dataset_service,
    )

    rows = [row for row in service.list_datasets() if row.key.startswith(f"dataset:{dataset.id}:")]

    assert [row.split for row in rows] == ["test"]
    assert rows[0].key == f"dataset:{dataset.id}:test"


def test_classification_evaluation_uses_predictor_scores(tmp_path: Path):
    class FakeSpec:
        task_type = "classification"

    class FakePredictor:
        def predict(self, image_path, parameters):
            return []

        def classify(self, image_path, parameters):
            if image_path.name.startswith("ok"):
                return {"ok": 0.9, "bad": 0.1}
            return {"ok": 0.2, "bad": 0.8}

    class FakeRegistry:
        def get_spec(self, model_id):
            return FakeSpec()

        def get_predictor(self, model_id):
            return FakePredictor()

    settings = make_settings(tmp_path)
    storage = Storage(settings)
    storage.ensure()
    dataset_service = DatasetService(settings, storage)
    dataset = dataset_service.create_dataset(
        DatasetCreate(
            name="Classifier Eval",
            task_type="classification",
            format="image_folder",
            labels=["ok", "bad"],
        )
    )
    for filename, class_id, class_name in [("ok.jpg", 0, "ok"), ("bad.jpg", 1, "bad")]:
        image_path = storage.datasets / dataset.id / "test" / "images" / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), "white").save(image_path)
        dataset_service.save_annotations(
            dataset.id,
            "test",
            filename,
            DatasetAnnotationSave(
                annotations=[
                    DatasetAnnotation(
                        class_id=class_id,
                        class_name=class_name,
                        kind="classification",
                    )
                ]
            ),
        )

    engine = create_engine(f"sqlite:///{tmp_path / 'classification.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    service = EvaluationService(settings, storage, FakeRegistry(), dataset_service)
    job = service.create_job(
        db,
        EvaluationJobCreate(model_id="fake_classifier", dataset_key=f"dataset:{dataset.id}:test"),
    )

    metrics, _artifacts = service._evaluate_classification(db, job, datetime.utcnow())

    assert metrics["samples"] == 2
    assert metrics["image"]["overall"]["accuracy"] == 1.0
    assert metrics["classification"]["macro_auc"] == 1.0
    db.close()
