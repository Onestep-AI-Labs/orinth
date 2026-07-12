from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.storage import Storage
from app.db.models import EvaluationJob, TrainingJob
from app.ml.model_registry import ModelRegistry
from app.schemas import (
    DatasetAnnotation,
    DatasetAnnotationSave,
    DatasetCreate,
    EvaluationJobBatchCreate,
    EvaluationJobCreate,
    InferenceParameters,
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
from app.training.runners.nlp_train import (
    train_keras_seq2seq_summarizer,
    train_keras_text_classifier,
    train_text_classifier,
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


def test_training_command_includes_observability_parameters(tmp_path: Path, settings: Settings):
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


def test_yolo26_option_uses_ultralytics_weights(tmp_path: Path, settings: Settings):
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


def test_training_options_gate_transformer_and_generate_keras_command(tmp_path: Path, settings: Settings):
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


def test_training_registration_copies_stable_yolo_artifact(tmp_path: Path, settings: Settings):
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


def test_ultralytics_catalog_is_task_filtered_and_gates_unvalidated_families(tmp_path: Path, settings: Settings):
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


def test_keras_application_catalog_includes_official_examples(tmp_path: Path, settings: Settings):
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


def test_clean_log_line_filters_expected_transformer_load_report():
    assert (
        clean_log_line(
            "[transformers] BertForSequenceClassification LOAD REPORT from: bert-base-uncased"
        )
        == ""
    )
    assert clean_log_line("classifier.weight | MISSING |") == ""
    assert clean_log_line("Hugging Face BERT classifier: loading the base checkpoint") != ""


def test_training_process_env_exports_hf_token_aliases(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
        HF_TOKEN="hf_alias_token",
    )
    storage = Storage(settings)
    service = TrainingService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )

    env = service._process_env()

    assert env["HF_TOKEN"] == "hf_alias_token"
    assert env["HUGGINGFACE_HUB_TOKEN"] == "hf_alias_token"


def test_evaluation_batch_creates_one_job_per_model(settings: Settings, db_session: Session):
    storage = Storage(settings)
    storage.ensure()
    service = EvaluationService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )

    jobs = service.create_jobs(
        db_session,
        EvaluationJobBatchCreate(
            model_ids=["yolo_11_best", "unet_inception"],
            dataset_key="yolo_test",
            limit=2,
        ),
    )

    assert [job.model_id for job in jobs] == ["yolo_11_best", "unet_inception"]
    assert {job.limit for job in jobs} == {2}
    assert len({job.comparison_id for job in jobs}) == 1
    assert db_session.query(EvaluationJob).count() == 2


def test_evaluation_single_job_has_stable_comparison_id(settings: Settings, db_session: Session):
    storage = Storage(settings)
    storage.ensure()
    service = EvaluationService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )

    job = service.create_job(db_session, EvaluationJobCreate(model_id="yolo_11_best", dataset_key="yolo_test"))
    comparison = service.comparison_jobs(db_session, job.id)

    assert job.comparison_id == job.id
    assert [item.id for item in comparison] == [job.id]


def test_evaluation_comparison_returns_sibling_jobs(settings: Settings, db_session: Session):
    storage = Storage(settings)
    storage.ensure()
    service = EvaluationService(
        settings,
        storage,
        ModelRegistry(settings, storage),
        DatasetService(settings, storage),
    )
    jobs = service.create_jobs(
        db_session,
        EvaluationJobBatchCreate(
            model_ids=["yolo_11_best", "unet_inception"],
            dataset_key="yolo_test",
        ),
    )

    comparison = service.comparison_jobs(db_session, jobs[0].id)

    assert {job.id for job in comparison} == {job.id for job in jobs}


def test_evaluation_dataset_discovery_only_lists_editable_test_split(tmp_path: Path, settings: Settings):
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


def test_classification_evaluation_uses_predictor_scores(settings: Settings, db_session: Session):
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

    service = EvaluationService(settings, storage, FakeRegistry(), dataset_service)
    job = service.create_job(
        db_session,
        EvaluationJobCreate(model_id="fake_classifier", dataset_key=f"dataset:{dataset.id}:test"),
    )

    metrics, _artifacts = service._evaluate_classification(db_session, job, datetime.utcnow())

    assert metrics["samples"] == 2
    assert metrics["image"]["overall"]["accuracy"] == 1.0
    assert metrics["classification"]["macro_auc"] == 1.0


@pytest.mark.slow
def test_nlp_training_option_command_and_runner(tmp_path: Path, settings: Settings):
    storage = Storage(settings)
    storage.ensure()
    dataset_service = DatasetService(settings, storage)
    dataset = dataset_service.create_dataset(
        DatasetCreate(
            name="NLP Train",
            task_type="text_classification",
            format="text_folder",
            labels=["positive", "negative"],
        )
    )
    for split, rows in {
        "train": [("good.txt", "good stable workflow", 0), ("bad.txt", "bad broken import", 1)],
        "valid": [("ok.txt", "good clear output", 0)],
    }.items():
        for filename, text, class_id in rows:
            text_path = storage.datasets / dataset.id / split / "texts" / filename
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text, encoding="utf-8")
            dataset_service.save_annotations(
                dataset.id,
                split,
                filename,
                DatasetAnnotationSave(
                    annotations=[
                        DatasetAnnotation(
                            class_id=class_id,
                            class_name="positive" if class_id == 0 else "negative",
                            kind="classification",
                        )
                    ]
                ),
            )
    service = TrainingService(settings, storage, ModelRegistry(settings, storage), dataset_service)
    text_option_ids = [option.id for option in service.model_options("text_classification")]
    assert text_option_ids[:3] == [
        "nlp_keras_cnn_classifier",
        "nlp_keras_lstm_classifier",
        "nlp_keras_bilstm_classifier",
    ]
    assert "nlp_tfidf_classifier" in text_option_ids
    assert "hf_bert_text_classifier" in text_option_ids
    assert service.prepare_model_asset(
        ModelAssetPrepareRequest(option_id="hf_bert_text_classifier", download=False)
    ).status == "missing"
    option = service.option_by_id("nlp_tfidf_classifier")
    assert option.runnable is True
    payload = TrainingJobCreate(
        model_option_id="nlp_tfidf_classifier",
        model_family="nlp_text_classification",
        task_type="text_classification",
        dataset_id=dataset.id,
        epochs=2,
    )
    job = TrainingJob(id="nlp-job", model_family="nlp_text_classification", status="queued", parameters=payload.model_dump())

    command = service._command_for_job(job, tmp_path / "run")
    assert "app.training.runners.nlp_train" in command
    assert "--task-type" in command

    metrics, predictions = train_text_classifier(
        dataset_service.dataset_root(dataset.id),
        tmp_path / "runner",
        ["positive", "negative"],
        epochs=1,
        learning_rate=1.0,
    )
    assert (tmp_path / "runner" / "model.pkl").exists()
    assert "accuracy" in metrics
    assert predictions

    keras_payload = TrainingJobCreate(
        model_option_id="nlp_keras_bilstm_classifier",
        model_family="nlp_keras_bilstm",
        task_type="text_classification",
        dataset_id=dataset.id,
        epochs=1,
        batch_size=1,
    )
    keras_job = TrainingJob(
        id="keras-nlp-job",
        model_family="nlp_keras_bilstm",
        status="queued",
        parameters=keras_payload.model_dump(),
    )
    keras_command = service._command_for_job(keras_job, tmp_path / "keras-run")
    assert "app.training.runners.nlp_train" in keras_command
    assert "nlp_keras_bilstm_classifier" in keras_command
    assert "--max-length" in keras_command


@pytest.mark.slow
def test_keras_nlp_runner_artifacts_and_predictor(tmp_path: Path, settings: Settings):
    storage = Storage(settings)
    storage.ensure()
    dataset_service = DatasetService(settings, storage)
    dataset = dataset_service.create_dataset(
        DatasetCreate(
            name="Keras NLP Train",
            task_type="text_classification",
            format="text_folder",
            labels=["positive", "negative"],
        )
    )
    for split, rows in {
        "train": [
            ("good.txt", "good stable workflow", 0),
            ("clear.txt", "clear helpful output", 0),
            ("bad.txt", "bad broken import", 1),
            ("slow.txt", "slow failed job", 1),
        ],
        "valid": [("ok.txt", "good clear output", 0), ("risk.txt", "broken error path", 1)],
    }.items():
        for filename, text, class_id in rows:
            text_path = storage.datasets / dataset.id / split / "texts" / filename
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text, encoding="utf-8")
            dataset_service.save_annotations(
                dataset.id,
                split,
                filename,
                DatasetAnnotationSave(
                    annotations=[
                        DatasetAnnotation(
                            class_id=class_id,
                            class_name="positive" if class_id == 0 else "negative",
                            kind="classification",
                        )
                    ]
                ),
            )

    run_dirs = {}
    for kind in ["cnn", "lstm", "bilstm"]:
        run_dir = tmp_path / f"keras-{kind}"
        metrics, predictions = train_keras_text_classifier(
            dataset_root=dataset_service.dataset_root(dataset.id),
            run_dir=run_dir,
            labels=["positive", "negative"],
            epochs=1,
            learning_rate=0.001,
            batch_size=1,
            max_length=12,
            vocab_size=128,
            model_kind=kind,
        )
        run_dirs[kind] = run_dir
        assert (run_dir / "best_model.keras").exists()
        assert (run_dir / "last_model.keras").exists()
        assert (run_dir / "tokenizer.json").exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "results.csv").exists()
        assert "accuracy" in metrics
        assert predictions

    registry = ModelRegistry(settings, storage)
    registry.register_model(
        model_id="trained_nlp_keras_cnn_test",
        name="Trained Keras CNN",
        family="nlp_keras_cnn",
        task_type="text_classification",
        paths={"model": run_dirs["cnn"] / "best_model.keras"},
        labels=["positive", "negative"],
    )
    prediction = registry.get_predictor("trained_nlp_keras_cnn_test").predict_text(
        "good helpful workflow",
        InferenceParameters(),
    )
    assert prediction["label"] in {"positive", "negative"}
    assert set(prediction["scores"]) == {"positive", "negative"}


def test_keras_seq2seq_and_hf_catalog(tmp_path: Path, settings: Settings):
    storage = Storage(settings)
    storage.ensure()
    dataset_service = DatasetService(settings, storage)
    dataset = dataset_service.create_dataset(
        DatasetCreate(
            name="Summary Train",
            task_type="summarization",
            format="text_folder",
            labels=["summary"],
        )
    )
    for split, rows in {
        "train": [
            ("one.txt", "The platform trains models from local datasets. It writes metrics.", "The platform trains models."),
            ("two.txt", "The dataset studio stores text files safely. It keeps annotations.", "The dataset studio stores text."),
        ],
        "valid": [
            ("valid.txt", "Testing compares model predictions with references. It records metrics.", "Testing compares predictions."),
        ],
    }.items():
        for filename, text, summary in rows:
            text_path = storage.datasets / dataset.id / split / "texts" / filename
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text, encoding="utf-8")
            dataset_service.save_annotations(
                dataset.id,
                split,
                filename,
                DatasetAnnotationSave(
                    annotations=[
                        DatasetAnnotation(class_id=0, class_name="summary", kind="summary", text=summary)
                    ]
                ),
            )

    service = TrainingService(settings, storage, ModelRegistry(settings, storage), dataset_service)
    summary_option_ids = {option.id for option in service.model_options("summarization")}
    qa_option_ids = {option.id for option in service.model_options("question_answering")}
    assert "nlp_keras_seq2seq_summarizer" in summary_option_ids
    assert "hf_bart_summarizer" in summary_option_ids
    assert "hf_bert_question_answering" in qa_option_ids

    run_dir = tmp_path / "seq2seq"
    metrics, predictions = train_keras_seq2seq_summarizer(
        dataset_root=dataset_service.dataset_root(dataset.id),
        run_dir=run_dir,
        epochs=1,
        learning_rate=0.001,
        batch_size=1,
        max_length=16,
        target_max_length=8,
        vocab_size=160,
    )
    assert (run_dir / "best_model.keras").exists()
    assert (run_dir / "tokenizer.json").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "results.csv").exists()
    assert "model_kind" in metrics
    assert predictions


def test_nlp_reference_models_and_evaluation(settings: Settings, db_session: Session):
    storage = Storage(settings)
    storage.ensure()
    dataset_service = DatasetService(settings, storage)
    registry = ModelRegistry(settings, storage)

    model_ids = {model.id for model in registry.list_models()}
    assert {"keyword_text_classifier", "extractive_summarizer", "keyword_qa"}.issubset(model_ids)
    prediction = registry.get_predictor("keyword_text_classifier").predict_text(
        "The run completed with good stable metrics",
        InferenceParameters(),
    )
    assert prediction["label"] in {"positive", "negative", "neutral"}

    service = EvaluationService(settings, storage, registry, dataset_service)
    summary_datasets = service.list_datasets(task_type="summarization")
    assert summary_datasets
    assert all(dataset.task_type == "summarization" for dataset in summary_datasets)
    assert any(dataset.key == "dataset:sample_summarization:test" for dataset in summary_datasets)

    # Test that sample datasets are also listed when a custom project_id is provided
    custom_project_datasets = service.list_datasets(project_id="custom-project-id", task_type="summarization")
    assert custom_project_datasets
    assert any(dataset.key == "dataset:sample_summarization:test" for dataset in custom_project_datasets)

    job = service.create_job(
        db_session,
        EvaluationJobCreate(
            model_id="keyword_text_classifier",
            dataset_key="dataset:sample_text_classification:test",
        ),
    )
    metrics, artifacts = service._evaluate_nlp(db_session, job, datetime.utcnow(), "text_classification")

    assert metrics["samples"] == 3
    assert "text_classification" in metrics
    assert Path(artifacts["metrics"]).exists()
