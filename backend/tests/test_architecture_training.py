"""Training a visually authored architecture (phase 17, stage C)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.core.database import Base
from app.core.storage import Storage
from app.db.models import Architecture as ArchitectureRow
from app.db.models import TrainingJob
from app.ml.architecture.emit_keras import MODULE_FILENAME
from app.ml.architecture.templates import TEMPLATES
from app.ml.architecture.train_catalog import (
    ARCHITECTURE_FAMILY,
    ARCHITECTURE_ID_KEY,
    ARCHITECTURE_OPTION_ID,
    TEXT_FAMILY,
    TEXT_OPTION_ID,
)
from app.ml.model_registry import ModelRegistry
from app.ml.training_catalog import training_model_options
from app.schemas import TrainingJobCreate
from app.services.training import TrainingService
from app.services.training.service import architecture_id_from


@pytest.fixture
def db(settings: Settings):
    engine = create_engine(settings.sqlalchemy_database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def service(settings: Settings, storage: Storage) -> TrainingService:
    return TrainingService(settings, storage, ModelRegistry(settings, storage), None)


def seed_architecture(db, template_id: str = "small_cnn", name: str = "Test graph") -> str:
    now = datetime.now(UTC).replace(tzinfo=None)
    row = ArchitectureRow(
        id=f"arch_{template_id}",
        project_id=None,
        name=name,
        description=None,
        task_type="classification",
        framework="keras",
        version=2,
        graph=TEMPLATES[template_id].graph.model_dump(),
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.commit()
    return row.id


def make_job(architecture_id: str | None, **overrides) -> TrainingJob:
    hyperparameters = {"seed": 7}
    if architecture_id:
        hyperparameters[ARCHITECTURE_ID_KEY] = architecture_id
    parameters = {
        "epochs": 3,
        "batch_size": 8,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "dataset_id": "reference_yolo",
        "hyperparameters": hyperparameters,
        **overrides,
    }
    return TrainingJob(
        id="job-1", model_family=ARCHITECTURE_FAMILY, status="queued", parameters=parameters
    )


def test_the_option_is_offered_for_classification():
    options = {option.id: option for option in training_model_options("classification")}

    assert ARCHITECTURE_OPTION_ID in options
    assert options[ARCHITECTURE_OPTION_ID].runnable
    assert options[ARCHITECTURE_OPTION_ID].family == ARCHITECTURE_FAMILY


def test_the_option_is_not_offered_for_unsupported_tasks():
    options = {option.id for option in training_model_options("summarization")}

    assert ARCHITECTURE_OPTION_ID not in options
    assert TEXT_OPTION_ID not in options


def test_a_text_option_is_offered_for_text_classification():
    """The studio ships BERT and transformer text-classifier presets, so the
    training form has to offer a runner that can train them."""

    options = {option.id: option for option in training_model_options("text_classification")}

    assert TEXT_OPTION_ID in options
    assert options[TEXT_OPTION_ID].runnable
    assert options[TEXT_OPTION_ID].family == TEXT_FAMILY
    # The image option must not leak into a text task; its pipeline batches JPEGs.
    assert ARCHITECTURE_OPTION_ID not in options


def test_a_text_job_routes_to_the_text_runner(service: TrainingService, db, tmp_path: Path):
    architecture_id = seed_architecture(db, "transformer_text_classifier", "Text graph")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    job = make_job(architecture_id)
    job.model_family = TEXT_FAMILY

    command = service._command_for_job(job, run_dir, db)

    assert "app.training.runners.architecture_text_train" in command
    assert "app.training.runners.architecture_train" not in command
    assert (run_dir / MODULE_FILENAME).exists()


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"hyperparameters": {ARCHITECTURE_ID_KEY: "arch_a"}}, "arch_a"),
        ({ARCHITECTURE_ID_KEY: "arch_b"}, "arch_b"),
        ({"hyperparameters": {}}, ""),
        ({}, ""),
    ],
)
def test_architecture_id_is_read_from_hyperparameters_or_the_top_level(params, expected):
    assert architecture_id_from(params) == expected


def test_command_emits_the_generated_module_into_the_run_dir(
    service: TrainingService, db, tmp_path: Path
):
    architecture_id = seed_architecture(db)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    command = service._command_for_job(make_job(architecture_id), run_dir, db)

    module = run_dir / MODULE_FILENAME
    assert module.exists()
    code = module.read_text(encoding="utf-8")
    assert "def build_model(" in code
    # Emitted from the stored graph at its stored version, so a run is pinned
    # to the architecture as it stood when training started.
    assert "arch_small_cnn, v2" in code

    assert "app.training.runners.architecture_train" in command
    assert str(module) in command
    assert "--epochs" in command and "3" in command


def test_the_routing_key_is_stripped_from_the_advanced_payload(
    service: TrainingService, db, tmp_path: Path
):
    """`architecture_id` is routing, not a hyperparameter the runner should log as ignored."""

    architecture_id = seed_architecture(db)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    command = service._command_for_job(make_job(architecture_id), run_dir, db)
    advanced = json.loads(command[command.index("--advanced") + 1])

    assert advanced == {"seed": 7}


def test_a_job_without_an_architecture_fails_with_a_useful_message(
    service: TrainingService, db, tmp_path: Path
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="no architecture attached"):
        service._command_for_job(make_job(None), run_dir, db)


def test_a_job_pointing_at_a_deleted_architecture_fails_clearly(
    service: TrainingService, db, tmp_path: Path
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="no architecture attached"):
        service._command_for_job(make_job("arch_gone"), run_dir, db)


def test_a_broken_graph_is_rejected_before_the_subprocess_starts(
    service: TrainingService, db, tmp_path: Path
):
    architecture_id = seed_architecture(db)
    row = db.get(ArchitectureRow, architecture_id)
    graph = dict(row.graph)
    graph["edges"] = [*graph["edges"], {"id": "loop", "source": "head", "target": "gap"}]
    row.graph = graph
    db.commit()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="still has errors"):
        service._command_for_job(make_job(architecture_id), run_dir, db)


def test_create_job_rejects_a_missing_architecture(service: TrainingService, db):
    payload = TrainingJobCreate(
        project_id="p1",
        task_type="classification",
        model_option_id=ARCHITECTURE_OPTION_ID,
        dataset_id="reference_yolo",
        epochs=1,
        batch_size=8,
        # Unused by this family — the graph's Input node owns the resolution —
        # but TrainingJobCreate still requires a valid value.
        image_size=128,
        hyperparameters={ARCHITECTURE_ID_KEY: "arch_does_not_exist"},
    )

    with pytest.raises(ValueError, match="Architecture not found"):
        service.create_job(db, payload)


def test_create_job_rejects_a_job_with_no_architecture_selected(service: TrainingService, db):
    payload = TrainingJobCreate(
        project_id="p1",
        task_type="classification",
        model_option_id=ARCHITECTURE_OPTION_ID,
        dataset_id="reference_yolo",
        epochs=1,
        batch_size=8,
        # Unused by this family — the graph's Input node owns the resolution —
        # but TrainingJobCreate still requires a valid value.
        image_size=128,
    )

    with pytest.raises(ValueError, match="Pick an architecture"):
        service.create_job(db, payload)


def test_create_job_accepts_a_real_architecture(service: TrainingService, db):
    architecture_id = seed_architecture(db)
    payload = TrainingJobCreate(
        project_id="p1",
        task_type="classification",
        model_option_id=ARCHITECTURE_OPTION_ID,
        dataset_id="reference_yolo",
        epochs=1,
        batch_size=8,
        # Unused by this family — the graph's Input node owns the resolution —
        # but TrainingJobCreate still requires a valid value.
        image_size=128,
        hyperparameters={ARCHITECTURE_ID_KEY: architecture_id},
    )

    job = service.create_job(db, payload)

    assert job.model_family == ARCHITECTURE_FAMILY
    assert architecture_id_from(job.parameters) == architecture_id


@pytest.mark.slow
def test_a_graph_built_model_trains_and_writes_the_registry_artifacts(tmp_path: Path):
    """End-to-end: emit → import → build → fit → the artifacts promotion expects."""

    import numpy as np
    from PIL import Image

    from app.ml.architecture.emit_keras import emit_module
    from app.ml.architecture.graph import build_graph
    from app.ml.architecture.shapes import infer_shapes
    from app.training.runners import architecture_train

    dataset_root = tmp_path / "dataset"
    labels = ["granuloma", "kista"]
    (dataset_root / "manifest.json").parent.mkdir(parents=True)
    (dataset_root / "manifest.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    for split in ("train", "valid"):
        images = dataset_root / split / "images"
        annotations = dataset_root / split / "annotations"
        images.mkdir(parents=True)
        annotations.mkdir(parents=True)
        for index in range(4):
            Image.fromarray(
                (np.random.rand(64, 64, 3) * 255).astype("uint8")
            ).save(images / f"img_{index}.png")
            (annotations / f"img_{index}.json").write_text(
                json.dumps({"annotations": [{"class_id": index % 2}]}), encoding="utf-8"
            )

    graph = TEMPLATES["small_cnn"].graph
    resolved = build_graph(graph)
    shapes, _params, _issues = infer_shapes(resolved, len(labels))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    module_path = run_dir / MODULE_FILENAME
    module_path.write_text(
        emit_module(resolved, shapes, architecture_name="Small CNN", default_num_classes=len(labels)),
        encoding="utf-8",
    )

    import sys

    argv = sys.argv
    sys.argv = [
        "architecture_train",
        "--run-dir", str(run_dir),
        "--dataset-root", str(dataset_root),
        "--model-file", str(module_path),
        "--epochs", "1",
        "--batch-size", "2",
        "--learning-rate", "0.001",
        "--advanced", json.dumps({"seed": 1}),
    ]
    try:
        architecture_train.main()
    finally:
        sys.argv = argv

    # Exactly the artifact set `_register_training_model` promotes.
    assert (run_dir / "best_model.keras").exists()
    assert (run_dir / "last_model.keras").exists()
    assert (run_dir / "results.csv").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["classes"] == labels
    assert "val_accuracy_final" in metrics
    assert (run_dir / "validation_predictions.json").exists()


@pytest.mark.slow
def test_a_graph_built_text_classifier_trains_and_writes_its_artifacts(tmp_path: Path):
    """The text sibling: the sequence length and vocabulary come off the graph.

    The tokenizer must be capped by the Embedding node's table size, or an
    out-of-range id crashes the first batch — which is the one failure a text
    graph cannot survive.
    """

    from app.ml.architecture.emit_keras import emit_module
    from app.ml.architecture.graph import build_graph
    from app.ml.architecture.shapes import infer_shapes
    from app.training.runners import architecture_text_train

    dataset_root = tmp_path / "dataset"
    labels = ["spam", "ham"]
    (dataset_root / "manifest.json").parent.mkdir(parents=True)
    (dataset_root / "manifest.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    for split in ("train", "valid"):
        texts = dataset_root / split / "texts"
        annotations = dataset_root / split / "annotations"
        texts.mkdir(parents=True)
        annotations.mkdir(parents=True)
        for index in range(4):
            (texts / f"doc_{index}.txt").write_text(
                "buy now cheap offer" if index % 2 == 0 else "meeting notes for monday",
                encoding="utf-8",
            )
            (annotations / f"doc_{index}.json").write_text(
                json.dumps({"annotations": [{"kind": "classification", "class_id": index % 2}]}),
                encoding="utf-8",
            )

    graph = TEMPLATES["transformer_text_classifier"].graph
    resolved = build_graph(graph)
    shapes, _params, _issues = infer_shapes(resolved, len(labels))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    module_path = run_dir / MODULE_FILENAME
    module_path.write_text(
        emit_module(
            resolved,
            shapes,
            architecture_name="Text classifier",
            default_num_classes=len(labels),
        ),
        encoding="utf-8",
    )

    import sys

    argv = sys.argv
    sys.argv = [
        "architecture_text_train",
        "--run-dir", str(run_dir),
        "--dataset-root", str(dataset_root),
        "--model-file", str(module_path),
        "--epochs", "1",
        "--batch-size", "2",
        "--learning-rate", "0.001",
        "--advanced", json.dumps({"seed": 1}),
    ]
    try:
        architecture_text_train.main()
    finally:
        sys.argv = argv

    assert (run_dir / "best_model.keras").exists()
    assert (run_dir / "tokenizer.json").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["classes"] == labels
    # The predictor reads both of these back to tokenize an incoming string the
    # same way the training run did.
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["max_length"] == 128
    assert metadata["vocab_size"] == 20000
