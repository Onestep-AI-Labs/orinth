from pathlib import Path
import json
import threading

from app.core.config import Settings, get_settings
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry, model_storage_dir_name


def test_model_registry_lists_static_models():
    settings = get_settings()
    registry = ModelRegistry(settings, Storage(settings))

    model_ids = {model.id for model in registry.list_models()}

    assert "yolo_11_best" in model_ids
    assert "unet_inception" in model_ids


def test_llm_models_do_not_inherit_medical_default_labels(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    registry = ModelRegistry(settings, storage)
    adapter_dir = storage.trained_models / "llm" / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    info = registry.register_model(
        model_id="trained_llm",
        name="Trained LLM",
        family="llm_adapter",
        task_type="llm_finetune",
        paths={"model": adapter_dir},
        labels=[],
    )
    # A label-free LLM must not render the medical DEFAULT_LABELS on its card.
    assert info.labels == []


def test_model_registry_renames_and_deletes_trained_models(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    registry = ModelRegistry(settings, storage)
    model_id = "trained_yolo_example"
    model_dir = storage.trained_models / model_id / "weights"
    model_dir.mkdir(parents=True)
    weights_path = model_dir / "best.pt"
    weights_path.write_text("weights", encoding="utf-8")

    registry.register_model(
        model_id=model_id,
        name="Original model",
        family="yolo",
        task_type="segmentation",
        paths={"weights": weights_path},
        labels=["granuloma", "kista"],
    )

    renamed = registry.update_model(model_id, name="Renamed model")
    renamed_dir = storage.trained_models / model_storage_dir_name("Renamed model", model_id)
    assert renamed.name == "Renamed model"
    assert renamed_dir.exists()
    assert renamed.paths["weights"].startswith(str(renamed_dir))

    registry.delete_model(model_id)

    assert model_id not in {model.id for model in registry.list_models()}
    assert not (storage.trained_models / model_id).exists()
    assert not renamed_dir.exists()


def test_model_registry_stores_trained_model_paths_relative_to_storage(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    registry = ModelRegistry(settings, storage)
    model_id = "trained_yolo_relative"
    weights_path = storage.trained_models / model_id / "weights" / "best.pt"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_text("weights", encoding="utf-8")

    registry.register_model(
        model_id=model_id,
        name="Relative model",
        family="yolo",
        task_type="segmentation",
        paths={"weights": weights_path},
        labels=["granuloma", "kista"],
    )

    registry_payload = json.loads(storage.registry_file.read_text(encoding="utf-8"))
    stored_path = registry_payload[model_id]["paths"]["weights"]
    listed_model = next(model for model in registry.list_models() if model.id == model_id)

    assert stored_path == f"trained_models/{model_id}/weights/best.pt"
    assert listed_model.available is True
    assert listed_model.paths["weights"] == str(weights_path)


def test_model_registry_filters_models_to_exact_project(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    registry = ModelRegistry(settings, storage)
    project_a_model = "trained_yolo_project_a"
    project_b_model = "trained_yolo_project_b"
    for model_id in [project_a_model, project_b_model]:
        weights_path = storage.trained_models / model_id / "weights" / "best.pt"
        weights_path.parent.mkdir(parents=True)
        weights_path.write_text("weights", encoding="utf-8")
        registry.register_model(
            model_id=model_id,
            name=model_id,
            family="yolo",
            task_type="segmentation",
            paths={"weights": weights_path},
            labels=["granuloma", "kista"],
            project_id="project-a" if model_id == project_a_model else "project-b",
        )

    project_a_ids = {model.id for model in registry.list_models(project_id="project-a")}
    project_b_ids = {model.id for model in registry.list_models(project_id="project-b")}

    nlp_reference_ids = {"keyword_text_classifier", "extractive_summarizer", "keyword_qa"}
    assert project_a_ids == {project_a_model, *nlp_reference_ids}
    assert project_b_ids == {project_b_model, *nlp_reference_ids}
    assert "yolo_11_best" not in project_a_ids


def test_model_registry_recovers_trained_model_paths_after_repo_move(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "new-repo" / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    model_id = "trained_keras_classification_moved"
    model_path = storage.trained_models / model_id / "best_model.keras"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("keras", encoding="utf-8")
    old_repo_path = tmp_path / "old-repo" / "storage" / "trained_models" / model_id / "best_model.keras"
    storage.registry_file.write_text(
        json.dumps(
            {
                model_id: {
                    "id": model_id,
                    "name": "Moved model",
                    "family": "keras_classification",
                    "description": "Moved trained model",
                    "paths": {"model": str(old_repo_path)},
                    "promoted": True,
                    "project_id": "default-research-project",
                    "task_type": "classification",
                    "labels": ["trash"],
                    "source": "trained",
                }
            }
        ),
        encoding="utf-8",
    )

    listed_model = next(model for model in ModelRegistry(settings, storage).list_models() if model.id == model_id)

    assert listed_model.available is True
    assert listed_model.paths["model"] == str(model_path)


def test_model_storage_dir_name_uses_name_and_id():
    assert (
        model_storage_dir_name("My Keras Model!", "trained_keras_classification_abcd1234")
        == "my-keras-model-trained_keras_classification_abcd1234"
    )


def test_model_registry_write_registry_is_atomic_under_concurrent_writes(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    registry = ModelRegistry(settings, storage)

    stop = threading.Event()
    errors: list[Exception] = []

    def writer(worker_id: int) -> None:
        counter = 0
        while not stop.is_set():
            try:
                registry._write_registry(
                    {f"model_{worker_id}_{counter}": {"id": f"model_{worker_id}_{counter}"}}
                )
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)
                return
            counter += 1

    def reader() -> None:
        while not stop.is_set():
            if storage.registry_file.exists():
                try:
                    raw = storage.registry_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                if raw:
                    try:
                        json.loads(raw)
                    except json.JSONDecodeError as exc:  # pragma: no cover - failure path
                        errors.append(exc)
                        return

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    threads.append(threading.Thread(target=reader))
    threads.append(threading.Thread(target=reader))

    for thread in threads:
        thread.start()
    stop.wait(0.5)
    stop.set()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert json.loads(storage.registry_file.read_text(encoding="utf-8")) is not None


def test_model_registry_concurrent_register_update_delete_preserves_entries(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    registry = ModelRegistry(settings, storage)

    model_ids = [f"trained_concurrent_{i}" for i in range(8)]
    for model_id in model_ids:
        weights_path = storage.trained_models / model_id / "weights" / "best.pt"
        weights_path.parent.mkdir(parents=True)
        weights_path.write_text("weights", encoding="utf-8")

    errors: list[Exception] = []

    def register(model_id: str) -> None:
        try:
            registry.register_model(
                model_id=model_id,
                name=f"Model {model_id}",
                family="yolo",
                task_type="segmentation",
                paths={"weights": storage.trained_models / model_id / "weights" / "best.pt"},
                labels=["granuloma", "kista"],
            )
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=register, args=(model_id,)) for model_id in model_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    registry_payload = json.loads(storage.registry_file.read_text(encoding="utf-8"))
    assert set(registry_payload.keys()) == set(model_ids)

    def update(model_id: str) -> None:
        try:
            registry.update_model(model_id, name=f"Updated {model_id}")
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def delete(model_id: str) -> None:
        try:
            registry.delete_model(model_id)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    to_update = model_ids[: len(model_ids) // 2]
    to_delete = model_ids[len(model_ids) // 2 :]

    threads = [threading.Thread(target=update, args=(model_id,)) for model_id in to_update]
    threads += [threading.Thread(target=delete, args=(model_id,)) for model_id in to_delete]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    registry_payload = json.loads(storage.registry_file.read_text(encoding="utf-8"))
    assert set(registry_payload.keys()) == set(to_update)
    for model_id in to_update:
        assert registry_payload[model_id]["name"] == f"Updated {model_id}"


def test_model_registry_get_predictor_double_checked_locking_is_thread_safe():
    settings = get_settings()
    registry = ModelRegistry(settings, Storage(settings))

    construct_calls: list[str] = []
    construct_started = threading.Event()
    release_construct = threading.Event()

    def fake_construct(spec):
        construct_calls.append(spec.id)
        construct_started.set()
        assert release_construct.wait(timeout=5)
        return object()

    registry._construct_predictor = fake_construct  # type: ignore[method-assign]

    results: list[object] = []
    errors: list[Exception] = []

    def call() -> None:
        try:
            results.append(registry.get_predictor("keyword_text_classifier"))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(5)]
    for thread in threads:
        thread.start()

    assert construct_started.wait(timeout=5)

    # While construction is blocked (simulating heavy model loading), the
    # registry lock must be free -- construction must happen outside it.
    acquired = registry._lock.acquire(timeout=1)
    assert acquired, "registry lock was held during predictor construction"
    registry._lock.release()

    release_construct.set()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert len(results) == 5
    assert all(result is results[0] for result in results)
    assert registry._predictors["keyword_text_classifier"] is results[0]


def test_model_registry_creates_download_archives_for_multi_asset_models(tmp_path: Path):
    settings = Settings(
        MODELS_DIR=str(tmp_path / "models"),
        DATASETS_DIR=str(tmp_path / "datasets"),
        STORAGE_DIR=str(tmp_path / "storage"),
        DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
    )
    storage = Storage(settings)
    storage.ensure()
    model_dir = settings.models_path / "unet_inception"
    model_dir.mkdir(parents=True)
    (model_dir / "best_unet_model.keras").write_text("unet", encoding="utf-8")
    (model_dir / "best_classifier_inception.keras").write_text("classifier", encoding="utf-8")

    archive_path, filename = ModelRegistry(settings, storage).model_download("unet_inception")

    assert filename.endswith(".zip")
    assert archive_path.exists()
    assert archive_path.parent == storage.model_downloads
