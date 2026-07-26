import asyncio
import io
import json
import pickle
import zipfile
from pathlib import Path

import pytest
from starlette.datastructures import UploadFile

from app.core.config import Settings
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry
from app.services.model_upload import ModelUploadError, ModelUploadService


def _make_settings(tmp_path: Path, **overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "MODELS_DIR": str(tmp_path / "models"),
        "DATASETS_DIR": str(tmp_path / "datasets"),
        "STORAGE_DIR": str(tmp_path / "storage"),
        "DATABASE_URL": f"sqlite:///{tmp_path / 'app.db'}",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _service(tmp_path: Path, **overrides: object) -> tuple[ModelUploadService, Storage, ModelRegistry]:
    settings = _make_settings(tmp_path, **overrides)
    storage = Storage(settings)
    storage.ensure()
    registry = ModelRegistry(settings, storage)
    return ModelUploadService(settings, storage, registry), storage, registry


def _upload(filename: str, data: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(data))


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _pt_bytes() -> bytes:
    # Modern torch .pt files are zip archives; a real zip gives the PK magic.
    return _zip_bytes({"data.pkl": b"weights"})


def _hf_zip() -> bytes:
    return _zip_bytes(
        {
            "model/config.json": json.dumps({"model_type": "llama"}).encode(),
            "model/model.safetensors": b"\x00" * 32,
            "model/tokenizer.json": b"{}",
        }
    )


def _run(coro):
    return asyncio.run(coro)


def test_upload_options_descriptor_shapes(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    options = {opt.family: opt for opt in service.upload_options()}

    assert set(options) == {
        "yolo",
        "keras_classification",
        "unet_inception",
        "sklearn_pipeline",
        "llm_hf",
        "llm_adapter",
        "llm_gguf",
    }
    assert options["sklearn_pipeline"].security_note
    for family in ("llm_hf", "llm_adapter", "llm_gguf"):
        assert options[family].servable is False
        assert options[family].gate_note
    assert options["unet_inception"].kind == "pair"
    assert options["llm_hf"].kind == "zip"


def test_upload_yolo_registers_uploaded_model(tmp_path: Path) -> None:
    service, storage, registry = _service(tmp_path)

    result = _run(
        service.create_upload(
            family="yolo",
            name="My detector",
            files={"weights": _upload("best.pt", _pt_bytes())},
            task_type="object_detection",
            labels="granuloma, kista",
        )
    )

    assert result.validated == "structural"
    model = result.model
    assert model.source == "uploaded"
    assert model.task_type == "object_detection"
    assert model.available is True
    assert model.format == "pt"
    assert model.size_bytes and model.size_bytes > 0
    assert (storage.uploaded_models / model.id).is_dir()
    # Round-trips through a fresh registry read.
    reread = next(m for m in ModelRegistry(service.settings, storage).list_models() if m.id == model.id)
    assert reread.source == "uploaded"
    assert reread.labels == ["granuloma", "kista"]


def test_upload_keras_requires_input_size(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    h5 = b"\x89HDF\r\n\x1a\n" + b"\x00" * 16

    with pytest.raises(ModelUploadError):
        _run(
            service.create_upload(
                family="keras_classification",
                name="Classifier",
                files={"model": _upload("model.h5", h5)},
                labels="a,b",
            )
        )

    result = _run(
        service.create_upload(
            family="keras_classification",
            name="Classifier",
            files={"model": _upload("model.h5", h5)},
            labels="a,b",
            input_size=224,
        )
    )
    assert result.model.artifacts.get("image_size") == 224


def test_upload_unet_pair_missing_member_rejected(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    h5 = b"\x89HDF\r\n\x1a\n" + b"\x00" * 16

    with pytest.raises(ModelUploadError) as exc:
        _run(
            service.create_upload(
                family="unet_inception",
                name="Pair",
                files={"unet": _upload("unet.h5", h5)},
                labels="granuloma",
            )
        )
    assert "Inception" in str(exc.value)


def test_upload_family_file_mismatch_rejected(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)

    with pytest.raises(ModelUploadError) as exc:
        _run(
            service.create_upload(
                family="yolo",
                name="Wrong",
                files={"weights": _upload("model.gguf", b"GGUF" + b"\x00" * 8)},
                task_type="segmentation",
                labels="granuloma",
            )
        )
    assert ".pt" in str(exc.value)


def test_upload_sklearn_pipeline_roundtrip(tmp_path: Path) -> None:
    service, storage, registry = _service(tmp_path)
    payload = pickle.dumps({"kind": "dummy-pipeline"})

    result = _run(
        service.create_upload(
            family="sklearn_pipeline",
            name="Text pipe",
            files={"model": _upload("pipe.pkl", payload)},
            task_type="text_classification",
            labels="positive,negative",
        )
    )
    assert result.model.family == "sklearn_pipeline"
    assert result.model.task_type == "text_classification"


def test_upload_sklearn_rejects_non_pickle(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    with pytest.raises(ModelUploadError):
        _run(
            service.create_upload(
                family="sklearn_pipeline",
                name="Bad",
                files={"model": _upload("pipe.pkl", b"not a pickle at all")},
                task_type="text_classification",
                labels="a,b",
            )
        )


def test_upload_llm_hf_is_registered_but_not_runnable(tmp_path: Path) -> None:
    service, storage, registry = _service(tmp_path)

    result = _run(
        service.create_upload(
            family="llm_hf",
            name="Local base",
            files={"archive": _upload("model.zip", _hf_zip())},
        )
    )
    model = result.model
    assert model.family == "llm_hf"
    assert model.artifacts.get("servable") is False
    assert model.available is True

    with pytest.raises(ValueError) as exc:
        registry.get_predictor(model.id)
    assert "Export it to GGUF" in str(exc.value)


def test_upload_llm_hf_missing_config_rejected(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    bad = _zip_bytes({"model/model.safetensors": b"\x00" * 4})
    with pytest.raises(ModelUploadError):
        _run(
            service.create_upload(
                family="llm_hf",
                name="Bad",
                files={"archive": _upload("model.zip", bad)},
            )
        )


def test_upload_adapter_requires_valid_base(tmp_path: Path) -> None:
    service, storage, registry = _service(tmp_path)
    adapter_zip = _zip_bytes(
        {
            "adapter_config.json": b"{}",
            "adapter_model.safetensors": b"\x00" * 8,
        }
    )

    # Dangling base id is rejected.
    with pytest.raises(ModelUploadError):
        _run(
            service.create_upload(
                family="llm_adapter",
                name="Adapter",
                files={"archive": _upload("adapter.zip", adapter_zip)},
                base_model_id="does-not-exist",
            )
        )

    base = _run(
        service.create_upload(
            family="llm_hf",
            name="Base",
            files={"archive": _upload("model.zip", _hf_zip())},
        )
    )
    result = _run(
        service.create_upload(
            family="llm_adapter",
            name="Adapter",
            files={"archive": _upload("adapter.zip", adapter_zip)},
            base_model_id=base.model.id,
        )
    )
    assert result.model.base_model_id == base.model.id


def test_upload_rejects_zip_slip(tmp_path: Path) -> None:
    service, storage, _ = _service(tmp_path)
    evil = _zip_bytes(
        {
            "config.json": b"{}",
            "../escape.safetensors": b"\x00" * 8,
        }
    )
    with pytest.raises(ModelUploadError):
        _run(
            service.create_upload(
                family="llm_hf",
                name="Evil",
                files={"archive": _upload("model.zip", evil)},
            )
        )
    # No temp or final directory left behind.
    assert not any(storage.uploaded_models.iterdir())


def test_upload_size_cap_rejected_and_cleaned(tmp_path: Path) -> None:
    service, storage, _ = _service(tmp_path, MODEL_UPLOAD_MAX_BYTES=16)

    with pytest.raises(ModelUploadError) as exc:
        _run(
            service.create_upload(
                family="yolo",
                name="Too big",
                files={"weights": _upload("best.pt", _pt_bytes() + b"\x00" * 4096)},
                task_type="segmentation",
                labels="granuloma",
            )
        )
    assert "cap" in str(exc.value).lower()
    assert not any(storage.uploaded_models.iterdir())


def test_upload_failure_leaves_no_orphan_dir(tmp_path: Path) -> None:
    service, storage, _ = _service(tmp_path)
    # Invalid magic bytes fail structural validation mid-materialize.
    with pytest.raises(ModelUploadError):
        _run(
            service.create_upload(
                family="yolo",
                name="Corrupt",
                files={"weights": _upload("best.pt", b"not a zip")},
                task_type="segmentation",
                labels="granuloma",
            )
        )
    assert not any(storage.uploaded_models.iterdir())


def test_uploaded_directory_downloads_as_zip(tmp_path: Path) -> None:
    service, storage, registry = _service(tmp_path)
    result = _run(
        service.create_upload(
            family="llm_hf",
            name="Base",
            files={"archive": _upload("model.zip", _hf_zip())},
        )
    )
    path, filename = registry.model_download(result.model.id)
    assert filename.endswith(".zip")
    assert zipfile.is_zipfile(path)


def test_uploaded_single_file_downloads_unchanged(tmp_path: Path) -> None:
    service, _, registry = _service(tmp_path)
    result = _run(
        service.create_upload(
            family="yolo",
            name="Detector",
            files={"weights": _upload("best.pt", _pt_bytes())},
            task_type="segmentation",
            labels="granuloma",
        )
    )
    path, filename = registry.model_download(result.model.id)
    assert path.suffix == ".pt"
    assert filename.endswith(".pt")


def test_uploaded_model_rename_and_delete(tmp_path: Path) -> None:
    service, storage, registry = _service(tmp_path)
    result = _run(
        service.create_upload(
            family="yolo",
            name="Detector",
            files={"weights": _upload("best.pt", _pt_bytes())},
            task_type="segmentation",
            labels="granuloma",
        )
    )
    model_id = result.model.id
    model_dir = storage.uploaded_models / model_id
    assert model_dir.is_dir()

    renamed = registry.update_model(model_id, name="Renamed detector")
    assert renamed.name == "Renamed detector"
    # Rename must not relocate uploaded weights out of uploaded_models/<id>.
    assert model_dir.is_dir()

    registry.delete_model(model_id)
    assert not model_dir.exists()
    assert model_id not in {m.id for m in registry.list_models()}


def test_duplicate_display_name_warns(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    first = _run(
        service.create_upload(
            family="yolo",
            name="Same name",
            files={"weights": _upload("best.pt", _pt_bytes())},
            task_type="segmentation",
            labels="granuloma",
        )
    )
    assert first.duplicate_name is False
    second = _run(
        service.create_upload(
            family="yolo",
            name="Same name",
            files={"weights": _upload("best.pt", _pt_bytes())},
            task_type="segmentation",
            labels="granuloma",
        )
    )
    assert second.duplicate_name is True
    assert second.warnings
