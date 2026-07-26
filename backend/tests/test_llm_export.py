"""Phase 15 LLM export: format validation, pipeline orchestration with the
subprocess layer mocked, GGUF registration, disk preflight, atomicity, and
stale-manifest reconciliation."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from app.ml.model_registry import ModelRegistry
from app.services.llm_export import ExportError, ExportService


class InlineExecutor:
    """Runs submitted jobs synchronously so tests observe final manifests."""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


@pytest.fixture
def registry(settings, storage) -> ModelRegistry:
    return ModelRegistry(settings, storage)


@pytest.fixture
def export_service(settings, storage, registry) -> ExportService:
    return ExportService(settings, storage, registry, InlineExecutor())


def register_adapter(registry: ModelRegistry, storage, *, base_model_id="qwen2_5_1_5b"):
    model_dir = storage.trained_models / "my-adapter-adp1"
    adapter_dir = model_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"}), encoding="utf-8"
    )
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return registry.register_model(
        model_id="adp1",
        name="My Adapter",
        family="llm_adapter",
        task_type="llm_finetune",
        paths={"model": adapter_dir},
        labels=[],
        base_model_id=base_model_id,
        format="safetensors",
        artifacts={"model_dir": str(model_dir), "base_model_ref": "Qwen/Qwen2.5-1.5B-Instruct"},
    )


def register_gguf(registry: ModelRegistry, storage, model_id="gguf1"):
    model_dir = storage.trained_models / f"served-{model_id}"
    model_dir.mkdir(parents=True)
    gguf_path = model_dir / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    return registry.register_model(
        model_id=model_id,
        name="Served GGUF",
        family="llm_gguf",
        task_type="llm_finetune",
        paths={"model": gguf_path},
        labels=[],
        format="gguf",
        artifacts={"model_dir": str(model_dir)},
    )


# -- format validation -------------------------------------------------------


def test_export_rejects_non_llm_family(export_service, registry, storage, settings):
    weights = settings.models_path / "yolo_11_best" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"x")
    with pytest.raises(ExportError, match="no LLM export formats"):
        export_service.create_export("yolo_11_best", "gguf_q4_k_m")


def test_export_rejects_format_family_mismatch(export_service, registry, storage):
    register_gguf(registry, storage)
    with pytest.raises(ExportError, match="not available"):
        export_service.create_export("gguf1", "adapter_zip")


def test_export_rejects_missing_files(export_service, registry, storage):
    register_adapter(registry, storage)
    adapter_dir = storage.trained_models / "my-adapter-adp1" / "adapter"
    for path in adapter_dir.iterdir():
        path.unlink()
    adapter_dir.rmdir()
    with pytest.raises(ExportError, match="missing on disk"):
        export_service.create_export("adp1", "adapter_zip")


# -- adapter zip (runs for real, no subprocess) ------------------------------


def test_adapter_zip_export_completes(export_service, registry, storage):
    register_adapter(registry, storage)
    created = export_service.create_export("adp1", "adapter_zip")

    statuses = export_service.list_exports("adp1")
    assert [status.id for status in statuses] == [created.id]
    status = statuses[0]
    assert status.status == "completed"
    assert status.artifact_name == "adp1-adapter.zip"
    assert status.size_bytes and status.size_bytes > 0

    artifact, filename = export_service.export_download("adp1", created.id)
    assert filename == "adp1-adapter.zip"
    with ZipFile(artifact) as archive:
        names = set(archive.namelist())
    assert "adapter_config.json" in names
    assert "adapter_model.safetensors" in names
    # work dir is cleaned after completion
    assert not (artifact.parent / "work").exists()


def test_gemma_license_notice_rides_along(export_service, registry, storage):
    model_dir = storage.trained_models / "gem-adp2"
    adapter_dir = model_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"w")
    registry.register_model(
        model_id="adp2",
        name="Gemma Tune",
        family="llm_adapter",
        task_type="llm_finetune",
        paths={"model": adapter_dir},
        labels=[],
        base_model_id="gemma3_1b",
        artifacts={"model_dir": str(model_dir), "base_model_ref": "google/gemma-3-1b-it"},
    )
    created = export_service.create_export("adp2", "adapter_zip")
    artifact, _ = export_service.export_download("adp2", created.id)
    with ZipFile(artifact) as archive:
        assert "LICENSE_NOTICE.txt" in archive.namelist()


# -- gguf pipeline (subprocess-mocked) ---------------------------------------


def fake_subprocess_runner(calls):
    def _fake(self, export_dir, command, *, step_name, extra_env=None):
        calls.append((step_name, command))
        if "merge" in command:
            out_dir = Path(command[command.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "config.json").write_text("{}", encoding="utf-8")
            (out_dir / "model.safetensors").write_bytes(b"merged")
        elif "quantize" in command:
            Path(command[command.index("--output") + 1]).write_bytes(b"GGUFq")
        elif any(str(part).endswith("convert_hf_to_gguf.py") for part in command):
            Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUFf16")
        else:  # pragma: no cover - unexpected command shape
            raise AssertionError(f"unexpected command: {command}")

    return _fake


def test_gguf_export_orchestrates_merge_convert_quantize(
    export_service, registry, storage, monkeypatch
):
    register_adapter(registry, storage)
    calls = []
    monkeypatch.setattr(ExportService, "_run_subprocess", fake_subprocess_runner(calls))
    monkeypatch.setattr(
        "app.services.llm_export.ensure_converter",
        lambda tools_root: tools_root / "llama.cpp" / "test" / "convert_hf_to_gguf.py",
    )

    created = export_service.create_export("adp1", "gguf_q4_k_m")
    status = export_service.list_exports("adp1")[0]
    assert status.status == "completed", status.error
    assert status.artifact_name == "adp1-q4_k_m.gguf"
    assert [name for name, _ in calls] == [
        "Merge",
        "GGUF conversion (llama.cpp b9000)",
        "Quantization",
    ]

    # The completed GGUF registers as a servable llm_gguf model.
    assert status.registered_model_id
    spec = registry.get_spec(status.registered_model_id)
    assert spec.family == "llm_gguf"
    assert spec.source == "trained"
    assert spec.base_model_id == "qwen2_5_1_5b"
    assert spec.available
    artifact, _ = export_service.export_download("adp1", created.id)
    assert spec.paths["model"] == artifact


def test_gguf_export_from_hf_model_skips_merge(export_service, registry, storage, monkeypatch):
    model_dir = storage.uploaded_models / "hf1"
    hf_dir = model_dir / "model"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_bytes(b"full")
    registry.register_model(
        model_id="hf1",
        name="Uploaded HF",
        family="llm_hf",
        task_type="llm_finetune",
        paths={"model": hf_dir},
        labels=[],
        source="uploaded",
    )
    calls = []
    monkeypatch.setattr(ExportService, "_run_subprocess", fake_subprocess_runner(calls))
    monkeypatch.setattr(
        "app.services.llm_export.ensure_converter",
        lambda tools_root: tools_root / "llama.cpp" / "test" / "convert_hf_to_gguf.py",
    )
    export_service.create_export("hf1", "gguf_f16")
    status = export_service.list_exports("hf1")[0]
    assert status.status == "completed", status.error
    assert [name for name, _ in calls] == ["GGUF conversion (llama.cpp b9000)"]
    # Uploaded models keep exports under storage/uploaded_models/<id>/exports/.
    assert (model_dir / "exports" / status.id / status.artifact_name).exists()


def test_failed_step_leaves_no_artifact(export_service, registry, storage, monkeypatch):
    register_adapter(registry, storage)

    def failing(self, export_dir, command, *, step_name, extra_env=None):
        raise RuntimeError(f"{step_name} failed (exit 1):\nboom")

    monkeypatch.setattr(ExportService, "_run_subprocess", failing)
    created = export_service.create_export("adp1", "merged_16bit")
    status = export_service.list_exports("adp1")[0]
    assert status.status == "failed"
    assert "boom" in (status.error or "")
    export_dir = storage.trained_models / "my-adapter-adp1" / "exports" / created.id
    assert not (export_dir / "work").exists()
    assert not list(export_dir.glob("*.zip"))
    with pytest.raises(FileNotFoundError):
        export_service.export_download("adp1", created.id)


def test_export_preflight_names_missing_extras(export_service, registry, storage, monkeypatch):
    register_adapter(registry, storage)
    # Simulate peft not installed: the merge path must refuse with an install hint
    # instead of failing mid-job with a ModuleNotFoundError traceback.
    real_find_spec = __import__("importlib.util", fromlist=["find_spec"]).find_spec
    monkeypatch.setattr(
        "app.services.llm_export.find_spec",
        lambda name: None if name == "peft" else real_find_spec(name),
    )
    with pytest.raises(ExportError, match="uv sync --extra llm"):
        export_service.create_export("adp1", "merged_16bit")


def test_adapter_zip_needs_no_extras(export_service, registry, storage, monkeypatch):
    register_adapter(registry, storage)
    # adapter_zip is pure zipping; a missing peft must not block it.
    monkeypatch.setattr("app.services.llm_export.find_spec", lambda name: None)
    created = export_service.create_export("adp1", "adapter_zip")
    assert export_service.list_exports("adp1")[0].status == "completed"
    assert created.id


def test_disk_preflight_names_sizes(export_service, registry, storage, monkeypatch):
    register_adapter(registry, storage)

    class FakeUsage:
        free = 1024

    monkeypatch.setattr("app.services.llm_export.shutil.disk_usage", lambda path: FakeUsage)
    with pytest.raises(ExportError, match="Not enough disk space"):
        export_service.create_export("adp1", "gguf_q4_k_m")


def test_reconcile_marks_stale_exports_failed(export_service, registry, storage, monkeypatch):
    register_adapter(registry, storage)
    # Queue without running: executor that drops the job (simulates a dead process).
    export_service.executor = type("DropExecutor", (), {"submit": lambda self, fn, *a, **k: None})()
    created = export_service.create_export("adp1", "adapter_zip")
    assert export_service.list_exports("adp1")[0].status == "queued"

    export_service.reconcile_stale_exports()
    status = export_service.list_exports("adp1")[0]
    assert status.status == "failed"
    assert "restarted" in (status.error or "")
    assert created.id == status.id
