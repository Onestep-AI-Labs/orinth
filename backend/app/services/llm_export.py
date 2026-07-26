"""LLM export jobs (phase 15): adapter zip, merged safetensors, GGUF.

Exports are manifest-backed jobs — no DB rows. Each export owns a directory
``<model exports root>/<export_id>/`` holding ``manifest.json`` (status, log
tail, artifact info) plus the finished artifact. Work happens in a ``work/``
subdirectory and the artifact reaches its final name with an atomic rename, so
an interrupted export never leaves a partial artifact behind.

Heavy steps (PEFT merge, HF→GGUF conversion, llama.cpp quantization) run as
subprocesses via ``app.ml.llm.export_runner`` and the pinned converter script
(``app.ml.llm.gguf_tools``) — memory isolation for multi-GB merges, and no
AGPL code imported into the API process.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import Executor
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from app.core.config import Settings
from app.core.storage import Storage
from app.ml.llm.gguf_tools import LLAMA_CPP_TAG, ConverterFetchError, ensure_converter
from app.ml.llm.model_paths import (
    ModelPathError,
    adapter_dir,
    hf_model_dir,
    resolve_base_ref,
)
from app.ml.model_registry import ModelRegistry, ModelSpec
from app.schemas import ModelExportFormat, ModelExportStatus

EXPORT_FORMATS: tuple[ModelExportFormat, ...] = (
    "adapter_zip",
    "merged_16bit",
    "gguf_q4_k_m",
    "gguf_q5_k_m",
    "gguf_q8_0",
    "gguf_f16",
)
GGUF_QUANT_BY_FORMAT = {
    "gguf_q4_k_m": "q4_k_m",
    "gguf_q5_k_m": "q5_k_m",
    "gguf_q8_0": "q8_0",
    "gguf_f16": "f16",
}

# Formats each LLM family can produce. `llm_hf` is already merged, so only
# GGUF conversion applies; `llm_gguf` can only be re-quantized.
FORMATS_BY_FAMILY: dict[str, tuple[ModelExportFormat, ...]] = {
    "llm_adapter": EXPORT_FORMATS,
    "llm_hf": ("gguf_q4_k_m", "gguf_q5_k_m", "gguf_q8_0", "gguf_f16"),
    "llm_gguf": ("gguf_q4_k_m", "gguf_q5_k_m", "gguf_q8_0", "gguf_f16"),
}

GEMMA_NOTICE = (
    "This artifact derives from a Gemma model. Gemma is provided under and\n"
    "subject to the Gemma Terms of Use: https://ai.google.dev/gemma/terms\n"
)

_LOG_LIMIT = 100


class ExportError(ValueError):
    """User-facing validation error for export requests."""


class ExportService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        registry: ModelRegistry,
        executor: Executor,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.registry = registry
        self.executor = executor
        # Serializes manifest read-modify-write cycles; export work itself runs
        # on the (single-worker) executor.
        self._lock = threading.Lock()

    # -- public API ---------------------------------------------------------

    def create_export(self, model_id: str, export_format: ModelExportFormat) -> ModelExportStatus:
        spec = self.registry.get_spec(model_id)
        allowed = FORMATS_BY_FAMILY.get(spec.family)
        if allowed is None:
            raise ExportError(
                f"Model family '{spec.family}' has no LLM export formats; exports apply to "
                "llm_adapter, llm_hf, and llm_gguf models."
            )
        if export_format not in allowed:
            raise ExportError(
                f"Format '{export_format}' is not available for a {spec.family} model. "
                f"Available: {', '.join(allowed)}."
            )
        if not spec.available:
            raise ExportError("Model files are missing on disk; cannot export.")
        self._preflight_extras(spec, export_format)
        self._preflight_disk(spec, export_format)

        export_id = uuid4().hex
        export_dir = self._exports_root(spec) / export_id
        export_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": export_id,
            "model_id": model_id,
            "format": export_format,
            "status": "queued",
            "logs": ["Export queued"],
            "current_step": "Queued",
            "created_at": _now_iso(),
        }
        self._write_manifest(export_dir, manifest)
        self.executor.submit(self._run_export, model_id, export_id)
        return ModelExportStatus.model_validate(manifest)

    def list_exports(self, model_id: str) -> list[ModelExportStatus]:
        spec = self.registry.get_spec(model_id)
        root = self._exports_root(spec)
        if not root.exists():
            return []
        statuses = []
        for manifest_path in sorted(root.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            statuses.append(ModelExportStatus.model_validate(manifest))
        statuses.sort(
            key=lambda status: status.created_at.isoformat() if status.created_at else "",
            reverse=True,
        )
        return statuses

    def export_download(self, model_id: str, export_id: str) -> tuple[Path, str]:
        spec = self.registry.get_spec(model_id)
        export_dir = self._exports_root(spec) / export_id
        manifest_path = export_dir / "manifest.json"
        if not manifest_path.exists():
            raise KeyError(export_id)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_name = manifest.get("artifact_name")
        if manifest.get("status") != "completed" or not artifact_name:
            raise FileNotFoundError("Export has no completed artifact to download.")
        artifact = export_dir / artifact_name
        if not artifact.exists():
            raise FileNotFoundError(f"Export artifact is missing on disk: {artifact_name}")
        return artifact, artifact_name

    def reconcile_stale_exports(self) -> None:
        """Fail any export left queued/running by a dead API process (startup hook)."""
        for spec in self.registry.list_specs():
            if spec.family not in FORMATS_BY_FAMILY:
                continue
            root = self._exports_root(spec)
            if not root.exists():
                continue
            for manifest_path in root.glob("*/manifest.json"):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if manifest.get("status") not in {"queued", "running"}:
                    continue
                export_dir = manifest_path.parent
                shutil.rmtree(export_dir / "work", ignore_errors=True)
                manifest["status"] = "failed"
                manifest["error"] = "Backend restarted before the export completed"
                manifest["finished_at"] = _now_iso()
                self._write_manifest(export_dir, manifest)

    # -- job execution ------------------------------------------------------

    def _run_export(self, model_id: str, export_id: str) -> None:
        spec = self.registry.get_spec(model_id)
        export_dir = self._exports_root(spec) / export_id
        work_dir = export_dir / "work"
        self._update_manifest(export_dir, status="running", current_step="Starting")
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            manifest = self._read_manifest(export_dir)
            export_format: ModelExportFormat = manifest["format"]
            if export_format == "adapter_zip":
                artifact = self._export_adapter_zip(spec, export_dir, work_dir)
            elif export_format == "merged_16bit":
                artifact = self._export_merged(spec, export_dir, work_dir)
            else:
                artifact = self._export_gguf(spec, export_dir, work_dir, export_format)
            registered_id = None
            if export_format.startswith("gguf_"):
                registered_id = self._register_gguf(spec, export_dir, export_id, artifact, export_format)
            self._update_manifest(
                export_dir,
                status="completed",
                current_step="Completed",
                artifact_name=artifact.name,
                size_bytes=_size_of(artifact),
                registered_model_id=registered_id,
                finished_at=_now_iso(),
                log="Export completed",
            )
        except Exception as exc:  # noqa: BLE001 — job boundary: every failure lands in the manifest
            self._update_manifest(
                export_dir,
                status="failed",
                current_step="Failed",
                error=str(exc) or exc.__class__.__name__,
                finished_at=_now_iso(),
                log=f"Export failed: {exc}",
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _export_adapter_zip(self, spec: ModelSpec, export_dir: Path, work_dir: Path) -> Path:
        adapter_dir = self._adapter_dir(spec)
        self._log(export_dir, "Zipping adapter directory", step="Zipping adapter")
        tmp_zip = work_dir / "adapter.zip.tmp"
        with ZipFile(tmp_zip, "w", compression=ZIP_DEFLATED) as archive:
            for nested in sorted(adapter_dir.rglob("*")):
                if nested.is_file():
                    archive.write(nested, arcname=nested.relative_to(adapter_dir).as_posix())
        self._copy_license_notices(spec, work_dir)
        for notice in work_dir.glob("LICENSE*"):
            with ZipFile(tmp_zip, "a", compression=ZIP_DEFLATED) as archive:
                archive.write(notice, arcname=notice.name)
        artifact = export_dir / f"{spec.id}-adapter.zip"
        tmp_zip.replace(artifact)
        return artifact

    def _export_merged(self, spec: ModelSpec, export_dir: Path, work_dir: Path) -> Path:
        merged_dir = self._merge_to(spec, export_dir, work_dir)
        self._copy_license_notices(spec, merged_dir)
        self._log(export_dir, "Packaging merged model", step="Packaging")
        tmp_zip = work_dir / "merged.zip.tmp"
        with ZipFile(tmp_zip, "w", compression=ZIP_DEFLATED) as archive:
            for nested in sorted(merged_dir.rglob("*")):
                if nested.is_file():
                    archive.write(nested, arcname=nested.relative_to(merged_dir).as_posix())
        artifact = export_dir / f"{spec.id}-merged-16bit.zip"
        tmp_zip.replace(artifact)
        return artifact

    def _export_gguf(
        self, spec: ModelSpec, export_dir: Path, work_dir: Path, export_format: ModelExportFormat
    ) -> Path:
        quant = GGUF_QUANT_BY_FORMAT[export_format]
        if spec.family == "llm_gguf":
            # Re-quantization: the registered GGUF file is the direct input.
            source_gguf = spec.paths["model"]
        else:
            if spec.family == "llm_adapter":
                hf_dir = self._merge_to(spec, export_dir, work_dir)
            else:
                hf_dir = self._hf_model_dir(spec)
            source_gguf = work_dir / "model-f16.gguf"
            self._convert_to_gguf(spec, export_dir, hf_dir, source_gguf)

        if quant == "f16" and source_gguf.parent == work_dir:
            final_path = export_dir / f"{spec.id}-f16.gguf"
            source_gguf.replace(final_path)
        else:
            self._log(export_dir, f"Quantizing to {quant}", step=f"Quantizing ({quant})")
            quant_tmp = work_dir / f"model-{quant}.gguf.tmp"
            self._run_subprocess(
                export_dir,
                [
                    sys.executable,
                    "-m",
                    "app.ml.llm.export_runner",
                    "quantize",
                    "--input",
                    str(source_gguf),
                    "--output",
                    str(quant_tmp),
                    "--type",
                    quant,
                ],
                step_name="Quantization",
            )
            final_path = export_dir / f"{spec.id}-{quant}.gguf"
            quant_tmp.replace(final_path)
        self._write_gguf_license_notice(spec, export_dir)
        return final_path

    def _merge_to(self, spec: ModelSpec, export_dir: Path, work_dir: Path) -> Path:
        """Merge the adapter into its base in a subprocess; returns the merged HF dir."""
        base_ref = self._base_ref(spec)
        merged_dir = work_dir / "merged"
        self._log(export_dir, f"Merging adapter into base: {base_ref}", step="Merging adapter")
        self._run_subprocess(
            export_dir,
            [
                sys.executable,
                "-m",
                "app.ml.llm.export_runner",
                "merge",
                "--adapter-dir",
                str(self._adapter_dir(spec)),
                "--base-ref",
                base_ref,
                "--out-dir",
                str(merged_dir),
                "--hf-cache-dir",
                str(self.storage.model_assets / "huggingface_cache"),
            ],
            step_name="Merge",
        )
        return merged_dir

    def _convert_to_gguf(
        self, spec: ModelSpec, export_dir: Path, hf_dir: Path, out_path: Path
    ) -> None:
        self._log(
            export_dir,
            f"Converting to GGUF f16 (llama.cpp {LLAMA_CPP_TAG})",
            step="Converting to GGUF",
        )
        try:
            converter = ensure_converter(self.storage.tools)
        except ConverterFetchError as exc:
            raise RuntimeError(str(exc)) from exc
        self._run_subprocess(
            export_dir,
            [
                sys.executable,
                str(converter),
                str(hf_dir),
                "--outfile",
                str(out_path),
                "--outtype",
                "f16",
            ],
            step_name=f"GGUF conversion (llama.cpp {LLAMA_CPP_TAG})",
            extra_env={"NO_LOCAL_GGUF": "1"},
        )

    def _run_subprocess(
        self,
        export_dir: Path,
        command: list[str],
        *,
        step_name: str,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("HF_HOME", str(self.storage.model_assets / "huggingface_home"))
        token = self.settings.huggingface_token
        if token:
            env["HF_TOKEN"] = token
            env["HUGGINGFACE_HUB_TOKEN"] = token
        if extra_env:
            env.update(extra_env)
        process = subprocess.Popen(
            command,
            cwd=self.settings.repo_root / "backend",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        tail: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.strip()
            if not clean:
                continue
            tail.append(clean)
            self._log(export_dir, clean)
        code = process.wait()
        if code != 0:
            tail_text = "\n".join(tail[-12:]) or "(no output)"
            raise RuntimeError(f"{step_name} failed (exit {code}):\n{tail_text}")

    # -- registration and metadata ------------------------------------------

    def _register_gguf(
        self,
        spec: ModelSpec,
        export_dir: Path,
        export_id: str,
        artifact: Path,
        export_format: ModelExportFormat,
    ) -> str:
        quant = GGUF_QUANT_BY_FORMAT[export_format]
        registered_id = f"export_gguf_{export_id[:8]}"
        name = f"{spec.name} GGUF {quant.upper().replace('_', '-')}"
        self.registry.register_model(
            model_id=registered_id,
            name=name,
            family="llm_gguf",
            task_type="llm_finetune",
            paths={"model": artifact},
            labels=[],
            project_id=spec.project_id,
            description=f"GGUF export ({quant}) of {spec.name}.",
            source="trained",
            base_model_id=spec.base_model_id or spec.id,
            format="gguf",
            artifacts={
                "model_dir": str(export_dir),
                "export_id": export_id,
                "source_model_id": spec.id,
                "quantization": quant,
            },
        )
        return registered_id

    # -- path/base resolution ------------------------------------------------

    def _exports_root(self, spec: ModelSpec) -> Path:
        if spec.source == "uploaded":
            return self.storage.uploaded_models / spec.id / "exports"
        model_dir = self.storage.owned_path((spec.artifacts or {}).get("model_dir"))
        if model_dir is None:
            first_path = next(iter(spec.paths.values()), None)
            owned = self.storage.owned_path(first_path) if first_path else None
            if owned is not None:
                base = owned if owned.is_dir() else owned.parent
                try:
                    relative = base.resolve().relative_to(self.storage.trained_models.resolve())
                    model_dir = self.storage.trained_models / relative.parts[0]
                except ValueError:
                    model_dir = None
        if model_dir is None:
            model_dir = self.storage.trained_models / spec.id
        return model_dir / "exports"

    def _adapter_dir(self, spec: ModelSpec) -> Path:
        try:
            return adapter_dir(spec)
        except ModelPathError as exc:
            raise ExportError(str(exc)) from exc

    def _hf_model_dir(self, spec: ModelSpec) -> Path:
        try:
            return hf_model_dir(spec)
        except ModelPathError as exc:
            raise ExportError(str(exc)) from exc

    def _base_ref(self, spec: ModelSpec) -> str:
        try:
            return resolve_base_ref(self.registry, spec)
        except ModelPathError as exc:
            raise ExportError(str(exc)) from exc

    # -- guards --------------------------------------------------------------

    def _preflight_extras(self, spec: ModelSpec, export_format: ModelExportFormat) -> None:
        """Fail fast with an install hint when the optional deps a format needs are absent.

        ``adapter_zip`` is pure zipping and needs nothing. A merge (``merged_16bit``,
        or ``gguf_*`` from an adapter) needs ``peft``; ``gguf_*`` also needs the
        ``gguf`` package and ``llama_cpp`` for conversion/quantization. Checking
        here turns a mid-job subprocess ``ModuleNotFoundError`` traceback into an
        actionable message before the job even starts.
        """
        needs_merge = export_format == "merged_16bit" or (
            export_format.startswith("gguf_") and spec.family == "llm_adapter"
        )
        missing: list[str] = []
        if needs_merge and find_spec("peft") is None:
            missing.append("peft")
        if export_format.startswith("gguf_"):
            if find_spec("gguf") is None:
                missing.append("gguf")
            if find_spec("llama_cpp") is None:
                missing.append("llama-cpp-python")
        if missing:
            raise ExportError(
                f"This export needs the LLM extras ({', '.join(dict.fromkeys(missing))}), which are "
                "not installed. Install them first: `cd backend && uv sync --extra llm`."
            )

    def _preflight_disk(self, spec: ModelSpec, export_format: ModelExportFormat) -> None:
        """Fail before heavy work when free disk clearly cannot hold the result."""
        source_bytes = spec.size_on_disk() or 0
        base_bytes = self._base_size_estimate(spec)
        if export_format == "adapter_zip":
            required = int(source_bytes * 1.2)
        elif export_format == "merged_16bit":
            # Merged fp16 weights plus the zip of the same — and the base
            # download itself may still be missing from the HF cache.
            required = int(base_bytes * 3) if base_bytes else 0
        else:
            factor = 3 if spec.family == "llm_gguf" else 4
            required = int((base_bytes or source_bytes) * factor)
        if not required:
            return
        free = shutil.disk_usage(self.storage.root).free
        if free < required:
            raise ExportError(
                f"Not enough disk space for this export: needs about {_format_bytes(required)}, "
                f"only {_format_bytes(free)} free under {self.storage.root}."
            )

    def _base_size_estimate(self, spec: ModelSpec) -> int:
        if spec.family != "llm_adapter":
            return spec.size_on_disk() or 0
        base_id = (spec.base_model_id or "").strip()
        if base_id:
            try:
                base_spec = self.registry.get_spec(base_id)
                return base_spec.size_on_disk() or 0
            except KeyError:
                pass
            try:
                from app.ml.llm.catalog import LLM_OPTIONS_BY_ID  # noqa: PLC0415

                option = LLM_OPTIONS_BY_ID.get(base_id)
                if option:
                    return int(option.get("approx_download_gb", 0)) * 1024**3
            except ImportError:
                pass
        return 0

    def _copy_license_notices(self, spec: ModelSpec, target_dir: Path) -> None:
        """Base-model licenses ride along with exported artifacts (Gemma terms)."""
        if not self._is_gemma_based(spec):
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "LICENSE_NOTICE.txt").write_text(GEMMA_NOTICE, encoding="utf-8")

    def _write_gguf_license_notice(self, spec: ModelSpec, export_dir: Path) -> None:
        if self._is_gemma_based(spec):
            (export_dir / "LICENSE_NOTICE.txt").write_text(GEMMA_NOTICE, encoding="utf-8")

    def _is_gemma_based(self, spec: ModelSpec) -> bool:
        haystack = " ".join(
            [
                spec.base_model_id or "",
                str((spec.artifacts or {}).get("base_model_ref") or ""),
                spec.id,
            ]
        ).lower()
        return "gemma" in haystack

    # -- manifest persistence -------------------------------------------------

    def _read_manifest(self, export_dir: Path) -> dict[str, Any]:
        return json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))

    def _write_manifest(self, export_dir: Path, manifest: dict[str, Any]) -> None:
        target = export_dir / "manifest.json"
        fd, tmp_name = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=str(export_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest, indent=2))
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _update_manifest(self, export_dir: Path, *, log: str | None = None, **fields: Any) -> None:
        with self._lock:
            manifest = self._read_manifest(export_dir)
            manifest.update({key: value for key, value in fields.items() if value is not None})
            if log:
                logs = list(manifest.get("logs", []))
                logs.append(log)
                manifest["logs"] = logs[-_LOG_LIMIT:]
            self._write_manifest(export_dir, manifest)

    def _log(self, export_dir: Path, message: str, *, step: str | None = None) -> None:
        self._update_manifest(export_dir, log=message, current_step=step)


def _now_iso() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat()


def _size_of(path: Path) -> int:
    if path.is_dir():
        return sum(nested.stat().st_size for nested in path.rglob("*") if nested.is_file())
    return path.stat().st_size


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
