"""Custom model upload: dynamic descriptor + structural validation (phase 12).

Validation is deliberately two-tier. Upload time runs synchronous *structural*
checks only — extension + magic bytes + archive member listing. Deep validation
(constructing the predictor) stays deferred to first use, exactly like every
other registry family, preserving the platform's lazy-loading rule.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from fastapi import UploadFile

from app.core.config import Settings
from app.core.defaults import DEFAULT_PROJECT_ID
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry
from app.schemas import (
    ModelUploadField,
    ModelUploadFileSlot,
    ModelUploadOption,
    ModelUploadResult,
)

_CHUNK = 1024 * 1024

_LLM_GATE = "Not directly servable — export to GGUF on the model detail page, then serve the GGUF."
_GGUF_GATE = "Serves through the managed llama.cpp chat runtime (model detail page → Serving)."
_SKLEARN_SECURITY = (
    "Loading a scikit-learn pickle executes arbitrary code by design. Only "
    "upload files you trust — this is safe solely under the platform's "
    "single-user local-trust posture."
)


class ModelUploadError(ValueError):
    """Structural validation failure — surfaced to the client as a 400."""


@dataclass
class _FamilySpec:
    family: str
    label: str
    description: str
    kind: Literal["single", "pair", "zip"]
    files: list[ModelUploadFileSlot]
    fields: list[ModelUploadField] = field(default_factory=list)
    fixed_task_type: str | None = None
    task_options: list[str] | None = None
    needs_labels: bool = False
    needs_input_size: bool = False
    needs_base_model: bool = False
    servable: bool = True
    gate_note: str | None = None
    security_note: str | None = None
    format: str | None = None


def _labels_field(required: bool) -> ModelUploadField:
    return ModelUploadField(
        key="labels",
        label="Labels",
        type="labels",
        required=required,
        placeholder="granuloma, kista",
        help="Comma-separated class labels in output order.",
    )


def _build_family_specs() -> list[_FamilySpec]:
    return [
        _FamilySpec(
            family="yolo",
            label="YOLO weights",
            description="Ultralytics YOLO detection/segmentation weights (.pt).",
            kind="single",
            files=[ModelUploadFileSlot(key="weights", label="Weights (.pt)", accept=[".pt"])],
            fields=[
                ModelUploadField(
                    key="task_type",
                    label="Task type",
                    type="select",
                    required=True,
                    options=["object_detection", "segmentation"],
                ),
                _labels_field(True),
            ],
            task_options=["object_detection", "segmentation"],
            needs_labels=True,
            format="pt",
        ),
        _FamilySpec(
            family="keras_classification",
            label="Keras classifier",
            description="Keras image classification model (.h5 or .keras).",
            kind="single",
            files=[
                ModelUploadFileSlot(
                    key="model", label="Model (.h5 / .keras)", accept=[".h5", ".keras"]
                )
            ],
            fields=[
                _labels_field(True),
                ModelUploadField(
                    key="input_size",
                    label="Input size (px)",
                    type="number",
                    required=True,
                    placeholder="224",
                    help="Square input edge the model expects.",
                ),
            ],
            fixed_task_type="classification",
            needs_labels=True,
            needs_input_size=True,
            format="keras",
        ),
        _FamilySpec(
            family="unet_inception",
            label="U-Net + Inception",
            description="Two-stage U-Net segmentation + Inception classifier pair.",
            kind="pair",
            files=[
                ModelUploadFileSlot(key="unet", label="U-Net (.h5 / .keras)", accept=[".h5", ".keras"]),
                ModelUploadFileSlot(
                    key="classifier", label="Inception (.h5 / .keras)", accept=[".h5", ".keras"]
                ),
            ],
            fields=[_labels_field(False)],
            fixed_task_type="segmentation",
            format="keras",
        ),
        _FamilySpec(
            family="sklearn_pipeline",
            label="scikit-learn pipeline",
            description="Pickled/joblib sklearn text pipeline (predict on raw text).",
            kind="single",
            files=[
                ModelUploadFileSlot(
                    key="model", label="Pipeline (.pkl / .joblib)", accept=[".pkl", ".joblib"]
                )
            ],
            fields=[
                ModelUploadField(
                    key="task_type",
                    label="Text task",
                    type="select",
                    required=True,
                    options=["text_classification", "summarization", "question_answering"],
                ),
                _labels_field(True),
            ],
            task_options=["text_classification", "summarization", "question_answering"],
            needs_labels=True,
            security_note=_SKLEARN_SECURITY,
            format="joblib",
        ),
        _FamilySpec(
            family="llm_hf",
            label="Hugging Face model",
            description="Zip of an HF model directory (config.json + safetensors + tokenizer).",
            kind="zip",
            files=[ModelUploadFileSlot(key="archive", label="Model directory (.zip)", accept=[".zip"])],
            fixed_task_type="llm_finetune",
            servable=False,
            gate_note=_LLM_GATE,
            format="safetensors",
        ),
        _FamilySpec(
            family="llm_adapter",
            label="LoRA adapter",
            description="Zip with adapter_config.json + adapter weights, bound to a base model.",
            kind="zip",
            files=[ModelUploadFileSlot(key="archive", label="Adapter (.zip)", accept=[".zip"])],
            fields=[
                ModelUploadField(
                    key="base_model_id",
                    label="Base model",
                    type="text",
                    required=True,
                    help="Id of a registered HF model, or a phase-14 catalog base id.",
                )
            ],
            fixed_task_type="llm_finetune",
            needs_base_model=True,
            servable=False,
            gate_note=_LLM_GATE,
            format="safetensors",
        ),
        _FamilySpec(
            family="llm_gguf",
            label="GGUF model",
            description="Single-file GGUF quantized model (.gguf).",
            kind="single",
            files=[ModelUploadFileSlot(key="model", label="Model (.gguf)", accept=[".gguf"])],
            fixed_task_type="llm_finetune",
            servable=False,
            gate_note=_GGUF_GATE,
            format="gguf",
        ),
    ]


class _Budget:
    """Shared byte budget across every file/entry of one upload."""

    def __init__(self, cap: int) -> None:
        self.remaining = cap
        self.cap = cap

    def take(self, size: int) -> None:
        self.remaining -= size
        if self.remaining < 0:
            raise ModelUploadError(
                f"Upload exceeds the {self.cap} byte cap. Reduce the file size or "
                "raise MODEL_UPLOAD_MAX_BYTES."
            )


class ModelUploadService:
    def __init__(self, settings: Settings, storage: Storage, registry: ModelRegistry) -> None:
        self.settings = settings
        self.storage = storage
        self.registry = registry
        self._specs = {spec.family: spec for spec in _build_family_specs()}

    # -- descriptor ---------------------------------------------------------

    def upload_options(self) -> list[ModelUploadOption]:
        return [
            ModelUploadOption(
                family=spec.family,
                label=spec.label,
                description=spec.description,
                kind=spec.kind,
                files=spec.files,
                fields=spec.fields,
                servable=spec.servable,
                gate_note=spec.gate_note,
                security_note=spec.security_note,
            )
            for spec in self._specs.values()
        ]

    # -- upload -------------------------------------------------------------

    async def create_upload(
        self,
        *,
        family: str,
        name: str,
        files: dict[str, UploadFile],
        project_id: str | None = None,
        task_type: str | None = None,
        labels: str | None = None,
        input_size: int | None = None,
        base_model_id: str | None = None,
    ) -> ModelUploadResult:
        spec = self._specs.get(family)
        if spec is None:
            raise ModelUploadError(f"Unknown model family: {family}")

        clean_name = (name or "").strip()
        if not clean_name:
            raise ModelUploadError("Model name is required.")

        resolved_task = self._resolve_task_type(spec, task_type)
        parsed_labels = self._parse_labels(labels)
        if spec.needs_labels and not parsed_labels:
            raise ModelUploadError("At least one label is required for this family.")
        resolved_input = self._resolve_input_size(spec, input_size)
        resolved_base = self._resolve_base_model(spec, base_model_id)

        self._check_required_files(spec, files)

        model_id = f"uploaded_{family}_{uuid4().hex[:12]}"
        tmp_dir = self.storage.uploaded_models / f".tmp-{model_id}"
        budget = _Budget(self.settings.model_upload_max_bytes)

        try:
            self.storage.uploaded_models.mkdir(parents=True, exist_ok=True)
            tmp_dir.mkdir(parents=True)
            warnings: list[str] = []
            paths = await self._materialize(spec, files, tmp_dir, budget, warnings)

            final_dir = self.storage.uploaded_models / model_id
            if final_dir.exists():
                shutil.rmtree(final_dir)
            tmp_dir.rename(final_dir)
            registered_paths = {
                key: final_dir / value.relative_to(tmp_dir) for key, value in paths.items()
            }
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        artifacts: dict[str, object] = {"validated": "structural"}
        if resolved_input is not None:
            artifacts["image_size"] = resolved_input
        if not spec.servable:
            artifacts["servable"] = False
            artifacts["gate_note"] = spec.gate_note

        model = self.registry.register_model(
            model_id=model_id,
            name=clean_name,
            family=family,
            task_type=resolved_task,
            paths=registered_paths,
            labels=parsed_labels or [],
            project_id=project_id or DEFAULT_PROJECT_ID,
            description=f"Uploaded {spec.label}.",
            source="uploaded",
            base_model_id=resolved_base,
            format=spec.format,
            artifacts=artifacts,
        )

        duplicate = any(
            other.name == clean_name and other.id != model_id
            for other in self.registry.list_specs()
        )
        if duplicate:
            warnings.append(
                "Another model already uses this display name; ids stay unique, so rename if needed."
            )

        return ModelUploadResult(model=model, duplicate_name=duplicate, warnings=warnings)

    # -- field resolution ---------------------------------------------------

    def _resolve_task_type(self, spec: _FamilySpec, task_type: str | None) -> str:
        if spec.fixed_task_type is not None:
            return spec.fixed_task_type
        options = spec.task_options or []
        chosen = (task_type or "").strip()
        if chosen not in options:
            raise ModelUploadError(
                f"task_type must be one of {options} for {spec.family}."
            )
        return chosen

    def _parse_labels(self, labels: str | None) -> list[str]:
        if not labels:
            return []
        return [part.strip() for part in labels.split(",") if part.strip()]

    def _resolve_input_size(self, spec: _FamilySpec, input_size: int | None) -> int | None:
        if not spec.needs_input_size:
            return None
        if input_size is None or input_size <= 0:
            raise ModelUploadError("A positive input size is required for this family.")
        return int(input_size)

    def _resolve_base_model(self, spec: _FamilySpec, base_model_id: str | None) -> str | None:
        if not spec.needs_base_model:
            return None
        clean = (base_model_id or "").strip()
        if not clean:
            raise ModelUploadError("base_model_id is required for adapter uploads.")
        known_hf = {
            item.id for item in self.registry.list_specs() if item.family == "llm_hf"
        }
        if clean in known_hf or clean in self._catalog_base_ids():
            return clean
        raise ModelUploadError(
            f"base_model_id '{clean}' does not reference a registered HF model or a known base."
        )

    def _catalog_base_ids(self) -> set[str]:
        # Phase 14 introduces the LLM catalog; accept its ids if present. Until
        # then only registered llm_hf models qualify as adapter bases.
        try:
            from app.ml.llm.catalog import catalog_base_ids

            return set(catalog_base_ids())
        except Exception:  # noqa: BLE001 — catalog is optional in phase 12
            return set()

    # -- file handling ------------------------------------------------------

    def _check_required_files(self, spec: _FamilySpec, files: dict[str, UploadFile]) -> None:
        for slot in spec.files:
            provided = files.get(slot.key)
            if slot.required and (provided is None or not provided.filename):
                raise ModelUploadError(f"Missing required file: {slot.label}.")
            if provided is not None and provided.filename:
                suffix = Path(provided.filename).suffix.lower()
                if suffix not in slot.accept:
                    raise ModelUploadError(
                        f"{slot.label} must be one of {slot.accept}; got '{suffix or provided.filename}'."
                    )

    async def _materialize(
        self,
        spec: _FamilySpec,
        files: dict[str, UploadFile],
        tmp_dir: Path,
        budget: _Budget,
        warnings: list[str],
    ) -> dict[str, Path]:
        if spec.kind == "zip":
            return await self._materialize_zip(spec, files, tmp_dir, budget, warnings)

        paths: dict[str, Path] = {}
        for slot in spec.files:
            upload = files.get(slot.key)
            if upload is None or not upload.filename:
                continue
            suffix = Path(upload.filename).suffix.lower()
            dest = tmp_dir / f"{slot.key}{suffix}"
            await self._stream(upload, dest, budget)
            self._validate_single(spec, slot.key, dest)
            paths[slot.key] = dest
        return paths

    async def _materialize_zip(
        self,
        spec: _FamilySpec,
        files: dict[str, UploadFile],
        tmp_dir: Path,
        budget: _Budget,
        warnings: list[str],
    ) -> dict[str, Path]:
        upload = files["archive"]
        raw_zip = tmp_dir / "_archive.zip"
        await self._stream(upload, raw_zip, budget)
        if not _is_zip(raw_zip):
            raise ModelUploadError("Uploaded archive is not a valid zip file.")

        extract_dir = tmp_dir / "model"
        extract_dir.mkdir(parents=True)
        names = self._extract_zip(raw_zip, extract_dir, budget)
        raw_zip.unlink(missing_ok=True)

        self._validate_zip_members(spec, names, warnings)
        return {"model": extract_dir}

    async def _stream(self, upload: UploadFile, dest: Path, budget: _Budget) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with dest.open("wb") as handle:
            while True:
                chunk = await upload.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                budget.take(len(chunk))
                handle.write(chunk)

    def _extract_zip(self, zip_path: Path, extract_dir: Path, budget: _Budget) -> list[str]:
        names: list[str] = []
        root = extract_dir.resolve()
        try:
            with ZipFile(zip_path) as archive:
                for info in archive.infolist():
                    name = info.filename
                    if info.is_dir():
                        continue
                    if name.startswith("/") or ".." in Path(name).parts or Path(name).is_absolute():
                        raise ModelUploadError(f"Rejected unsafe archive path: {name}")
                    target = (extract_dir / name).resolve()
                    if not str(target).startswith(str(root)):
                        raise ModelUploadError(f"Rejected zip-slip archive path: {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("wb") as handle:
                        while True:
                            chunk = source.read(_CHUNK)
                            if not chunk:
                                break
                            budget.take(len(chunk))
                            handle.write(chunk)
                    names.append(name)
        except BadZipFile as exc:
            raise ModelUploadError("Uploaded archive is not a valid zip file.") from exc
        if not names:
            raise ModelUploadError("Archive is empty.")
        return names

    # -- structural validation ---------------------------------------------

    def _validate_single(self, spec: _FamilySpec, slot_key: str, path: Path) -> None:
        if spec.family in {"yolo", "keras_classification", "unet_inception"}:
            # .pt/.keras are zip containers; .h5 is HDF5.
            suffix = path.suffix.lower()
            if suffix in {".pt", ".keras"} and not _is_zip(path):
                raise ModelUploadError(
                    f"{path.name} is not a valid {suffix} archive (bad magic bytes)."
                )
            if suffix == ".h5" and not _is_hdf5(path):
                raise ModelUploadError(f"{path.name} is not a valid HDF5 (.h5) file.")
        elif spec.family == "sklearn_pipeline":
            if not _looks_like_pickle(path):
                raise ModelUploadError(
                    f"{path.name} does not look like a pickle/joblib file."
                )
        elif spec.family == "llm_gguf":
            if not _is_gguf(path):
                raise ModelUploadError(f"{path.name} is missing the GGUF magic header.")

    def _validate_zip_members(
        self, spec: _FamilySpec, names: list[str], warnings: list[str]
    ) -> None:
        basenames = {Path(name).name for name in names}
        suffixes = {Path(name).suffix.lower() for name in names}
        if spec.family == "llm_hf":
            if "config.json" not in basenames:
                raise ModelUploadError("HF model zip must contain config.json.")
            if ".safetensors" not in suffixes:
                raise ModelUploadError("HF model zip must contain at least one .safetensors file.")
            tokenizer_files = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt"}
            if not (basenames & tokenizer_files):
                warnings.append("No tokenizer file found in the archive; add one before serving.")
        elif spec.family == "llm_adapter":
            if "adapter_config.json" not in basenames:
                raise ModelUploadError("Adapter zip must contain adapter_config.json.")
            weight_present = ".safetensors" in suffixes or ".bin" in suffixes
            if not weight_present:
                raise ModelUploadError("Adapter zip must contain adapter weights (.safetensors/.bin).")


# -- magic-byte helpers -----------------------------------------------------


def _read_head(path: Path, size: int = 8) -> bytes:
    with path.open("rb") as handle:
        return handle.read(size)


def _is_zip(path: Path) -> bool:
    head = _read_head(path, 4)
    return head[:2] == b"PK" and head[2:4] in {b"\x03\x04", b"\x05\x06", b"\x07\x08"}


def _is_hdf5(path: Path) -> bool:
    return _read_head(path, 8) == b"\x89HDF\r\n\x1a\n"


def _is_gguf(path: Path) -> bool:
    return _read_head(path, 4) == b"GGUF"


def _looks_like_pickle(path: Path) -> bool:
    head = _read_head(path, 4)
    if not head:
        return False
    # Pickle protocol 2+ starts with the PROTO opcode (0x80). joblib may
    # compress the stream, so accept common compression magics too.
    if head[0] == 0x80:
        return True
    compression_magics = (
        b"\x1f\x8b",  # gzip
        b"BZ",  # bz2
        b"\x78",  # zlib
        b"\x04\x22\x4d\x18",  # lz4
        b"\x28\xb5\x2f\xfd",  # zstd
        b"\x93NUM",  # numpy .npy (joblib memmap sidecars)
    )
    return any(head.startswith(magic) for magic in compression_magics)
