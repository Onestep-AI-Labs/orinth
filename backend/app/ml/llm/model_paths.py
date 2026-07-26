"""Shared on-disk resolution for LLM model artifacts (phase 15).

Both the export service (merge/convert) and the evaluation runner need to find
an adapter directory, an HF model directory, and the base checkpoint an adapter
was tuned against. Keeping that logic in one place means export and eval agree
on what "the base model" is.
"""

import json
from pathlib import Path

from app.ml.llm.catalog import llm_model_id
from app.ml.model_registry import ModelRegistry, ModelSpec


class ModelPathError(ValueError):
    """Raised when an LLM model's on-disk layout cannot be resolved."""


def dir_containing(spec: ModelSpec, marker: str) -> Path:
    """The directory holding ``marker`` for a model spec (handles nested zips)."""
    root = spec.paths.get("model") or next(iter(spec.paths.values()), None)
    if root is None:
        raise ModelPathError("Model has no on-disk files.")
    if root.is_file():
        return root.parent
    if (root / marker).exists():
        return root
    nested = next(iter(sorted(root.rglob(marker))), None)
    if nested is not None:
        return nested.parent
    raise ModelPathError(f"Could not find {marker} under {root.name}; not a usable model directory.")


def adapter_dir(spec: ModelSpec) -> Path:
    return dir_containing(spec, "adapter_config.json")


def hf_model_dir(spec: ModelSpec) -> Path:
    return dir_containing(spec, "config.json")


def resolve_base_ref(registry: ModelRegistry, spec: ModelSpec) -> str:
    """Resolve the base checkpoint an adapter was tuned against.

    Preference order: the exact ref the training run recorded (a hub id or a
    local path), a registered base model's local path, a catalog option id
    mapped to its hub id, then the adapter's own ``adapter_config.json``.
    """
    artifacts = spec.artifacts or {}
    recorded = str(artifacts.get("base_model_ref") or "").strip()
    if recorded:
        recorded_path = Path(recorded)
        if not recorded_path.is_absolute() or recorded_path.exists():
            return recorded

    base_id = (spec.base_model_id or "").strip()
    if base_id:
        try:
            base_spec = registry.get_spec(base_id)
            base_path = next(iter(base_spec.paths.values()), None)
            if base_path is not None and base_path.exists():
                return str(hf_model_dir(base_spec))
        except KeyError:
            pass
        mapped = llm_model_id(base_id)
        if mapped != base_id or "/" in base_id:
            return mapped

    adapter_config = adapter_dir(spec) / "adapter_config.json"
    if adapter_config.exists():
        try:
            config = json.loads(adapter_config.read_text(encoding="utf-8"))
            ref = str(config.get("base_model_name_or_path") or "").strip()
            if ref:
                return ref
        except json.JSONDecodeError:
            pass
    raise ModelPathError(
        "Could not resolve the adapter's base model. Re-register the adapter with a "
        "valid base_model_id, or use the original training run."
    )
