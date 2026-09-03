"""The model catalog, and the platform's own predictors.

`predictor()` is the reason this module is worth having rather than reading
`model_registry.json` by hand: it returns the exact object `/inference` runs, so
a notebook can diff two models against a split and know it is measuring what the
platform will actually do. It imports TensorFlow or Ultralytics on first call —
inside the function, in the kernel process, which is allowed and is the whole
reason the kernel is not the API process.
"""

import builtins
from pathlib import Path

from orinth import _workspace
from orinth.errors import OrinthError
from orinth.types import ModelRef


def _registry():
    from app.ml.model_registry import ModelRegistry

    return ModelRegistry(_workspace.settings(), _workspace.storage())


def _ref(info) -> ModelRef:
    # `ModelInfo.paths` maps an artifact role to a location — "weights",
    # "config", and so on, because a model is often several files. The SDK
    # exposes the primary one as `path` and leaves the rest to `paths()`, since
    # a notebook asking "where is this model" means the weights.
    artifacts = dict(getattr(info, "paths", {}) or {})
    primary = artifacts.get("weights") or next(iter(artifacts.values()), None)
    return ModelRef(
        id=info.id,
        name=getattr(info, "name", info.id),
        family=getattr(info, "family", "") or "",
        task_type=getattr(info, "task_type", "") or "",
        source=getattr(info, "source", "") or "",
        path=Path(primary) if primary else None,
        available=bool(getattr(info, "available", True)),
    )


def list(  # noqa: A001 - matches `orinth.datasets.list`; the builtin is one attribute away
    *, task_type: str | None = None, source: str | None = None
) -> builtins.list[ModelRef]:
    refs = [_ref(info) for info in _registry().list_models()]
    if task_type:
        refs = [ref for ref in refs if ref.task_type == task_type]
    if source:
        refs = [ref for ref in refs if ref.source == source]
    return refs


def get(model_id: str) -> ModelRef:
    for ref in list():
        if ref.id == model_id:
            return ref
    raise OrinthError(f"No model '{model_id}' in this workspace.")


def path(model_id: str) -> Path:
    resolved = get(model_id).path
    if resolved is None:
        raise OrinthError(f"Model '{model_id}' has no artifact on disk.")
    return resolved


def predictor(model_id: str):
    """The platform's predictor for this model, loaded the way inference loads it.

    `get_predictor` caches per registry instance, so repeated calls in one cell
    do not reload a multi-gigabyte model.
    """
    return _registry().get_predictor(model_id)


def artifacts(model_id: str) -> dict:
    """Every artifact path this model declares, by role."""
    for info in _registry().list_models():
        if info.id == model_id:
            return dict(info.paths or {})
    raise OrinthError(f"No model '{model_id}' in this workspace.")
