"""Shared runner-side consumption of catalog advanced hyperparameters.

Runners receive the whole `hyperparameters` dict as a JSON blob and each keeps
an explicit allowlist derived from its own catalog specs. `partition` splits a
submitted dict into the keys a runner understands and the rest; unknown or
stale keys are returned so the runner can log-and-ignore them. A stale frontend
or a hand-edited request must never crash a run — it may only produce a log
note naming the ignored keys.

The allowlist is derived from the catalog `AdvancedParameterSpec` list rather
than hand-maintained here, so the field a user sees and the argument a runner
consumes cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from app.schemas import AdvancedParameterSpec


def allowed_keys(specs: Iterable[AdvancedParameterSpec]) -> set[str]:
    return {spec.key for spec in specs}


def parse_advanced(raw: str | None) -> dict[str, Any]:
    """Decode the ``--advanced`` JSON blob, tolerating empty/malformed input."""

    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def partition(
    hyperparameters: dict[str, Any], allowed: set[str]
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(accepted, ignored)`` for a submitted hyperparameter dict.

    ``accepted`` keeps only keys in ``allowed`` and drops ``None`` values
    (an empty multiselect means "runner default", not "set to nothing").
    ``ignored`` is the sorted list of every other key, for the log note.
    """

    accepted = {
        key: value
        for key, value in hyperparameters.items()
        if key in allowed and value is not None
    }
    ignored = sorted(key for key in hyperparameters if key not in allowed)
    return accepted, ignored


def log_ignored(ignored: list[str]) -> None:
    if ignored:
        print(f"Ignoring unknown advanced hyperparameters: {', '.join(ignored)}")


def resolve_precision(
    requested: str | None, *, cuda: bool, mps: bool
) -> tuple[str, str | None]:
    """Downgrade a requested precision to what the actual device supports.

    Only the runner knows the real device, so device-incompatible requests are
    downgraded here with a log note rather than rejected at the API. ``bf16``
    and ``fp16`` require CUDA; on MPS or CPU they fall back to ``fp32``.
    Returns ``(effective_precision, note_or_None)``.
    """

    value = (requested or "fp32").strip().lower()
    if value not in {"fp32", "fp16", "bf16"}:
        return "fp32", f"Unknown precision {requested!r}; using fp32."
    if value == "fp32" or cuda:
        return value, None
    device = "MPS" if mps else "CPU"
    return "fp32", f"{value} is not supported on {device}; using fp32."
