"""Concise builders for catalog `AdvancedParameterSpec` declarations.

Every catalog declares its advanced hyperparameters with these helpers so the
group names and field shapes stay consistent across families. The `key` each
builder produces must match the corresponding runner allowlist entry in
`app/training/runners/advanced.py` — that pairing is what makes a submitted
value take effect rather than being logged and ignored.
"""

from typing import Any

from app.schemas import AdvancedParameterSpec

# The group micro-headers the training form renders, in display order. The
# first four are LLM-specific (phase 14); the frontend GROUP_ORDER list in
# `features/training/advanced-settings.tsx` mirrors this ordering.
METHOD = "Method"
LORA = "LoRA"
QUANTIZATION = "Quantization"
SEQUENCE = "Sequence"
OPTIMIZATION = "Optimization"
AUGMENTATION = "Augmentation"
REGULARIZATION = "Regularization"
RUNTIME = "Runtime"


def number(
    key: str,
    label: str,
    *,
    default: float,
    group: str,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
    integer: bool = False,
    help: str | None = None,
) -> AdvancedParameterSpec:
    return AdvancedParameterSpec(
        key=key,
        label=label,
        type="int" if integer else "float",
        default=int(default) if integer else float(default),
        min=minimum,
        max=maximum,
        step=step,
        help=help,
        group=group,
    )


def toggle(
    key: str, label: str, *, default: bool, group: str, help: str | None = None
) -> AdvancedParameterSpec:
    return AdvancedParameterSpec(
        key=key, label=label, type="bool", default=default, help=help, group=group
    )


def select(
    key: str,
    label: str,
    *,
    options: list[Any],
    default: Any,
    group: str,
    help: str | None = None,
) -> AdvancedParameterSpec:
    return AdvancedParameterSpec(
        key=key,
        label=label,
        type="select",
        default=default,
        options=list(options),
        help=help,
        group=group,
    )


def multiselect(
    key: str,
    label: str,
    *,
    options: list[Any],
    default: list[Any],
    group: str,
    help: str | None = None,
) -> AdvancedParameterSpec:
    return AdvancedParameterSpec(
        key=key,
        label=label,
        type="multiselect",
        default=list(default),
        options=list(options),
        help=help,
        group=group,
    )
