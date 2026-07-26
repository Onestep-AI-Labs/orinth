"""LLM training environment probe (phase 14).

Answers "what would an ``llm_sft`` run actually do on this machine" — device,
backend availability, and the recommended backend — so the training form can
say it up front instead of the run log saying it an hour in.
"""

from importlib.util import find_spec

from app.ml.llm.catalog import LLM_INSTALL_HINT, llm_extras_available
from app.schemas import LlmEnvironment


def _importable(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def probe_llm_environment() -> LlmEnvironment:
    # Lazy torch import: this handler must not add torch to API startup.
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    unsloth_available = _importable("unsloth")
    peft_available = _importable("peft")
    bitsandbytes_available = _importable("bitsandbytes")
    recommended = "unsloth" if device == "cuda" and unsloth_available else "peft"

    notes: list[str] = []
    if not llm_extras_available():
        notes.append(LLM_INSTALL_HINT)
    if device == "cuda":
        if not unsloth_available:
            notes.append(
                "CUDA is available but Unsloth is not installed; runs use transformers + PEFT. "
                "Install `--extra llm-cuda` for 4-bit QLoRA."
            )
        elif not bitsandbytes_available:
            notes.append(
                "bitsandbytes is not importable; 4-bit loading will fall back to fp16 LoRA."
            )
    elif device == "mps":
        notes.append(
            "4-bit QLoRA is unavailable on MPS; runs use PEFT LoRA with gradient checkpointing."
        )
    else:
        notes.append("No accelerator detected; training on CPU will be very slow.")

    return LlmEnvironment(
        device=device,
        unsloth_available=unsloth_available,
        peft_available=peft_available,
        bitsandbytes_available=bitsandbytes_available,
        recommended_backend=recommended,
        notes=notes,
    )
