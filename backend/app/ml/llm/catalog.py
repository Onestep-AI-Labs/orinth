"""LLM SFT training catalog (phase 14): family ``llm_sft``, task ``llm_finetune``.

Five small instruction-tuned hub bases plus a dynamic option per registered
``llm_hf`` model (phase 12 uploads and prior merges). Options list even when
the optional ``llm`` dependency group is not installed — gated ``runnable:
False`` with an install note — so the UI can explain the feature instead of
hiding it.
"""

from importlib.util import find_spec
from typing import Any

from app.ml.common.advanced import (
    LORA,
    METHOD,
    OPTIMIZATION,
    QUANTIZATION,
    RUNTIME,
    SEQUENCE,
    multiselect,
    number,
    select,
    toggle,
)
from app.ml.common.catalog import TrainingModelDefinition
from app.schemas import TrainingModelOption

LLM_SFT_FAMILY = "llm_sft"
LLM_TASK_TYPE = "llm_finetune"
# Registry model ids are prefixed to become catalog option ids for the
# "Local base models" group; the training service strips it back off to
# resolve the registered model's filesystem path.
LOCAL_BASE_OPTION_PREFIX = "llm_local_"

# The "type any Hugging Face model id" catalog option. Its model ref rides the
# job's `base_model` field instead of resolving from a fixed catalog entry.
LLM_HF_CUSTOM_ID = "llm_hf_custom"

LLM_INSTALL_HINT = (
    "Install the LLM extras first: `cd backend && uv sync --extra llm` "
    "(add `--extra llm-cuda` on CUDA machines for Unsloth 4-bit QLoRA)."
)

# Which on-disk model formats can serve as a fine-tuning base, surfaced to the
# user so a GGUF/TFLite upload is not mistaken for a trainable base. Only the
# Hugging Face Transformers family (the `llm_hf` upload family) is trainable:
# LoRA attaches to live PyTorch modules, which the quantized inference formats
# do not expose.
TRAINABLE_BASE_FAMILIES = ["llm_hf"]
TRAINABLE_FORMAT_NOTE = (
    "Fine-tuning needs a Hugging Face Transformers model: safetensors or PyTorch "
    "weights with config.json and a tokenizer (the “Hugging Face model” upload "
    "family, or any hub id). GGUF and TFLite are quantized inference/export "
    "formats and cannot be used as a training base — use the original Hugging "
    "Face checkpoint instead. GGUF becomes available as an export target after "
    "training (phase 15)."
)

# All attention + MLP projections of the Llama-style architectures in this
# catalog. The default targets everything; trimming trades adaptation
# capacity for speed and memory.
TARGET_MODULE_OPTIONS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Keys map to the LLM SFT runner (`runners/llm_sft.py`); epochs, learning rate
# and per-device batch size stay as basic fields like every other family.
LLM_ADVANCED_PARAMETERS = [
    select(
        "finetune_method",
        "Fine-tuning method",
        options=["lora", "qlora", "full", "continued_pretrain"],
        default="lora",
        group=METHOD,
        help="LoRA: light adapter (recommended). QLoRA: 4-bit adapter, least VRAM (CUDA). "
        "Full: update every weight (heavy). Continued pretrain: LoRA on the whole sequence.",
    ),
    number("lora_r", "LoRA rank", default=16, group=LORA, minimum=1, maximum=256, integer=True, help="Adapter rank; higher adapts more but trains slower."),
    number("lora_alpha", "LoRA alpha", default=32, group=LORA, minimum=1, maximum=512, integer=True),
    number("lora_dropout", "LoRA dropout", default=0.05, group=LORA, minimum=0.0, maximum=0.5, step=0.01),
    multiselect("target_modules", "Target modules", options=TARGET_MODULE_OPTIONS, default=TARGET_MODULE_OPTIONS, group=LORA, help="Attention and MLP projections the adapter attaches to."),
    toggle("load_in_4bit", "Load in 4-bit (QLoRA)", default=True, group=QUANTIZATION, help="Requires CUDA + bitsandbytes; downgraded with a log note elsewhere."),
    number("max_seq_length", "Max sequence length", default=2048, group=SEQUENCE, minimum=128, maximum=32768, integer=True, help="Longer records are truncated and counted in the log."),
    toggle("packing", "Pack sequences", default=False, group=SEQUENCE, help="Concatenate short records into full-length sequences."),
    select("lr_scheduler", "LR scheduler", options=["linear", "cosine"], default="linear", group=OPTIMIZATION),
    number("warmup_ratio", "Warmup ratio", default=0.03, group=OPTIMIZATION, minimum=0.0, maximum=0.5, step=0.01),
    number("weight_decay", "Weight decay", default=0.01, group=OPTIMIZATION, minimum=0.0, maximum=0.3, step=0.01),
    number("gradient_accumulation_steps", "Gradient accumulation", default=4, group=OPTIMIZATION, minimum=1, maximum=64, integer=True, help="Effective batch size is batch size × this."),
    number("max_steps", "Max steps", default=0, group=OPTIMIZATION, minimum=0, maximum=100000, integer=True, help="0 trains for the configured epochs; a positive value overrides epochs with a fixed step budget."),
    number("seed", "Seed", default=42, group=RUNTIME, minimum=0, maximum=1_000_000, integer=True),
    select("precision", "Precision", options=["bf16", "fp16", "fp32"], default="bf16", group=RUNTIME, help="Downgraded with a log note on devices without support."),
    toggle("gradient_checkpointing", "Gradient checkpointing", default=True, group=RUNTIME, help="Trades compute for memory; keep on for MPS and small GPUs."),
    toggle("train_on_full_sequence", "Train on full sequence", default=False, group=RUNTIME, help="Compute loss on prompt tokens too, not just assistant responses."),
]

# Ordering is load-bearing: the training page lists the smallest, least
# memory-hungry option first. Descriptions carry the memory guidance the spec
# requires so users can self-select for their machine.
LLM_MODEL_OPTIONS: list[dict[str, Any]] = [
    {
        "id": "qwen2_5_1_5b",
        "name": "Qwen2.5 1.5B Instruct",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "approx_download_gb": 4,
        "description": (
            "Apache-2.0 instruction model — the safest default and the lightest to "
            "run. QLoRA needs ~4 GB CUDA VRAM; MPS LoRA runs on an 8 GB+ Mac."
        ),
    },
    {
        "id": "smollm2_1_7b",
        "name": "SmolLM2 1.7B Instruct",
        "model_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "approx_download_gb": 4,
        "description": (
            "Apache-2.0, ungated, and small — a reliable no-login option. QLoRA "
            "needs ~4 GB CUDA VRAM; MPS LoRA runs on an 8 GB+ Mac."
        ),
    },
    {
        "id": "qwen2_5_3b",
        "name": "Qwen2.5 3B Instruct",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "approx_download_gb": 7,
        "description": (
            "Stronger 3B-class Apache-2.0 model. QLoRA needs ~7 GB CUDA VRAM; MPS "
            "LoRA wants a 16 GB+ unified-memory Mac."
        ),
    },
    {
        "id": "gemma3_1b",
        "name": "Gemma 3 1B IT",
        "model_id": "google/gemma-3-1b-it",
        "gated_license": False,
        "approx_download_gb": 3,
        "description": (
            "Google's compact Gemma 3. Requires an accepted license on huggingface.co. "
            "QLoRA needs ~4 GB CUDA VRAM; MPS LoRA runs on an 8 GB+ Mac."
        ),
    },
    {
        "id": "gemma3_4b",
        "name": "Gemma 3 4B IT",
        "model_id": "google/gemma-3-4b-it",
        "gated_license": False,
        "approx_download_gb": 9,
        "description": (
            "Gemma 3 4B-class. Requires an accepted license on huggingface.co. QLoRA "
            "needs ~9 GB CUDA VRAM; on MPS this needs a high-memory (24 GB+) machine."
        ),
    },
]

LLM_OPTIONS_BY_ID = {item["id"]: item for item in LLM_MODEL_OPTIONS}


def llm_extras_available() -> bool:
    """Whether the optional ``llm`` dependency group is importable.

    ``find_spec`` keeps this lazy — it checks importability without importing
    peft/trl (and their torch initialisation) into the API process.
    """
    try:
        return find_spec("peft") is not None and find_spec("trl") is not None
    except (ImportError, ValueError):
        return False


def _custom_hf_option(runnable: bool) -> TrainingModelDefinition:
    """The "any Hugging Face model id" base option.

    Its hub id is entered by the user and rides the job's ``base_model`` field,
    so any Transformers checkpoint can be fine-tuned without a catalog entry.
    """
    description = (
        "Type any Hugging Face model id (for example `Qwen/Qwen2.5-1.5B-Instruct`) "
        "to fine-tune a checkpoint not listed above. It must be a Transformers "
        "model — GGUF/TFLite repos cannot be trained."
    )
    return TrainingModelDefinition(
        id=LLM_HF_CUSTOM_ID,
        name="Custom Hugging Face model…",
        family=LLM_SFT_FAMILY,
        task_types=[LLM_TASK_TYPE],
        source="huggingface",
        runnable=runnable,
        needs_download=True,
        description=description if runnable else f"{description} {LLM_INSTALL_HINT}",
        defaults={
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 0.0002,
            "optimizer": "adamw",
            "custom_hf": True,
        },
        advanced_parameters=LLM_ADVANCED_PARAMETERS,
    )


def llm_training_options() -> list[TrainingModelDefinition]:
    runnable = llm_extras_available()
    options = [
        TrainingModelDefinition(
            id=item["id"],
            name=item["name"],
            family=LLM_SFT_FAMILY,
            task_types=[LLM_TASK_TYPE],
            source="huggingface",
            runnable=runnable,
            needs_download=True,
            description=item["description"] if runnable else f"{item['description']} {LLM_INSTALL_HINT}",
            defaults={
                "epochs": 3,
                "batch_size": 2,
                "learning_rate": 0.0002,
                "optimizer": "adamw",
                "model_id": item["model_id"],
                "gated_license": bool(item.get("gated_license", False)),
                "approx_download_gb": item["approx_download_gb"],
            },
            advanced_parameters=LLM_ADVANCED_PARAMETERS,
        )
        for item in LLM_MODEL_OPTIONS
    ]
    options.append(_custom_hf_option(runnable))
    return options


def local_llm_base_options(registry) -> list[TrainingModelOption]:
    """A dynamic option per registered ``llm_hf`` base model.

    Satisfies "fine-tune using a local model or from Hugging Face": the runner
    receives the registered model's filesystem path instead of a hub id.
    """
    runnable = llm_extras_available()
    options = []
    for spec in registry.list_specs():
        if spec.family != "llm_hf":
            continue
        description = "Local base models: registered Hugging Face model (phase 12 upload or prior merge)."
        options.append(
            TrainingModelOption(
                id=f"{LOCAL_BASE_OPTION_PREFIX}{spec.id}",
                name=f"{spec.name} (local)",
                family=LLM_SFT_FAMILY,
                task_types=[LLM_TASK_TYPE],
                source="local",
                runnable=runnable and spec.available,
                needs_download=False,
                description=description if runnable else f"{description} {LLM_INSTALL_HINT}",
                defaults={
                    "epochs": 3,
                    "batch_size": 2,
                    "learning_rate": 0.0002,
                    "optimizer": "adamw",
                    "local_base": True,
                },
                advanced_parameters=LLM_ADVANCED_PARAMETERS,
            )
        )
    return options


def catalog_base_ids() -> list[str]:
    """Hub base-model option ids — consumed by phase-12 adapter upload validation."""
    return [item["id"] for item in LLM_MODEL_OPTIONS]


def llm_model_id(option_id: str) -> str:
    option = LLM_OPTIONS_BY_ID.get(option_id)
    return str(option["model_id"]) if option else option_id


def llm_gated_license(option_id: str) -> bool:
    return bool(LLM_OPTIONS_BY_ID.get(option_id, {}).get("gated_license", False))
