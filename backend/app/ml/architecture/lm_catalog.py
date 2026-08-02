"""Training-catalog entry for from-scratch language models (phase 17, stage E).

Separate from `train_catalog` because the two runners share no dataset
pipeline: one windows a text corpus into next-token pairs, the other batches
labelled images. They only share the graph in front of them.
"""

from app.ml.common.advanced import (
    GENERATION,
    OPTIMIZATION,
    RUNTIME,
    SEQUENCE,
    number,
    select,
)
from app.ml.common.catalog import TrainingModelDefinition
from app.schemas import AdvancedParameterSpec

LM_FAMILY = "architecture_lm"
LM_OPTION_ID = "architecture_lm"

LM_ADVANCED_PARAMETERS: list[AdvancedParameterSpec] = [
    select(
        "tokenizer",
        "Tokenizer",
        options=["word", "char"],
        default="word",
        group=SEQUENCE,
        help="Character level learns spelling from tiny corpora; word level reads better but needs more text.",
    ),
    number(
        "vocab_size",
        "Vocabulary size",
        default=2000,
        group=SEQUENCE,
        minimum=16,
        maximum=100_000,
        integer=True,
        help="Most frequent tokens kept; the rest map to <unk>.",
    ),
    number(
        "stride",
        "Window stride",
        default=0,
        group=SEQUENCE,
        minimum=0,
        maximum=4096,
        integer=True,
        help="Token gap between training windows; 0 uses half the sequence length.",
    ),
    select(
        "lr_schedule",
        "LR schedule",
        options=["cosine", "constant", "step", "plateau"],
        default="cosine",
        group=OPTIMIZATION,
        help="Cosine decay is the usual choice for language-model pretraining.",
    ),
    number(
        "early_stop_patience",
        "Early stop patience",
        default=0,
        group=OPTIMIZATION,
        minimum=0,
        maximum=50,
        integer=True,
        help="Stop after N epochs without a validation-loss improvement; 0 disables.",
    ),
    number(
        "temperature",
        "Sampling temperature",
        default=0.8,
        group=GENERATION,
        minimum=0.0,
        maximum=2.0,
        step=0.05,
        help="0 is greedy; higher is more varied. Applies to the samples shown after training.",
    ),
    number(
        "max_new_tokens",
        "Sample length",
        default=40,
        group=GENERATION,
        minimum=1,
        maximum=512,
        integer=True,
    ),
    number(
        "seed", "Seed", default=42, group=RUNTIME, minimum=0, maximum=1_000_000, integer=True
    ),
]


def lm_training_options() -> list[TrainingModelDefinition]:
    return [
        TrainingModelDefinition(
            id=LM_OPTION_ID,
            name="Visual architecture (language model)",
            family=LM_FAMILY,
            task_types=["language_modeling"],
            source="local",
            runnable=True,
            needs_download=False,
            description=(
                "Train a transformer you composed in the studio on next-token prediction. "
                "Research scale — use LLM fine-tuning for a production model."
            ),
            defaults={
                "epochs": 20,
                "batch_size": 32,
                "optimizer": "adamw",
                "learning_rate": 0.001,
            },
            advanced_parameters=LM_ADVANCED_PARAMETERS,
        )
    ]
