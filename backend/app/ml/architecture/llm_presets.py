"""Decoder-only LLM blueprints at real scales (phase 17).

These are *architectures*, not checkpoints. Each preset drops its family's own
block node onto the canvas — a Qwen3 preset carries a `qwen3_block`, a DeepSeek
preset a `deepseek_block` — so opening two presets shows two different models
rather than the same six nodes with different numbers. The block's structural
parameters come from the model's published `config.json`, named in `source`,
and the studio's estimate reproduces the published parameter count.

**None of the multi-billion presets is trainable on a workstation.** They exist
to be read, edited, and exported: the estimate tells you what a config actually
costs before you rent the hardware, and the generated code is a starting point
rather than a training plan. `specs/phase-14` remains the path for fine-tuning a
real pretrained model. The small presets at the top of the list are the ones
that train locally.

Nothing here downloads or reproduces any provider's weights.
"""

from dataclasses import dataclass, field
from typing import Any

from app.ml.architecture.layout import auto_layout
from app.schemas import ArchitectureGraph


@dataclass(frozen=True)
class LlmPreset:
    id: str
    name: str
    description: str
    # The family block node this preset is built from. Everything structural —
    # attention kind, norm placement, feed-forward, routing — lives in the
    # block's own defaults; `overrides` states only what this size changes.
    block_type: str
    model_dim: int
    vocab: int
    sequence: int
    overrides: dict[str, Any] = field(default_factory=dict)
    # Reuse the embedding table as the output projection.
    tie_embeddings: bool = False
    # Bound the final logits with tanh (Gemma 2).
    logit_softcap: float = 0.0
    # Absolute learned positions, for the families that predate rotary.
    positional: bool = False
    # Where the numbers came from, shown in the studio.
    source: str = ""

    @property
    def layers(self) -> int:
        return int(self.overrides.get("layers", 0))


# Ordered smallest first: the top of this list is what a local machine can
# actually train, and the rest are blueprints.
PRESETS: list[LlmPreset] = [
    LlmPreset(
        id="llm_tiny",
        name="Tiny LM (~9M)",
        description=(
            "Six Llama-style layers that train on a laptop in minutes. Start here to see the "
            "whole loop work before scaling anything up."
        ),
        block_type="llama_block",
        model_dim=256,
        vocab=8000,
        sequence=256,
        overrides={"layers": 6, "num_heads": 8, "num_kv_heads": 8, "head_dim": 32, "ffn_dim": 768},
    ),
    LlmPreset(
        id="gpt2_124m",
        name="GPT-2 124M",
        description=(
            "The original small GPT: twelve pre-LayerNorm blocks, full multi-head attention, "
            "learned absolute positions, and biases on every projection. The smallest "
            "config that produces recognisable language, and trainable on one consumer GPU."
        ),
        block_type="gpt2_block",
        model_dim=768,
        vocab=50257,
        sequence=1024,
        overrides={"layers": 12, "num_heads": 12, "num_kv_heads": 12, "head_dim": 64, "ffn_dim": 3072},
        # GPT-2 projects with the transpose of its token embedding table.
        tie_embeddings=True,
        positional=True,
        source="openai-community/gpt2 config.json",
    ),
    LlmPreset(
        id="llama3_1b",
        name="Llama 3.2 1B",
        description=(
            "Sixteen layers at 2048 wide with grouped-query attention and tied embeddings. "
            "The smallest size modern instruction-tuned models ship at."
        ),
        block_type="llama_block",
        model_dim=2048,
        vocab=128256,
        sequence=8192,
        overrides={"layers": 16, "num_heads": 32, "num_kv_heads": 8, "head_dim": 64, "ffn_dim": 8192},
        tie_embeddings=True,
        source="meta-llama/Llama-3.2-1B config.json",
    ),
    LlmPreset(
        id="qwen3_0_6b",
        name="Qwen3 0.6B",
        description=(
            "Qwen3's smallest dense model: 28 layers at 1024 wide with QK-norm and tied "
            "embeddings. Small enough to fine-tune on one consumer GPU."
        ),
        block_type="qwen3_block",
        model_dim=1024,
        vocab=151936,
        sequence=32768,
        overrides={"layers": 28, "num_heads": 16, "num_kv_heads": 8, "head_dim": 128, "ffn_dim": 3072},
        tie_embeddings=True,
        source="Qwen/Qwen3-0.6B config.json",
    ),
    LlmPreset(
        id="llama3_3b",
        name="Llama 3.2 3B",
        description="Twenty-eight layers at 3072 wide. The usual ceiling for on-device inference.",
        block_type="llama_block",
        model_dim=3072,
        vocab=128256,
        sequence=8192,
        overrides={"layers": 28, "num_heads": 24, "num_kv_heads": 8, "head_dim": 128, "ffn_dim": 8192},
        tie_embeddings=True,
        source="meta-llama/Llama-3.2-3B config.json",
    ),
    LlmPreset(
        id="gemma3_4b",
        name="Gemma 3 4B (text)",
        description=(
            "Gemma's four-norms-per-layer sandwich arrangement, QK-norm, a GeGLU "
            "feed-forward, five sliding-window layers to every global one, and a 256-wide "
            "head dimension against a 2560-wide residual stream. The text tower only — the "
            "published 4B checkpoint adds a SigLIP vision encoder this canvas does not model."
        ),
        block_type="gemma3_block",
        model_dim=2560,
        vocab=262208,
        sequence=131072,
        overrides={
            "layers": 34,
            "num_heads": 8,
            "num_kv_heads": 4,
            "head_dim": 256,
            "ffn_dim": 10240,
            "sliding_window": 1024,
            "global_every": 6,
        },
        tie_embeddings=True,
        source="google/gemma-3-4b-it config.json (text_config)",
    ),
    LlmPreset(
        id="mistral_7b",
        name="Mistral 7B",
        description=(
            "Thirty-two layers at 4096 wide with 8 key/value heads over a 4096-token "
            "sliding window — the most-copied open-weight configuration there is. "
            "Set the window to 0 for the v0.3 behaviour, which attends globally."
        ),
        block_type="mistral_block",
        model_dim=4096,
        vocab=32768,
        sequence=32768,
        overrides={
            "layers": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "ffn_dim": 14336,
            "sliding_window": 4096,
        },
        source="mistralai/Mistral-7B-v0.3 config.json",
    ),
    LlmPreset(
        id="llama3_8b",
        name="Llama 3 8B",
        description=(
            "Thirty-two layers at 4096 wide, 8 key/value heads, a 128k vocabulary, and a "
            "rotary base of 500,000. The reference dense model of its generation."
        ),
        block_type="llama_block",
        model_dim=4096,
        vocab=128256,
        sequence=8192,
        overrides={"layers": 32, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128, "ffn_dim": 14336},
        source="meta-llama/Meta-Llama-3-8B config.json",
    ),
    LlmPreset(
        id="qwen3_8b",
        name="Qwen3 8B",
        description=(
            "Thirty-six layers at 4096 wide with grouped-query attention and RMSNorm on the "
            "query and key head vectors — Qwen3's signature QK-norm."
        ),
        block_type="qwen3_block",
        model_dim=4096,
        vocab=151936,
        sequence=40960,
        overrides={"layers": 36, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128, "ffn_dim": 12288},
        source="Qwen/Qwen3-8B config.json",
    ),
    LlmPreset(
        id="qwen3_30b_a3b",
        name="Qwen3 30B-A3B (MoE)",
        description=(
            "128 experts with 8 routed per token across 48 layers — 30B of weights for the "
            "cost of about 3B per token. Fine-grained routing rather than Mixtral's eight, "
            "and QK-norm throughout."
        ),
        block_type="qwen3_block",
        model_dim=2048,
        vocab=151936,
        sequence=32768,
        overrides={
            "layers": 48,
            "num_heads": 32,
            "num_kv_heads": 4,
            "head_dim": 128,
            "ffn": "moe",
            "ffn_dim": 768,
            "num_experts": 128,
            "experts_per_token": 8,
        },
        source="Qwen/Qwen3-30B-A3B config.json",
    ),
    LlmPreset(
        id="mixtral_8x7b",
        name="Mixtral 8×7B (sparse MoE)",
        description=(
            "Eight experts per layer, two active per token. Total parameters near 47B while "
            "each token pays for roughly 13B — the sparse trade, made visible."
        ),
        block_type="mixtral_block",
        model_dim=4096,
        vocab=32000,
        sequence=32768,
        overrides={
            "layers": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "ffn_dim": 14336,
            "num_experts": 8,
            "experts_per_token": 2,
        },
        source="mistralai/Mixtral-8x7B-v0.1 config.json",
    ),
    LlmPreset(
        id="llama3_70b",
        name="Llama 3 70B",
        description=(
            "Eighty layers at 8192 wide. A blueprint for reading and costing, not for "
            "training anywhere outside a cluster."
        ),
        block_type="llama_block",
        model_dim=8192,
        vocab=128256,
        sequence=8192,
        overrides={"layers": 80, "num_heads": 64, "num_kv_heads": 8, "head_dim": 128, "ffn_dim": 28672},
        source="meta-llama/Meta-Llama-3-70B config.json",
    ),
    LlmPreset(
        id="deepseek_v3",
        name="DeepSeek V3 (MLA + MoE)",
        description=(
            "Multi-head latent attention at 61 layers and 128 heads, with one shared expert "
            "and 256 routed at 8 active — and the first three layers left dense. The "
            "canonical trillion-scale sparse design, at its published 671B."
        ),
        block_type="deepseek_block",
        model_dim=7168,
        vocab=129280,
        sequence=163840,
        overrides={},
        source="deepseek-ai/DeepSeek-V3 config.json",
    ),
    LlmPreset(
        id="kimi_k2",
        name="Kimi K2 (1T MoE)",
        description=(
            "DeepSeek V3's block widened to 384 experts and halved to 64 attention heads, "
            "with one dense layer instead of three. Just over a trillion total parameters "
            "for roughly 32B active per token."
        ),
        block_type="kimi_block",
        model_dim=7168,
        vocab=163840,
        sequence=131072,
        overrides={},
        source="moonshotai/Kimi-K2-Instruct config.json",
    ),
]

PRESETS_BY_ID = {preset.id: preset for preset in PRESETS}


def preset_graph(preset: LlmPreset) -> ArchitectureGraph:
    """Build the decoder-only graph for a preset.

    Every preset is the same five or six nodes — embedding, the family's block,
    a final norm, an LM head — because that *is* the shape of a decoder-only
    model. What differs is which block node sits in the middle, and the code it
    generates: a `deepseek_block` emits latent attention and a sigmoid router
    where a `qwen3_block` emits grouped-query attention and a SwiGLU.

    Rotary position is not a node here. Every rotary family applies it to the
    queries and keys *inside* attention, so the block owns it; only the
    pre-rotary families (GPT-2, BERT) take a Positional embedding node.
    """

    # Imported here rather than at module scope: `templates` imports this
    # module for the preset list, so a top-level import would be circular.
    from app.ml.architecture.blocks import llm_block_params
    from app.ml.architecture.templates import chain, node

    block_params = {**llm_block_params(preset.block_type), **preset.overrides}
    nodes = [
        node("input", "input", shape=str(preset.sequence)),
        node(
            "embed",
            "embedding",
            input_dim=preset.vocab,
            output_dim=preset.model_dim,
            mask_zero=False,
        ),
    ]
    order = ["input", "embed"]
    if preset.positional:
        nodes.append(node("pos", "positional_embedding", max_length=preset.sequence))
        order.append("pos")
    nodes.append(node("blocks", preset.block_type, **block_params))
    nodes.append(node("final_norm", "rms_norm" if block_params["norm"] == "rms" else "layer_norm"))
    nodes.append(
        node(
            "head",
            "lm_head",
            vocab_size=preset.vocab,
            vocab_from_dataset=False,
            tie_embeddings=preset.tie_embeddings,
            logit_softcap=preset.logit_softcap,
        )
    )
    nodes.append(node("output", "output"))
    order += ["blocks", "final_norm", "head", "output"]

    return auto_layout(
        ArchitectureGraph(
            nodes=nodes,
            edges=chain(*order),
            training_defaults={
                "epochs": 1,
                "batch_size": 1,
                "learning_rate": 0.0003,
                "optimizer": "adamw",
                "num_classes_preview": preset.vocab,
            },
        )
    )
