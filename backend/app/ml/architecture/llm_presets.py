"""Decoder-only LLM blueprints at real scales (phase 17).

These are *architectures*, not checkpoints. Each preset reproduces the shape of
a published open-weight model — layer count, model width, head and KV-head
counts, feed-forward width, vocabulary — so the studio reports its true
parameter count and you can open it up and change it.

**None of the multi-billion presets is trainable on a workstation.** They exist
to be read, edited, and exported: the studio's parameter estimate tells you
what a config actually costs before you rent the hardware, and the generated
code is a starting point rather than a training plan. `specs/phase-14` remains
the path for fine-tuning a real pretrained model. The small presets at the top
of the list are the ones that actually train locally.

Shapes follow the public configs of the Llama, Qwen, Mistral, and Mixtral
families. Vocabulary sizes are rounded to the nearest common value; nothing
here downloads or reproduces any provider's weights.
"""

from dataclasses import dataclass

from app.ml.architecture.layout import auto_layout
from app.schemas import ArchitectureGraph


@dataclass(frozen=True)
class LlmPreset:
    id: str
    name: str
    description: str
    layers: int
    model_dim: int
    heads: int
    kv_heads: int
    ffn_dim: int
    vocab: int
    sequence: int
    # Explicit when the model decouples it from width / heads; 0 derives it.
    _head_dim: int = 0
    # Mixture-of-experts presets route each token to `experts_per_token` of
    # `experts`; dense presets leave `experts` at 0.
    experts: int = 0
    experts_per_token: int = 2
    # Always-on experts alongside the routed ones (DeepSeek's design).
    shared_experts: int = 0
    # RMSNorm on the query and key head dimensions (Qwen3, Gemma 3+).
    qk_norm: bool = False
    # Local attention window, with every `global_every`-th layer promoted to
    # full attention (Gemma 4's hybrid pattern).
    sliding_window: int = 0
    global_every: int = 0
    # Reuse the embedding table as the output projection (Gemma, small Qwen).
    tie_embeddings: bool = False
    # Bound the final logits with tanh (Gemma).
    logit_softcap: float = 0.0
    # Where the numbers came from, shown in the studio.
    source: str = ""

    @property
    def is_sparse(self) -> bool:
        return self.experts > 0

    @property
    def head_dim(self) -> int:
        """Published head dimension, which is not always width / heads.

        Gemma decouples the two: 8 heads of 256 total 2048 against a 2304-wide
        residual stream. Deriving it as width / heads would give 288 and
        silently produce a different model.
        """

        return self._head_dim or self.model_dim // self.heads


# Ordered smallest first: the top of this list is what a local machine can
# actually train, and the rest are blueprints.
PRESETS: list[LlmPreset] = [
    LlmPreset(
        id="llm_tiny",
        name="Tiny LM (~6M)",
        description=(
            "Six-layer decoder that trains on a laptop in minutes. Start here to see the "
            "whole loop work before scaling anything up."
        ),
        layers=6, model_dim=256, heads=8, kv_heads=8, ffn_dim=768, vocab=8000, sequence=256,
    ),
    LlmPreset(
        id="llm_small",
        name="Small LM (~124M, GPT-2 shape)",
        description=(
            "The classic 12-layer, 768-wide decoder. Trainable on one consumer GPU and the "
            "smallest config that produces recognisable language."
        ),
        layers=12, model_dim=768, heads=12, kv_heads=12, ffn_dim=3072, vocab=50257, sequence=1024,
    ),
    LlmPreset(
        id="llm_1b",
        name="1B dense (Llama 3.2 1B shape)",
        description=(
            "Sixteen layers, 2048 wide, grouped-query attention with 8 KV heads. The smallest "
            "size modern instruction-tuned models ship at."
        ),
        layers=16, model_dim=2048, heads=32, kv_heads=8, ffn_dim=8192, vocab=128256, sequence=8192,
    ),
    LlmPreset(
        id="llm_3b",
        name="3B dense (Llama 3.2 3B shape)",
        description=(
            "Twenty-eight layers at 3072 wide. The usual ceiling for on-device inference."
        ),
        layers=28, model_dim=3072, heads=24, kv_heads=8, ffn_dim=8192, vocab=128256, sequence=8192,
    ),
    LlmPreset(
        id="llm_7b",
        name="7B dense (Llama / Mistral 7B shape)",
        description=(
            "Thirty-two layers at 4096 wide with 8 KV heads — the most-copied open-weight "
            "configuration there is."
        ),
        layers=32, model_dim=4096, heads=32, kv_heads=8, ffn_dim=14336, vocab=32000, sequence=8192,
    ),
    LlmPreset(
        id="llm_13b",
        name="13B dense",
        description=(
            "Forty layers at 5120 wide with full multi-head attention. The last common size "
            "before grouped-query attention became mandatory for serving cost."
        ),
        layers=40, model_dim=5120, heads=40, kv_heads=40, ffn_dim=13824, vocab=32000, sequence=4096,
    ),
    LlmPreset(
        id="llm_70b",
        name="70B dense (Llama 3 70B shape)",
        description=(
            "Eighty layers at 8192 wide. A blueprint for reading and costing, not for "
            "training anywhere outside a cluster."
        ),
        layers=80, model_dim=8192, heads=64, kv_heads=8, ffn_dim=28672, vocab=128256, sequence=8192,
    ),
    LlmPreset(
        id="llm_moe_8x7b",
        name="8×7B sparse MoE (Mixtral shape)",
        description=(
            "Eight experts per layer, two active per token. Total parameters near 47B while "
            "each token pays for roughly 13B — the sparse trade, made visible."
        ),
        layers=32, model_dim=4096, heads=32, kv_heads=8, ffn_dim=14336, vocab=32000,
        sequence=8192, experts=8, experts_per_token=2,
    ),
    LlmPreset(
        id="llm_moe_small",
        name="Small sparse MoE (~200M)",
        description=(
            "Eight small experts, two active per token. Big enough to show routing behaviour "
            "and small enough to fine-tune on one GPU."
        ),
        layers=12, model_dim=512, heads=8, kv_heads=4, ffn_dim=1024, vocab=32000,
        sequence=1024, experts=8, experts_per_token=2,
    ),

    # --- named open-weight configurations ---------------------------------
    #
    # Every field below is read from the model's published `config.json` /
    # `PretrainedConfig` defaults, not inferred. Where a family's signature
    # feature is expressible on the canvas — QK-norm, sliding windows, shared
    # experts, tied embeddings — it is switched on, so the generated code has
    # the shape of the real thing rather than a generic transformer wearing its
    # parameter count.

    LlmPreset(
        id="qwen3_0_6b",
        name="Qwen3 0.6B",
        description=(
            "Qwen3's smallest dense model: 28 layers at 1024 wide with QK-norm and tied "
            "embeddings. Small enough to fine-tune on one consumer GPU."
        ),
        layers=28, model_dim=1024, heads=16, kv_heads=8, ffn_dim=3072, vocab=151936,
        sequence=32768, _head_dim=128, qk_norm=True, tie_embeddings=True,
        source="Qwen3Config / Qwen/Qwen3-0.6B",
    ),
    LlmPreset(
        id="qwen3_8b",
        name="Qwen3 8B",
        description=(
            "Thirty-six layers at 4096 wide, grouped-query attention with 8 KV heads, and "
            "RMSNorm on the query and key head dimensions — Qwen3's signature QK-norm."
        ),
        layers=36, model_dim=4096, heads=32, kv_heads=8, ffn_dim=12288, vocab=151936,
        sequence=32768, _head_dim=128, qk_norm=True,
        source="Qwen3Config / Qwen/Qwen3-8B",
    ),
    LlmPreset(
        id="qwen3_30b_a3b",
        name="Qwen3 30B-A3B (MoE)",
        description=(
            "128 experts with 8 routed per token across 48 layers — 30B of weights for the "
            "cost of about 3B per token. Fine-grained routing rather than Mixtral's eight."
        ),
        layers=48, model_dim=2048, heads=32, kv_heads=4, ffn_dim=768, vocab=151936,
        sequence=32768, _head_dim=128, qk_norm=True, experts=128, experts_per_token=8,
        source="Qwen3MoeConfig / Qwen/Qwen3-30B-A3B",
    ),
    LlmPreset(
        id="gemma4_sparse",
        name="Gemma 4 sparse (config defaults)",
        description=(
            "Gemma 4's signature shape: hybrid attention — five sliding-window layers to "
            "every global one, at a 512-token window — with tied embeddings, logit "
            "softcapping, and a 256-wide head dimension that is deliberately wider than "
            "the 2304 residual stream. Sized from Gemma4TextConfig's documented defaults, "
            "so it lands near 16B rather than the published 26B-A4B: that checkpoint adds "
            "per-layer embeddings (PLE) and an expert count the public config does not "
            "state, neither of which this canvas models."
        ),
        layers=30, model_dim=2304, heads=8, kv_heads=4, ffn_dim=9216, vocab=262144,
        sequence=8192, _head_dim=256, sliding_window=512, global_every=6,
        tie_embeddings=True, logit_softcap=30.0, experts=8, experts_per_token=2,
        source="Gemma4TextConfig documented defaults",
    ),
    LlmPreset(
        id="gemma4_dense",
        name="Gemma 4 dense (config defaults)",
        description=(
            "The same hybrid sliding/global attention, tied embeddings, and softcapped "
            "logits with one feed-forward per layer instead of a router. Sized from the "
            "documented Gemma4TextConfig defaults, which is a ~3B model — smaller than the "
            "published 31B checkpoint."
        ),
        layers=30, model_dim=2304, heads=8, kv_heads=4, ffn_dim=9216, vocab=262144,
        sequence=8192, _head_dim=256, sliding_window=512, global_every=6,
        tie_embeddings=True, logit_softcap=30.0,
        source="Gemma4TextConfig documented defaults",
    ),
    LlmPreset(
        id="deepseek_moe",
        name="DeepSeek-style shared-expert MoE",
        description=(
            "DeepSeek V3's routing design at its published width and depth: one shared "
            "expert every token passes through, plus 256 routed with 8 active. The shared "
            "expert absorbs what all tokens need so the routed ones can specialize. "
            "Attention is modelled as grouped-query; DeepSeek's multi-head latent attention "
            "compresses the KV projections in a way this canvas has no node for, so the "
            "total lands above the published 671B."
        ),
        layers=61, model_dim=7168, heads=128, kv_heads=128, ffn_dim=2048, vocab=129280,
        sequence=8192, _head_dim=128, experts=256, experts_per_token=8, shared_experts=1,
        source="DeepSeek-V3 technical report (arXiv 2412.19437), MLA not modelled",
    ),
]

PRESETS_BY_ID = {preset.id: preset for preset in PRESETS}


def preset_graph(preset: LlmPreset) -> ArchitectureGraph:
    """Build the canonical decoder-only graph for a preset.

    Every preset is the same six nodes — the differences are entirely in the
    transformer block's parameters, which is exactly the point: a 1B and a 70B
    model are the same architecture at different settings.
    """

    # Imported here rather than at module scope: `templates` imports this
    # module for the preset list, so a top-level import would be circular.
    from app.ml.architecture.templates import chain, node

    return auto_layout(
        ArchitectureGraph(
            nodes=[
                node("input", "input", shape=str(preset.sequence)),
                node(
                    "embed",
                    "embedding",
                    input_dim=preset.vocab,
                    output_dim=preset.model_dim,
                    mask_zero=False,
                ),
                node("rope", "rotary_embedding", base=10000.0),
                node(
                    "blocks",
                    "transformer_block",
                    layers=preset.layers,
                    num_heads=preset.heads,
                    num_kv_heads=preset.kv_heads,
                    key_dim=preset.head_dim,
                    qk_norm=preset.qk_norm,
                    sliding_window=preset.sliding_window,
                    global_every=preset.global_every,
                    ffn_dim=preset.ffn_dim,
                    norm="rms",
                    ffn="moe" if preset.is_sparse else "swiglu",
                    num_experts=preset.experts or 8,
                    experts_per_token=preset.experts_per_token,
                    shared_experts=preset.shared_experts,
                    causal=True,
                    dropout=0.0,
                ),
                node("final_norm", "rms_norm"),
                node(
                    "head",
                    "lm_head",
                    vocab_size=preset.vocab,
                    vocab_from_dataset=False,
                    tie_embeddings=preset.tie_embeddings,
                    logit_softcap=preset.logit_softcap,
                ),
                node("output", "output"),
            ],
            edges=chain("input", "embed", "rope", "blocks", "final_norm", "head", "output"),
            training_defaults={
                "epochs": 1,
                "batch_size": 1,
                "learning_rate": 0.0003,
                "optimizer": "adamw",
                "num_classes_preview": preset.vocab,
            },
        )
    )
