"""Named architecture blocks — the families, not the generic transformer.

Phase 17 shipped one `transformer_block` node that every LLM preset used, so a
Qwen graph and a DeepSeek graph were the same six nodes with different numbers.
That is true of *parameter counts* and false of *architectures*: Gemma
normalizes four times per layer, DeepSeek's attention is a pair of low-rank
projections rather than one, Mixtral's feed-forward is a router, and Qwen3
normalizes the query and key head vectors. This module is where those
differences live.

Each entry is a **node type in its own right**, so the palette offers
"Qwen3 block" rather than "transformer block, configured like Qwen3", and the
generated code branches on real structural parameters rather than carrying a
name in a comment. The defaults are read from the model's published
`config.json`; `source` records where, and `backend/tests/test_architecture_families.py`
asserts the parameter counts the configs imply.

The registries that consume this:

- `catalog.py` turns each entry into a `NodeSpec` with the shared param set.
- `shapes.py` reads `defaults` for validation and parameter estimation.
- `emit_keras.py` / `emit_torch.py` emit one helper per structure, called with
  the family's parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- attention kinds --------------------------------------------------------
MHA = "mha"
GQA = "gqa"
MLA = "mla"

# --- feed-forward kinds -----------------------------------------------------
SWIGLU = "swiglu"
GEGLU = "geglu"
GELU = "gelu"
MOE = "moe"

# --- normalization placement ------------------------------------------------
# Where the norms sit relative to each sub-layer:
#   pre       — norm(x) → sublayer → add.  Llama, Qwen, Mistral, GPT-2.
#   sandwich  — norm(x) → sublayer → norm → add.  Gemma 2/3's four norms.
#   post      — x → sublayer → add → norm.  The original transformer, BERT.
PRE = "pre"
SANDWICH = "sandwich"
POST = "post"

# --- MoE routing ------------------------------------------------------------
# softmax      — softmax over all experts, take top-k, renormalize (Mixtral).
# sigmoid_bias — sigmoid per expert plus a learned selection bias that is not
#                part of the gate value, DeepSeek V3's aux-loss-free balancing.
ROUTER_SOFTMAX = "softmax"
ROUTER_SIGMOID_BIAS = "sigmoid_bias"


@dataclass(frozen=True)
class BlockFamily:
    """One named decoder block, with the defaults its published config states."""

    type: str
    name: str
    description: str
    source: str
    defaults: dict[str, Any] = field(default_factory=dict)


# Every LLM block node carries this param set; a family only overrides values.
# Keeping one set means the emitters branch on structure rather than on a family
# name, so a user can turn a Llama block into a Mixtral one by switching the
# feed-forward kind — which is exactly what the two papers differ by.
LLM_BLOCK_DEFAULTS: dict[str, Any] = {
    "layers": 4,
    # attention
    "attention": GQA,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "qk_norm": False,
    # 0 derives the usual 1/sqrt(head_dim); Gemma states its own scalar.
    "query_scale": 0,
    "rope_theta": 500000.0,
    "sliding_window": 0,
    "global_every": 0,
    "causal": True,
    # GPT-2 and BERT carry a bias on every projection; every RoPE-era model
    # dropped them (`attention_bias: false` in Llama, Qwen, Mistral, Gemma and
    # DeepSeek configs alike), and at these widths that is not a rounding error.
    "use_bias": False,
    # multi-head latent attention (DeepSeek, Kimi)
    "q_lora_rank": 0,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "v_head_dim": 128,
    # feed-forward
    "ffn": SWIGLU,
    "ffn_dim": 14336,
    # mixture of experts
    "num_experts": 8,
    "experts_per_token": 2,
    "shared_experts": 0,
    "dense_layers": 0,
    "dense_ffn_dim": 18432,
    "router": ROUTER_SOFTMAX,
    "routed_scaling": 1.0,
    # normalization
    "norm": "rms",
    "norm_placement": PRE,
    "dropout": 0.0,
}


# Which parameters are rendered as which literal type when a block is emitted.
# Both emitters read these, so a parameter cannot be an int in the Keras module
# and a string in the torch one.
LLM_BLOCK_INT_PARAMS: tuple[str, ...] = (
    "layers",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "query_scale",
    "sliding_window",
    "global_every",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "qk_nope_head_dim",
    "v_head_dim",
    "ffn_dim",
    "num_experts",
    "experts_per_token",
    "shared_experts",
    "dense_layers",
    "dense_ffn_dim",
)
LLM_BLOCK_FLOAT_PARAMS: tuple[str, ...] = ("rope_theta", "routed_scaling", "dropout")
LLM_BLOCK_BOOL_PARAMS: tuple[str, ...] = ("qk_norm", "causal", "use_bias")
LLM_BLOCK_TEXT_PARAMS: tuple[str, ...] = ("attention", "ffn", "router", "norm", "norm_placement")


LLM_FAMILIES: list[BlockFamily] = [
    BlockFamily(
        type="llama_block",
        name="Llama block",
        description=(
            "The shape most open-weight models copied: pre-norm RMSNorm, grouped-query "
            "attention with rotary embeddings applied inside the attention, and a SwiGLU "
            "feed-forward. Defaults are Llama 3 8B."
        ),
        source="meta-llama/Meta-Llama-3-8B config.json",
        defaults={
            "layers": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "ffn_dim": 14336,
            "rope_theta": 500000.0,
        },
    ),
    BlockFamily(
        type="qwen3_block",
        name="Qwen3 block",
        description=(
            "Llama's layout plus QK-norm — an RMSNorm over each query and key head vector "
            "before the dot product, which is what keeps Qwen3's attention logits stable at "
            "depth. Head dim is 128 regardless of width. Defaults are Qwen3 8B."
        ),
        source="Qwen/Qwen3-8B config.json",
        defaults={
            "layers": 36,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "ffn_dim": 12288,
            "qk_norm": True,
            "rope_theta": 1000000.0,
        },
    ),
    BlockFamily(
        type="mistral_block",
        name="Mistral block",
        description=(
            "Grouped-query attention over a sliding window, so cost per token stops growing "
            "with context length. The window is Mistral 7B v0.1's 4096; later Mistral "
            "releases set it to null and attend globally."
        ),
        source="mistralai/Mistral-7B-v0.1 and v0.3 config.json",
        defaults={
            "layers": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "ffn_dim": 14336,
            "sliding_window": 4096,
            "rope_theta": 1000000.0,
        },
    ),
    BlockFamily(
        type="mixtral_block",
        name="Mixtral block (sparse MoE)",
        description=(
            "Mistral's attention with the feed-forward replaced by a router over eight "
            "experts, two active per token. Softmax over all experts, top-2, then "
            "renormalize — the ordering matters and this is Mixtral's."
        ),
        source="mistralai/Mixtral-8x7B-v0.1 config.json",
        defaults={
            "layers": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "ffn": MOE,
            "ffn_dim": 14336,
            "num_experts": 8,
            "experts_per_token": 2,
            "router": ROUTER_SOFTMAX,
            "rope_theta": 1000000.0,
        },
    ),
    BlockFamily(
        type="gemma3_block",
        name="Gemma 3 block",
        description=(
            "Four RMSNorms per layer, not two: each sub-layer is normalized on the way in "
            "and again on the way out before the residual adds. Plus QK-norm, a GeGLU "
            "feed-forward, a head dim of 256 against a 2560-wide stream, and five "
            "sliding-window layers to every global one."
        ),
        source="google/gemma-3-4b-it config.json (text_config)",
        defaults={
            "layers": 34,
            "num_heads": 8,
            "num_kv_heads": 4,
            "head_dim": 256,
            "qk_norm": True,
            "query_scale": 256,
            "ffn": GEGLU,
            "ffn_dim": 10240,
            "sliding_window": 1024,
            "global_every": 6,
            "norm_placement": SANDWICH,
            "rope_theta": 1000000.0,
        },
    ),
    BlockFamily(
        type="deepseek_block",
        name="DeepSeek V3 block (MLA + MoE)",
        description=(
            "Multi-head latent attention: queries and key/values pass through low-rank "
            "bottlenecks, so the cache holds a 512-wide latent instead of 128 full K/V "
            "heads. The feed-forward is one shared expert plus 256 routed with 8 active, "
            "selected by sigmoid scores with a learned balancing bias. The first three "
            "layers stay dense."
        ),
        source="deepseek-ai/DeepSeek-V3 config.json + arXiv 2412.19437",
        defaults={
            "layers": 61,
            "attention": MLA,
            "num_heads": 128,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "ffn": MOE,
            "ffn_dim": 2048,
            "dense_ffn_dim": 18432,
            "num_experts": 256,
            "experts_per_token": 8,
            "shared_experts": 1,
            "dense_layers": 3,
            "router": ROUTER_SIGMOID_BIAS,
            "routed_scaling": 2.5,
            "rope_theta": 10000.0,
        },
    ),
    BlockFamily(
        type="kimi_block",
        name="Kimi K2 block (MLA + wide MoE)",
        description=(
            "DeepSeek V3's block taken wider and thinner: the same latent attention at 64 "
            "heads instead of 128, and 384 routed experts instead of 256 at the same 8 "
            "active. One dense layer rather than three. Its config declares itself a "
            "DeepseekV3ForCausalLM, which is the honest description."
        ),
        source="moonshotai/Kimi-K2-Instruct config.json",
        defaults={
            "layers": 61,
            "attention": MLA,
            "num_heads": 64,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "ffn": MOE,
            "ffn_dim": 2048,
            "dense_ffn_dim": 18432,
            "num_experts": 384,
            "experts_per_token": 8,
            "shared_experts": 1,
            "dense_layers": 1,
            "router": ROUTER_SIGMOID_BIAS,
            "routed_scaling": 2.827,
            "rope_theta": 50000.0,
        },
    ),
    BlockFamily(
        type="gpt2_block",
        name="GPT-2 block",
        description=(
            "The pre-LayerNorm decoder everything else descends from: full multi-head "
            "attention, a two-layer GELU feed-forward at 4× width, and learned absolute "
            "positions supplied by a Positional embedding node rather than rotary ones."
        ),
        source="openai-community/gpt2 config.json",
        defaults={
            "layers": 12,
            "attention": MHA,
            "num_heads": 12,
            "num_kv_heads": 12,
            "head_dim": 64,
            "ffn": GELU,
            "ffn_dim": 3072,
            "use_bias": True,
            "norm": "layer",
            "norm_placement": PRE,
            "rope_theta": 0.0,
        },
    ),
    BlockFamily(
        type="bert_block",
        name="BERT encoder block",
        description=(
            "Bidirectional and post-norm — attention sees the whole sequence and the norm "
            "comes after the residual add, which is the original 2017 arrangement. Pair it "
            "with GlobalAvgPool1D and a Dense head for classification."
        ),
        source="google-bert/bert-base-uncased config.json",
        defaults={
            "layers": 12,
            "attention": MHA,
            "num_heads": 12,
            "num_kv_heads": 12,
            "head_dim": 64,
            "ffn": GELU,
            "ffn_dim": 3072,
            "use_bias": True,
            "norm": "layer",
            "norm_placement": POST,
            "causal": False,
            "rope_theta": 0.0,
            "dropout": 0.1,
        },
    ),
]

LLM_FAMILIES_BY_TYPE: dict[str, BlockFamily] = {family.type: family for family in LLM_FAMILIES}


def llm_block_params(node_type: str) -> dict[str, Any]:
    """Full resolved parameter set for one family's block node."""

    family = LLM_FAMILIES_BY_TYPE.get(node_type)
    return {**LLM_BLOCK_DEFAULTS, **(family.defaults if family else {})}


def is_llm_block(node_type: str) -> bool:
    return node_type in LLM_FAMILIES_BY_TYPE


# Markers the block-source builders read to decide which branches to emit. Not
# all of them name a helper class — `classic_ffn` and `layer_norm` describe code
# inside `llm_block` itself — and the emitters' helper lookups skip the ones that
# do not.
NORM_RMS = "rms_norm"
NORM_LAYER = "layer_norm"
CLASSIC_FFN = "classic_ffn"


def llm_block_structures(params: dict[str, Any]) -> set[str]:
    """Which generated structures a block's parameters actually reach.

    Both emitters read this to decide what to write out, and the block's own
    source is assembled from it — so a Gemma module has no MoE branch at all,
    rather than a branch guarded by a condition that is always false and a call
    to a class the file never defines.

    Keying on the parameters rather than the node type is the point: a generated
    file should describe the model it builds and nothing else. It also means a
    graph holding both a Gemma block and a Mixtral one emits both branches once,
    which is correct rather than merely tidy.
    """

    attention = str(params.get("attention", GQA))
    feed_forward = str(params.get("ffn", SWIGLU))
    norm = str(params.get("norm", "rms"))
    needed = {"attention_mask", "llm_block"}

    if attention == MLA:
        # Latent attention normalizes both of its bottlenecks with RMSNorm
        # regardless of what the residual stream uses.
        needed.update({"latent_attention", NORM_RMS})
    else:
        needed.add("family_attention")
        if params.get("qk_norm"):
            needed.add(NORM_RMS)
    needed.add(NORM_RMS if norm == "rms" else NORM_LAYER)

    # Both attention classes take `rope_theta` and branch on it, so the rotary
    # helpers come with them even when this model passes 0. Pruning them would
    # leave a live reference to a function the module never defines, and the
    # class is genuinely parameterized — a reader editing `rope_theta` in the
    # generated file should get a working model, not a NameError. That the model
    # does not use rotary is already stated by `rope_theta=0.0` at the call.
    needed.add("rope_apply")

    if feed_forward == MOE:
        # A router's experts are gated feed-forwards, and a leading dense layer
        # is one on its own.
        needed.update({"sparse_moe", "gated_ffn"})
        if int(params.get("dense_layers", 0) or 0):
            needed.add("gated_ffn")
    elif feed_forward in {SWIGLU, GEGLU}:
        needed.add("gated_ffn")
    else:
        needed.add(CLASSIC_FFN)
    return needed


# --- vision blocks ----------------------------------------------------------
#
# Unlike the LLM families these are structurally different from one another, so
# each gets its own parameter set rather than a shared one. Every default is the
# reference implementation's, named in `source`.

VISION_BLOCKS: list[BlockFamily] = [
    BlockFamily(
        type="resnet_block",
        name="ResNet stage",
        description=(
            "A stack of residual blocks with a projection shortcut on the first one, which "
            "is where the channel count and stride change. Basic is two 3×3 convolutions; "
            "bottleneck is 1×1 → 3×3 → 1×1 with a 4× expansion, which is what ResNet-50 and "
            "deeper use."
        ),
        source="He et al. 2015 (arXiv 1512.03385), torchvision BasicBlock / Bottleneck",
        defaults={
            "variant": "basic",
            "filters": 64,
            "blocks": 2,
            "stride": 1,
            "expansion": 4,
        },
    ),
    BlockFamily(
        type="inverted_residual_block",
        name="Inverted residual (MBConv)",
        description=(
            "MobileNetV2's block, and EfficientNet's with squeeze-excite on: expand 1×1, "
            "depthwise, optionally recalibrate channels, project 1×1 with no activation. "
            "The residual only connects when stride is 1 and the channel count is unchanged."
        ),
        source="MobileNetV2 (arXiv 1801.04381), torchvision mobilenetv3.InvertedResidual",
        defaults={
            "filters": 32,
            "expand_ratio": 6,
            "kernel_size": 3,
            "stride": 1,
            "use_se": False,
            "se_ratio": 4,
            "activation": "relu6",
            "blocks": 1,
        },
    ),
    BlockFamily(
        type="dense_block",
        name="DenseNet block",
        description=(
            "Every layer reads the concatenation of all previous layers' outputs and "
            "contributes `growth rate` more channels. BN → ReLU → 1×1 bottleneck → BN → "
            "ReLU → 3×3, which is DenseNet-BC."
        ),
        source="Huang et al. 2016 (arXiv 1608.06993), torchvision _DenseLayer",
        defaults={"growth_rate": 32, "layers": 6, "bottleneck_ratio": 4},
    ),
    BlockFamily(
        type="inception_block",
        name="Inception module",
        description=(
            "Four parallel paths at different receptive fields — 1×1, 1×1→3×3, 1×1→5×5, and "
            "3×3 pool→1×1 — concatenated on the channel axis. The 1×1 reductions are what "
            "make the wide paths affordable."
        ),
        source="Szegedy et al. 2014 (arXiv 1409.4842), GoogLeNet inception(3a) defaults",
        defaults={
            "filters_1x1": 64,
            "reduce_3x3": 96,
            "filters_3x3": 128,
            "reduce_5x5": 16,
            "filters_5x5": 32,
            "filters_pool": 32,
        },
    ),
    BlockFamily(
        type="convnext_block",
        name="ConvNeXt block",
        description=(
            "A transformer's layout built from convolutions: 7×7 depthwise, LayerNorm, a 4× "
            "pointwise expansion with GELU, a pointwise projection back, and a learned "
            "per-channel scale before the residual."
        ),
        source="Liu et al. 2022 (arXiv 2201.03545), facebookresearch/ConvNeXt Block",
        defaults={
            "filters": 96,
            "blocks": 3,
            "kernel_size": 7,
            "expand_ratio": 4,
            "layer_scale": 1e-6,
        },
    ),
    BlockFamily(
        type="vit_block",
        name="ViT encoder block",
        description=(
            "Pre-LayerNorm multi-head self-attention and a GELU MLP over a patch sequence, "
            "stacked N deep. Feed it a Patch embedding node; ViT-Base is 12 layers, 12 "
            "heads, 768 wide."
        ),
        source="Dosovitskiy et al. 2020 (arXiv 2010.11929), ViT-B/16",
        defaults={"layers": 12, "num_heads": 12, "head_dim": 64, "mlp_dim": 3072, "dropout": 0.0},
    ),
]

VISION_BLOCKS_BY_TYPE: dict[str, BlockFamily] = {block.type: block for block in VISION_BLOCKS}


# --- NLP blocks -------------------------------------------------------------
#
# The text-side counterparts to the vision blocks: multi-layer structures that
# are a whole encoder rather than one operation. They sit apart from the LLM
# families because neither is a decoder — a Text CNN has no attention at all,
# and a bidirectional recurrent encoder cannot be causal by construction — so
# they carry their own parameters instead of the shared decoder set.

NLP_BLOCKS: list[BlockFamily] = [
    BlockFamily(
        type="text_cnn_block",
        name="Text CNN (Kim)",
        description=(
            "Parallel 1D convolutions at several kernel widths over the token "
            "embeddings, each max-pooled over time and concatenated. Reads n-grams of "
            "every listed width at once, trains in minutes, and is still the baseline a "
            "text classifier has to beat."
        ),
        source="Kim 2014 (arXiv 1408.5882), kernel widths 3/4/5 at 100 filters each",
        defaults={
            "filters": 128,
            "kernel_sizes": "3,4,5",
            "activation": "relu",
            "dropout": 0.0,
        },
    ),
    BlockFamily(
        type="bilstm_encoder",
        name="BiLSTM encoder",
        description=(
            "Stacked bidirectional recurrent layers, so every token is read with both "
            "its left and right context. Returning the sequence feeds a pooling or "
            "tagging head; turning that off emits one vector per document."
        ),
        source="Graves & Schmidhuber 2005; the standard pre-transformer text encoder",
        defaults={
            "cell": "lstm",
            "units": 128,
            "layers": 2,
            "dropout": 0.2,
            "return_sequences": True,
        },
    ),
]

NLP_BLOCKS_BY_TYPE: dict[str, BlockFamily] = {block.type: block for block in NLP_BLOCKS}


BLOCK_TYPES: frozenset[str] = frozenset(
    {*LLM_FAMILIES_BY_TYPE, *VISION_BLOCKS_BY_TYPE, *NLP_BLOCKS_BY_TYPE, "transformer_block"}
)
