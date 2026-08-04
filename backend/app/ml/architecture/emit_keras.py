"""Keras code generation: graph IR → a standalone runnable Python module.

The emitted file is both the user-facing "export to code" artifact and what
the training runner actually imports and builds, so there is exactly one
definition of what a graph means. Generation is deterministic — variable names
come from the graph's topological order — so regenerating an unchanged graph
produces a byte-identical file and diffs stay meaningful.

Emitters are keyed by the same `type` string as `catalog.NODE_SPECS` and
`shapes.SHAPE_RULES`; a test asserts the three key sets match.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.ml.architecture.blocks import (
    LLM_BLOCK_BOOL_PARAMS,
    LLM_BLOCK_DEFAULTS,
    LLM_BLOCK_FLOAT_PARAMS,
    LLM_BLOCK_INT_PARAMS,
    LLM_BLOCK_TEXT_PARAMS,
    LLM_FAMILIES,
    is_llm_block,
    llm_block_structures,
)
from app.ml.architecture.graph import (
    OUTPUT_TYPE,
    ResolvedGraph,
    ResolvedNode,
    parse_int_list,
    parse_shape,
)
from app.ml.architecture.keras_helpers import (
    BI_RNN_ENCODER_HELPER,
    CAUSAL_MASK_HELPER,
    CONVNEXT_HELPER,
    DENSE_BLOCK_HELPER,
    FAMILY_ATTENTION_HELPER,
    GATED_FFN_HELPER,
    INCEPTION_HELPER,
    INVERTED_RESIDUAL_HELPER,
    LATENT_ATTENTION_HELPER,
    LAYER_SCALE_HELPER,
    PATCH_EMBEDDING_HELPER,
    RESNET_HELPER,
    ROPE_APPLY_HELPER,
    SEQUENCE_POOL_HELPER,
    SINUSOIDAL_HELPER,
    SPARSE_MOE_HELPER,
    SQUEEZE_EXCITE_HELPER,
    TEXT_CNN_HELPER,
    VIT_HELPER,
    llm_block_source,
)
from app.ml.architecture.shapes import Shape

MODULE_FILENAME = "generated_model.py"


class EmitError(RuntimeError):
    """The graph cannot be turned into code — always a validation gap upstream."""


@dataclass(frozen=True)
class EmitContext:
    node: ResolvedNode
    # Python variable names of this node's inputs, in edge order.
    inputs: list[str]
    input_shapes: list[Shape]
    # Layer name, which doubles as the node's variable name so a row in
    # model.summary() maps back to a node on the canvas by eye.
    name: str
    # Variable names of every Embedding layer in the graph. A tied LM head
    # projects with one of these tables instead of learning its own.
    embedding_vars: tuple[str, ...] = ()

    @property
    def params(self) -> dict[str, Any]:
        return self.node.params

    @property
    def first(self) -> str:
        return self.inputs[0]


def emit_module(
    resolved: ResolvedGraph,
    shapes: dict[str, Shape] | None = None,
    *,
    architecture_name: str = "model",
    architecture_id: str = "",
    version: int = 1,
    default_num_classes: int = 2,
) -> str:
    """Render the full `generated_model.py` for a resolved graph.

    Raises `EmitError` if the graph still has structural errors; callers
    validate first and surface those as issues rather than as a traceback.
    """

    if not resolved.ok:
        raise EmitError(
            "Cannot generate code for a graph with errors: "
            + "; ".join(issue.message for issue in resolved.errors())
        )
    if not resolved.order:
        raise EmitError("Cannot generate code for an empty graph.")

    shapes = shapes or {}
    model_name = _identifier(architecture_name) or "model"
    # An Output node emits no layer of its own; it forwards its input's
    # variable so `outputs=` points at the last real layer.
    aliases: dict[str, str] = {}
    body: list[str] = []
    # Collected up front: a tied LM head refers back to an embedding layer
    # that was emitted earlier in the same function.
    embedding_vars = tuple(
        resolved.nodes[node_id].var_name
        for node_id in resolved.order
        if resolved.nodes[node_id].type == "embedding"
    )
    tied_head = any(
        resolved.nodes[node_id].type == "lm_head"
        and resolved.nodes[node_id].params.get("tie_embeddings")
        for node_id in resolved.order
    )

    for node_id in resolved.order:
        node = resolved.nodes[node_id]
        input_vars = [aliases.get(source, resolved.nodes[source].var_name) for source in node.inputs]
        if node.type == OUTPUT_TYPE:
            if not input_vars:
                raise EmitError("The Output node has nothing connected to it.")
            aliases[node_id] = input_vars[0]
            continue
        emitter = EMITTERS.get(node.type)
        if emitter is None:
            raise EmitError(f"No code generator for node type {node.type!r}.")
        context = EmitContext(
            node=node,
            inputs=input_vars,
            input_shapes=[shapes.get(source) for source in node.inputs],
            name=node.var_name,
            embedding_vars=embedding_vars,
        )
        expression = emitter(context)
        if node.type == "embedding" and tied_head:
            # A tied head projects with this layer's table, so the layer itself
            # has to survive as a name — the tensor alone is not enough.
            constructor, _, applied = expression.rpartition("(")
            body.append(f"    {node.var_name}_layer = {constructor}")
            body.append(f"    {node.var_name} = {node.var_name}_layer({applied}")
            continue
        body.append(f"    {node.var_name} = {expression}")

    input_vars = [resolved.nodes[node_id].var_name for node_id in resolved.input_ids]
    output_vars = [aliases[node_id] for node_id in resolved.output_ids if node_id in aliases]
    if not input_vars or not output_vars:
        raise EmitError("The graph needs a connected Input node and Output node.")

    inputs_arg = input_vars[0] if len(input_vars) == 1 else "[" + ", ".join(input_vars) + "]"
    header = _header(architecture_name, architecture_id, version)
    required: set[str] = set()
    for node_id in resolved.order:
        node = resolved.nodes[node_id]
        required.update(HELPER_DEPENDENCIES.get(node.type, ()))
        # A family block's helpers follow its parameters, not its name: a Gemma
        # module should not carry DeepSeek's latent attention, and a GPT-2 one
        # should not carry rotary helpers it never calls.
        if is_llm_block(node.type):
            required.update(llm_block_structures(node.params))
        # The LM head needs a helper only when tying or softcapping is on, so
        # a plain head keeps the generated file free of both.
        if node.type == "lm_head":
            if node.params.get("tie_embeddings"):
                required.add("tied_head")
            elif float(node.params.get("logit_softcap", 0.0) or 0.0):
                required.add("softcap")
    helpers: list[str] = []
    for helper in HELPER_ORDER:
        if helper not in required:
            continue
        # `llm_block` is assembled rather than looked up: its branches are
        # pruned to the structures this graph reaches, so it never calls a
        # helper class the module has not defined.
        source = llm_block_source(required) if helper == "llm_block" else HELPER_SOURCE[helper]
        helpers.extend([source, "", ""])
    helpers.extend(_custom_definitions(resolved))
    return "\n".join(
        [
            *header,
            "",
            "import tensorflow as tf",
            "",
            "",
            *helpers,
            f"def build_model(num_classes: int = {default_num_classes}) -> tf.keras.Model:",
            f'    """Build the {architecture_name!r} architecture as a Keras model."""',
            "",
            *body,
            "    return tf.keras.Model(",
            f"        inputs={inputs_arg},",
            f"        outputs={output_vars[0]},",
            f"        name={_lit(model_name)},",
            "    )",
            "",
            "",
            'if __name__ == "__main__":',
            "    build_model().summary()",
            "",
        ]
    )


def _header(architecture_name: str, architecture_id: str, version: int) -> list[str]:
    origin = f"{architecture_name!r}"
    if architecture_id:
        origin += f" ({architecture_id}, v{version})"
    return [
        f"# Generated by Onestep AI Platform from architecture {origin}.",
        "# Regenerated from the visual graph on every export, compile, and training",
        "# run — edits made here are not read back into the studio.",
    ]



# --- generated helper classes ----------------------------------------------
#
# Keras ships no RMSNorm, RoPE, or SwiGLU layer, so the graph emits them. They
# are written into the module only when the graph actually uses them, keeping
# a plain CNN's generated file free of transformer machinery.

BACKBONE_HELPER = '''def apply_backbone(backbone, inputs, *, trainable: bool):
    """Apply a tf.keras.applications backbone to a tensor.

    A frozen backbone is called with training=False so its BatchNormalization
    layers stay in inference mode. Without that, the moving statistics drift
    during fine-tuning and transfer learning quietly fails to converge.
    """

    backbone.trainable = trainable
    return backbone(inputs) if trainable else backbone(inputs, training=False)'''

RMS_NORM_HELPER = '''@tf.keras.utils.register_keras_serializable(package="onestep")
class RMSNorm(tf.keras.layers.Layer):
    """Root-mean-square normalization: LayerNorm without the mean subtraction."""

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", shape=(input_shape[-1],), initializer="ones", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        variance = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        return inputs * tf.math.rsqrt(variance + self.epsilon) * self.scale

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon}'''

ROPE_HELPER = '''@tf.keras.utils.register_keras_serializable(package="onestep")
class RotaryEmbedding(tf.keras.layers.Layer):
    """Rotary position embedding.

    Rotates feature pairs by an angle proportional to position, so attention
    sees relative distance rather than absolute index. Requires an even final
    dimension.
    """

    def __init__(self, base=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.base = base

    def call(self, inputs):
        shape = tf.shape(inputs)
        length, width = shape[1], shape[2]
        half = width // 2
        positions = tf.cast(tf.range(length), tf.float32)[:, None]
        frequencies = tf.pow(
            tf.cast(self.base, tf.float32),
            -tf.cast(tf.range(half), tf.float32) * 2.0 / tf.cast(width, tf.float32),
        )[None, :]
        angles = positions * frequencies
        cos, sin = tf.cos(angles), tf.sin(angles)
        first, second = inputs[..., :half], inputs[..., half:]
        return tf.concat([first * cos - second * sin, first * sin + second * cos], axis=-1)

    def get_config(self):
        return {**super().get_config(), "base": self.base}'''

SWIGLU_HELPER = '''@tf.keras.utils.register_keras_serializable(package="onestep")
class SwiGLU(tf.keras.layers.Layer):
    """Gated feed-forward block: silu(gate(x)) * up(x), projected back down."""

    def __init__(self, hidden_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        width = input_shape[-1]
        self.gate = tf.keras.layers.Dense(self.hidden_dim, name="gate")
        self.up = tf.keras.layers.Dense(self.hidden_dim, name="up")
        self.down = tf.keras.layers.Dense(width, name="down")
        self.drop = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs, training=None):
        hidden = tf.nn.silu(self.gate(inputs)) * self.up(inputs)
        return self.down(self.drop(hidden, training=training))

    def get_config(self):
        return {
            **super().get_config(),
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
        }'''

POSITIONAL_HELPER = '''@tf.keras.utils.register_keras_serializable(package="onestep")
class PositionalEmbedding(tf.keras.layers.Layer):
    """Learned absolute position vectors added to the incoming sequence."""

    def __init__(self, max_length, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length

    def build(self, input_shape):
        self.positions = self.add_weight(
            name="positions",
            shape=(self.max_length, input_shape[-1]),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.positions[: tf.shape(inputs)[1]]

    def get_config(self):
        return {**super().get_config(), "max_length": self.max_length}'''

GQA_HELPER = '''@tf.keras.utils.register_keras_serializable(package="onestep")
class GroupedQueryAttention(tf.keras.layers.Layer):
    """Attention with fewer key/value heads than query heads.

    Query heads are split into groups that share one K/V head, which shrinks
    the KV cache by the grouping factor at nearly no quality cost. This is what
    Llama 3, Qwen, and Mistral use.
    """

    def __init__(self, num_heads, num_kv_heads, key_dim, causal=True, dropout=0.0,
                 qk_norm=False, sliding_window=0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.key_dim = key_dim
        self.causal = causal
        self.dropout_rate = dropout
        self.qk_norm = qk_norm
        self.sliding_window = sliding_window

    def build(self, input_shape):
        width = input_shape[-1]
        self.q = tf.keras.layers.Dense(self.num_heads * self.key_dim, name="q")
        self.k = tf.keras.layers.Dense(self.num_kv_heads * self.key_dim, name="k")
        self.v = tf.keras.layers.Dense(self.num_kv_heads * self.key_dim, name="v")
        self.o = tf.keras.layers.Dense(width, name="o")
        self.drop = tf.keras.layers.Dropout(self.dropout_rate)
        # QK-norm normalizes each head's query and key vectors before the dot
        # product, which is what keeps attention logits from drifting at depth.
        # Qwen3 and Gemma 3+ both do this on the head dimension.
        self.q_norm = RMSNorm(name="q_norm") if self.qk_norm else None
        self.k_norm = RMSNorm(name="k_norm") if self.qk_norm else None
        super().build(input_shape)

    def call(self, inputs, training=None):
        batch = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        groups = self.num_heads // self.num_kv_heads

        def split(x, heads):
            x = tf.reshape(x, (batch, length, heads, self.key_dim))
            return tf.transpose(x, (0, 2, 1, 3))

        query = split(self.q(inputs), self.num_heads)
        key = split(self.k(inputs), self.num_kv_heads)
        value = split(self.v(inputs), self.num_kv_heads)
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        # Each K/V head serves `groups` query heads.
        key = tf.repeat(key, groups, axis=1)
        value = tf.repeat(value, groups, axis=1)

        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(
            tf.cast(self.key_dim, inputs.dtype)
        )
        mask = None
        if self.causal:
            mask = tf.linalg.band_part(tf.ones((length, length), dtype=tf.bool), -1, 0)
        if self.sliding_window:
            # Local attention: a token sees only the last `sliding_window`
            # positions. Gemma 4 runs most layers this way and promotes every
            # sixth to full attention.
            positions = tf.range(length)
            within = positions[:, None] - positions[None, :] < self.sliding_window
            mask = within if mask is None else tf.logical_and(mask, within)
        if mask is not None:
            scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))
        weights = self.drop(tf.nn.softmax(scores, axis=-1), training=training)
        context = tf.matmul(weights, value)
        context = tf.transpose(context, (0, 2, 1, 3))
        return self.o(tf.reshape(context, (batch, length, self.num_heads * self.key_dim)))

    def get_config(self):
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "key_dim": self.key_dim,
            "causal": self.causal,
            "dropout": self.dropout_rate,
            "qk_norm": self.qk_norm,
            "sliding_window": self.sliding_window,
        }'''

MOE_HELPER = '''@tf.keras.utils.register_keras_serializable(package="onestep")
class MixtureOfExperts(tf.keras.layers.Layer):
    """Sparse feed-forward: a router sends each token to its top-k experts.

    Total parameters grow with the expert count while the cost per token grows
    only with `experts_per_token` — the trade Mixtral and DeepSeek are built on.
    This is a readable dense-gather implementation, not a production kernel:
    every expert runs on every token and the unselected results are masked out,
    so it is correct and slow rather than fast.
    """

    def __init__(self, num_experts, experts_per_token, hidden_dim, dropout=0.0,
                 shared_experts=0, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.experts_per_token = min(experts_per_token, num_experts)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.shared_experts = shared_experts

    def build(self, input_shape):
        width = input_shape[-1]
        self.router = tf.keras.layers.Dense(self.num_experts, name="router")
        # Gated experts, as every production mixture-of-experts uses.
        self.experts = [
            SwiGLU(self.hidden_dim, dropout=self.dropout_rate, name=f"expert_{index}")
            for index in range(self.num_experts)
        ]
        # Shared experts run on every token, unrouted, alongside the top-k.
        # DeepSeek V3 pairs 1 shared with 256 routed: the shared expert absorbs
        # what every token needs so the routed ones can specialize.
        self.shared = [
            SwiGLU(self.hidden_dim, dropout=self.dropout_rate, name=f"shared_{index}")
            for index in range(self.shared_experts)
        ]
        super().build(input_shape)

    def call(self, inputs, training=None):
        logits = self.router(inputs)
        top_values, top_indices = tf.math.top_k(logits, k=self.experts_per_token)
        gates = tf.nn.softmax(top_values, axis=-1)
        # Scatter the top-k gates back over all experts; unselected weight is 0.
        weights = tf.reduce_sum(
            tf.one_hot(top_indices, self.num_experts) * gates[..., None], axis=-2
        )
        outputs = tf.stack(
            [expert(inputs, training=training) for expert in self.experts], axis=-2
        )
        routed = tf.reduce_sum(outputs * weights[..., None], axis=-2)
        for expert in self.shared:
            routed += expert(inputs, training=training)
        return routed

    def get_config(self):
        return {
            **super().get_config(),
            "num_experts": self.num_experts,
            "experts_per_token": self.experts_per_token,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_rate,
            "shared_experts": self.shared_experts,
        }'''

TIED_HEAD_HELPER = '''@tf.keras.utils.register_keras_serializable(package="onestep")
class TiedLMHead(tf.keras.layers.Layer):
    """Vocabulary projection that reuses the input embedding table.

    Weight tying: instead of learning a second `[width, vocab]` matrix, project
    with the transpose of the embedding already in the model. Gemma and the
    smaller Qwen models do this — at a 262k vocabulary it removes hundreds of
    millions of parameters and is a real regularizer besides.

    The embedding layer is held from construction rather than passed through
    `call`: Keras treats every call argument as a tensor and cannot infer an
    output shape when handed a Layer.
    """

    def __init__(self, vocab_size, embedding_layer, softcap=0.0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_layer = embedding_layer
        self.softcap = softcap

    def call(self, inputs):
        table = self.embedding_layer.embeddings[: self.vocab_size]
        logits = tf.matmul(inputs, table, transpose_b=True)
        if self.softcap:
            logits = tf.tanh(logits / self.softcap) * self.softcap
        return logits

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.vocab_size)

    def get_config(self):
        return {**super().get_config(), "vocab_size": self.vocab_size, "softcap": self.softcap}'''

SOFTCAP_HELPER = '''@tf.keras.utils.register_keras_serializable(package="onestep")
class SoftcappedDense(tf.keras.layers.Layer):
    """Vocabulary projection whose logits are bounded by tanh.

    `tanh(logit / cap) * cap` keeps the output distribution from saturating,
    which is what Gemma's `final_logit_softcapping` does.
    """

    def __init__(self, vocab_size, softcap, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.softcap = softcap

    def build(self, input_shape):
        self.projection = tf.keras.layers.Dense(self.vocab_size, name="projection")
        super().build(input_shape)

    def call(self, inputs):
        logits = self.projection(inputs)
        return tf.tanh(logits / self.softcap) * self.softcap

    def get_config(self):
        return {**super().get_config(), "vocab_size": self.vocab_size, "softcap": self.softcap}'''

TRANSFORMER_BLOCK_HELPER = '''def transformer_block(
    x,
    *,
    layers,
    num_heads,
    key_dim,
    ffn_dim,
    num_kv_heads=0,
    norm="rms",
    ffn="swiglu",
    num_experts=8,
    experts_per_token=2,
    shared_experts=0,
    qk_norm=False,
    sliding_window=0,
    global_every=0,
    causal=True,
    dropout=0.0,
    name="block",
):
    """A stack of pre-norm transformer layers, each with its own weights.

    Pre-norm (normalize before attention rather than after) is what makes deep
    stacks trainable without a warmup schedule.
    """

    def make_norm(norm_name):
        if norm == "rms":
            return RMSNorm(name=norm_name)
        return tf.keras.layers.LayerNormalization(epsilon=1e-6, name=norm_name)

    for index in range(layers):
        prefix = f"{name}_{index + 1}"
        # Hybrid attention: most layers see a local window, every `global_every`
        # -th sees the whole sequence. Gemma 4 runs 5 local to 1 global, which
        # keeps long-context cost down without losing global reach.
        is_global = bool(global_every) and (index + 1) % global_every == 0
        window = 0 if (is_global or not sliding_window) else sliding_window

        residual = x
        h = make_norm(f"{prefix}_attn_norm")(x)
        if (num_kv_heads and num_kv_heads != num_heads) or qk_norm or window:
            h = GroupedQueryAttention(
                num_heads,
                num_kv_heads or num_heads,
                key_dim,
                causal=causal,
                dropout=dropout,
                qk_norm=qk_norm,
                sliding_window=window,
                name=f"{prefix}_attn",
            )(h)
        else:
            h = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=key_dim,
                dropout=dropout,
                name=f"{prefix}_attn",
            )(h, h, use_causal_mask=causal)
        x = tf.keras.layers.Add(name=f"{prefix}_attn_residual")([residual, h])

        residual = x
        h = make_norm(f"{prefix}_ffn_norm")(x)
        if ffn == "moe":
            h = MixtureOfExperts(
                num_experts, experts_per_token, ffn_dim, dropout=dropout,
                shared_experts=shared_experts, name=f"{prefix}_ffn",
            )(h)
        elif ffn == "swiglu":
            h = SwiGLU(ffn_dim, dropout=dropout, name=f"{prefix}_ffn")(h)
        else:
            width = x.shape[-1]
            h = tf.keras.layers.Dense(ffn_dim, activation="gelu", name=f"{prefix}_ffn_up")(h)
            h = tf.keras.layers.Dropout(dropout, name=f"{prefix}_ffn_drop")(h)
            h = tf.keras.layers.Dense(width, name=f"{prefix}_ffn_down")(h)
        x = tf.keras.layers.Add(name=f"{prefix}_ffn_residual")([residual, h])
    return x'''

# Node type -> the helpers its generated code depends on, in emission order.
# Family blocks are absent here: their helpers depend on how they are configured
# rather than on which family they are, so `emit_module` asks
# `blocks.llm_block_structures` per node. Two blocks that reach the same
# structure still emit one copy of the class.
HELPER_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "pretrained_backbone": ("backbone",),
    "rms_norm": ("rms_norm",),
    "rotary_embedding": ("rope",),
    "swiglu": ("swiglu",),
    "positional_embedding": ("positional",),
    "transformer_block": ("rms_norm", "swiglu", "gqa", "moe", "transformer_block"),
    "grouped_query_attention": ("rms_norm", "gqa"),
    "moe_feed_forward": ("swiglu", "moe"),
    # new primitives
    "squeeze_excite": ("squeeze_excite",),
    "patch_embedding": ("patch_embedding",),
    "geglu": ("gated_ffn",),
    "mla_attention": ("rms_norm", "rope_apply", "attention_mask", "latent_attention"),
    # vision blocks
    "resnet_block": ("resnet",),
    "inverted_residual_block": ("squeeze_excite", "inverted_residual"),
    "dense_block": ("dense_block",),
    "inception_block": ("inception",),
    "convnext_block": ("layer_scale", "convnext"),
    "vit_block": ("vit",),
    # NLP
    "sinusoidal_position_encoding": ("sinusoidal",),
    "sequence_pool": ("sequence_pool",),
    "text_cnn_block": ("text_cnn",),
    "bilstm_encoder": ("bi_rnn_encoder",),
}

HELPER_SOURCE: dict[str, str] = {
    "backbone": BACKBONE_HELPER,
    "rms_norm": RMS_NORM_HELPER,
    "rope": ROPE_HELPER,
    "rope_apply": ROPE_APPLY_HELPER,
    "attention_mask": CAUSAL_MASK_HELPER,
    "swiglu": SWIGLU_HELPER,
    "gated_ffn": GATED_FFN_HELPER,
    "positional": POSITIONAL_HELPER,
    "gqa": GQA_HELPER,
    "family_attention": FAMILY_ATTENTION_HELPER,
    "latent_attention": LATENT_ATTENTION_HELPER,
    "moe": MOE_HELPER,
    "sparse_moe": SPARSE_MOE_HELPER,
    "tied_head": TIED_HEAD_HELPER,
    "softcap": SOFTCAP_HELPER,
    "transformer_block": TRANSFORMER_BLOCK_HELPER,
    # `llm_block` is built by `llm_block_source` from the required set rather
    # than looked up here; the key still appears in HELPER_ORDER for position.
    "squeeze_excite": SQUEEZE_EXCITE_HELPER,
    "patch_embedding": PATCH_EMBEDDING_HELPER,
    "layer_scale": LAYER_SCALE_HELPER,
    "resnet": RESNET_HELPER,
    "inverted_residual": INVERTED_RESIDUAL_HELPER,
    "dense_block": DENSE_BLOCK_HELPER,
    "inception": INCEPTION_HELPER,
    "convnext": CONVNEXT_HELPER,
    "vit": VIT_HELPER,
    "sinusoidal": SINUSOIDAL_HELPER,
    "sequence_pool": SEQUENCE_POOL_HELPER,
    "text_cnn": TEXT_CNN_HELPER,
    "bi_rnn_encoder": BI_RNN_ENCODER_HELPER,
}
# Emission order, so a helper never references one defined below it.
HELPER_ORDER = [
    "backbone",
    "rms_norm",
    "rope",
    "rope_apply",
    "attention_mask",
    "swiglu",
    "gated_ffn",
    "positional",
    "squeeze_excite",
    "patch_embedding",
    "layer_scale",
    "gqa",
    "family_attention",
    "latent_attention",
    "moe",
    "sparse_moe",
    "tied_head",
    "softcap",
    "transformer_block",
    "llm_block",
    "resnet",
    "inverted_residual",
    "dense_block",
    "inception",
    "convnext",
    "vit",
    "sinusoidal",
    "sequence_pool",
    "text_cnn",
    "bi_rnn_encoder",
]


# --- literal and call helpers ----------------------------------------------


def _lit(value: Any) -> str:
    """Render a Python literal, preferring double quotes for strings."""

    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        items = ", ".join(_lit(item) for item in value)
        # A one-element tuple needs its trailing comma to stay a tuple.
        return f"({items},)" if len(value) == 1 else f"({items})"
    return json.dumps(value)


def _identifier(text: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    if slug and slug[0].isdigit():
        slug = f"m_{slug}"
    return slug


def _layer(
    layer: str, *args: Any, name: str, applied_to: str, **kwargs: Any
) -> str:
    """`tf.keras.layers.<layer>(<args>, <kwargs>, name=...)(<applied_to>)`."""

    rendered = [_lit(arg) for arg in args]
    rendered += [f"{key}={_lit(value)}" for key, value in kwargs.items() if value is not None]
    rendered.append(f"name={_lit(name)}")
    return f"tf.keras.layers.{layer}({', '.join(rendered)})({applied_to})"


def _identity_layer(layer: str, **kwargs: Any) -> Callable[[EmitContext], str]:
    def emit(ctx: EmitContext) -> str:
        values = {key: ctx.params.get(key) for key in kwargs}
        return _layer(layer, name=ctx.name, applied_to=ctx.first, **values)

    return emit


def _emit_random_zoom(ctx: EmitContext) -> str:
    """RandomZoom's first argument is `height_factor`, not `factor`.

    Passing `width_factor=None` is what makes the zoom isotropic — Keras then
    reuses the height factor, so the image scales without changing its aspect
    ratio. Naming the canvas param `factor` keeps it consistent with the other
    three augmentation nodes.
    """

    return _layer(
        "RandomZoom",
        name=ctx.name,
        applied_to=ctx.first,
        height_factor=float(ctx.params["factor"]),
    )


def _emit_random_contrast(ctx: EmitContext) -> str:
    """RandomContrast needs the pipeline's value range stated.

    Its default is `(0, 255)`, but `keras_common.image_datasets` divides by 255
    before batching, so the tensors reaching this layer are floats in [0, 1].
    Left at the default the layer's clip is a no-op against the wrong scale;
    stating the real range is what makes the augmentation match the data.
    """

    return _layer(
        "RandomContrast",
        name=ctx.name,
        applied_to=ctx.first,
        factor=float(ctx.params["factor"]),
        value_range=(0.0, 1.0),
    )


# --- emitters --------------------------------------------------------------


def _emit_input(ctx: EmitContext) -> str:
    shape = parse_shape(ctx.params.get("shape"))
    if shape is None:
        raise EmitError(f"Node {ctx.node.id!r} has an unparseable input shape.")
    return f"tf.keras.Input(shape={_lit(shape)}, name={_lit(ctx.name)})"


def _emit_dense(ctx: EmitContext) -> str:
    units = "num_classes" if ctx.params.get("units_from_dataset") else _lit(int(ctx.params["units"]))
    parts = [
        units,
        f"activation={_lit(ctx.params['activation'])}",
        f"use_bias={_lit(bool(ctx.params['use_bias']))}",
        f"name={_lit(ctx.name)}",
    ]
    return f"tf.keras.layers.Dense({', '.join(parts)})({ctx.first})"


def _emit_activation(ctx: EmitContext) -> str:
    return _layer("Activation", ctx.params["activation"], name=ctx.name, applied_to=ctx.first)


def _emit_flatten(ctx: EmitContext) -> str:
    return _layer("Flatten", name=ctx.name, applied_to=ctx.first)


def _emit_reshape(ctx: EmitContext) -> str:
    shape = parse_shape(ctx.params.get("target_shape"))
    if shape is None:
        raise EmitError(f"Node {ctx.node.id!r} has an unparseable target shape.")
    return _layer("Reshape", shape, name=ctx.name, applied_to=ctx.first)


def _emit_embedding(ctx: EmitContext) -> str:
    return _layer(
        "Embedding",
        int(ctx.params["input_dim"]),
        int(ctx.params["output_dim"]),
        name=ctx.name,
        applied_to=ctx.first,
        mask_zero=bool(ctx.params["mask_zero"]),
    )


def _emit_conv(layer: str) -> Callable[[EmitContext], str]:
    def emit(ctx: EmitContext) -> str:
        return _layer(
            layer,
            int(ctx.params["filters"]),
            int(ctx.params["kernel_size"]),
            name=ctx.name,
            applied_to=ctx.first,
            strides=int(ctx.params["strides"]),
            padding=ctx.params["padding"],
            activation=ctx.params["activation"],
        )

    return emit


def _emit_pool(layer: str) -> Callable[[EmitContext], str]:
    def emit(ctx: EmitContext) -> str:
        pool = int(ctx.params["pool_size"])
        # The catalog spells "match the pool size" as 0, because
        # AdvancedParameterSpec has no unset value for a number; Keras spells
        # it as None.
        strides = int(ctx.params["strides"]) or None
        return _layer(
            layer,
            pool,
            name=ctx.name,
            applied_to=ctx.first,
            strides=strides,
            padding=ctx.params["padding"],
        )

    return emit


def _emit_recurrent(layer: str) -> Callable[[EmitContext], str]:
    """LSTM/GRU, optionally wrapped in Bidirectional.

    Bidirectionality is a param rather than a separate wrapper node: a node
    that wraps another node has no natural representation on a flat canvas,
    and the only wrapper Keras users reach for here is this one.
    """

    def emit(ctx: EmitContext) -> str:
        args = [
            _lit(int(ctx.params["units"])),
            f"return_sequences={_lit(bool(ctx.params['return_sequences']))}",
            f"dropout={_lit(float(ctx.params['dropout']))}",
        ]
        if ctx.params.get("bidirectional"):
            inner = f"tf.keras.layers.{layer}({', '.join(args)})"
            return (
                f"tf.keras.layers.Bidirectional({inner}, name={_lit(ctx.name)})({ctx.first})"
            )
        args.append(f"name={_lit(ctx.name)}")
        return f"tf.keras.layers.{layer}({', '.join(args)})({ctx.first})"

    return emit


def _emit_merge(layer: str) -> Callable[[EmitContext], str]:
    def emit(ctx: EmitContext) -> str:
        operands = "[" + ", ".join(ctx.inputs) + "]"
        if layer == "Concatenate":
            return (
                f"tf.keras.layers.Concatenate(axis={_lit(int(ctx.params['axis']))}, "
                f"name={_lit(ctx.name)})({operands})"
            )
        return f"tf.keras.layers.{layer}(name={_lit(ctx.name)})({operands})"

    return emit


def _emit_backbone(ctx: EmitContext) -> str:
    """Instantiate a `tf.keras.applications` backbone and apply it.

    `input_shape` comes from the inferred shape of whatever feeds the node, so
    a backbone works at whatever resolution the graph uses rather than only at
    its own default.
    """

    shape = ctx.input_shapes[0] if ctx.input_shapes else None
    if shape is None or any(dim is None for dim in shape):
        raise EmitError(
            f"Node {ctx.node.id!r}: a pretrained backbone needs a fully known input shape. "
            "Check the Input node feeding it."
        )
    weights = ctx.params["weights"]
    pooling = ctx.params["pooling"]
    # Deliberately no `name=`: tf.keras.applications derives the weights
    # download filename from the model's name, so naming the backbone after
    # its canvas node sends Keras looking for `<node>_notop.h5` and the
    # ImageNet download 403s. The backbone keeps its stock name in
    # model.summary(); the generated variable still identifies it in code.
    application = (
        f"tf.keras.applications.{ctx.params['application']}("
        f"include_top=False, "
        f"weights={_lit(None if weights == 'none' else weights)}, "
        f"pooling={_lit(None if pooling == 'none' else pooling)}, "
        f"input_shape={_lit(shape)})"
    )
    trainable = _lit(bool(ctx.params["trainable"]))
    return f"apply_backbone({application}, {ctx.first}, trainable={trainable})"





# --- transformer and custom emitters ---------------------------------------


def _emit_positional(ctx: EmitContext) -> str:
    return (
        f"PositionalEmbedding({int(ctx.params['max_length'])}, "
        f"name={_lit(ctx.name)})({ctx.first})"
    )


def _emit_rope(ctx: EmitContext) -> str:
    return f"RotaryEmbedding(base={_lit(float(ctx.params['base']))}, name={_lit(ctx.name)})({ctx.first})"


def _emit_rms_norm(ctx: EmitContext) -> str:
    return f"RMSNorm(epsilon={_lit(float(ctx.params['epsilon']))}, name={_lit(ctx.name)})({ctx.first})"


def _emit_swiglu(ctx: EmitContext) -> str:
    return (
        f"SwiGLU({int(ctx.params['hidden_dim'])}, "
        f"dropout={_lit(float(ctx.params['dropout']))}, name={_lit(ctx.name)})({ctx.first})"
    )


def _emit_attention(ctx: EmitContext) -> str:
    """Self-attention: the same tensor is query, key, and value."""

    return (
        f"tf.keras.layers.MultiHeadAttention("
        f"num_heads={_lit(int(ctx.params['num_heads']))}, "
        f"key_dim={_lit(int(ctx.params['key_dim']))}, "
        f"dropout={_lit(float(ctx.params['dropout']))}, "
        f"name={_lit(ctx.name)})"
        f"({ctx.first}, {ctx.first}, use_causal_mask={_lit(bool(ctx.params['causal']))})"
    )


def _emit_feed_forward(ctx: EmitContext) -> str:
    """Two Dense layers around an activation, projecting back to the input width.

    Emitted as a Sequential so the whole block is one named node on the canvas
    rather than three, and so the output width can be read from the input.
    """

    width = ctx.input_shapes[0][-1] if ctx.input_shapes and ctx.input_shapes[0] else None
    if width is None:
        raise EmitError(
            f"Node {ctx.node.id!r}: a feed-forward block needs a known input width."
        )
    return (
        f"tf.keras.Sequential(["
        f"tf.keras.layers.Dense({int(ctx.params['hidden_dim'])}, "
        f"activation={_lit(ctx.params['activation'])}), "
        f"tf.keras.layers.Dropout({_lit(float(ctx.params['dropout']))}), "
        f"tf.keras.layers.Dense({int(width)})"
        f"], name={_lit(ctx.name)})({ctx.first})"
    )


def _emit_transformer_block(ctx: EmitContext) -> str:
    return (
        f"transformer_block({ctx.first}, "
        f"layers={_lit(int(ctx.params['layers']))}, "
        f"num_heads={_lit(int(ctx.params['num_heads']))}, "
        f"key_dim={_lit(int(ctx.params['key_dim']))}, "
        f"ffn_dim={_lit(int(ctx.params['ffn_dim']))}, "
        f"num_kv_heads={_lit(int(ctx.params['num_kv_heads']))}, "
        f"norm={_lit(ctx.params['norm'])}, "
        f"ffn={_lit(ctx.params['ffn'])}, "
        f"num_experts={_lit(int(ctx.params['num_experts']))}, "
        f"experts_per_token={_lit(int(ctx.params['experts_per_token']))}, "
        f"shared_experts={_lit(int(ctx.params['shared_experts']))}, "
        f"qk_norm={_lit(bool(ctx.params['qk_norm']))}, "
        f"sliding_window={_lit(int(ctx.params['sliding_window']))}, "
        f"global_every={_lit(int(ctx.params['global_every']))}, "
        f"causal={_lit(bool(ctx.params['causal']))}, "
        f"dropout={_lit(float(ctx.params['dropout']))}, "
        f"name={_lit(ctx.name)})"
    )


def _emit_gqa(ctx: EmitContext) -> str:
    return (
        f"GroupedQueryAttention("
        f"{int(ctx.params['num_heads'])}, "
        f"{int(ctx.params['num_kv_heads'])}, "
        f"{int(ctx.params['key_dim'])}, "
        f"causal={_lit(bool(ctx.params['causal']))}, "
        f"dropout={_lit(float(ctx.params['dropout']))}, "
        f"qk_norm={_lit(bool(ctx.params['qk_norm']))}, "
        f"sliding_window={_lit(int(ctx.params['sliding_window']))}, "
        f"name={_lit(ctx.name)})({ctx.first})"
    )


def _emit_moe(ctx: EmitContext) -> str:
    return (
        f"MixtureOfExperts("
        f"{int(ctx.params['num_experts'])}, "
        f"{int(ctx.params['experts_per_token'])}, "
        f"{int(ctx.params['hidden_dim'])}, "
        f"dropout={_lit(float(ctx.params['dropout']))}, "
        f"shared_experts={_lit(int(ctx.params['shared_experts']))}, "
        f"name={_lit(ctx.name)})({ctx.first})"
    )


def _emit_lm_head(ctx: EmitContext) -> str:
    vocab = (
        "num_classes"
        if ctx.params.get("vocab_from_dataset")
        else _lit(int(ctx.params["vocab_size"]))
    )
    softcap = float(ctx.params.get("logit_softcap", 0.0) or 0.0)
    if ctx.params.get("tie_embeddings"):
        # Weight tying: project with the transpose of the embedding table
        # rather than a second matrix. On a 262k-token vocabulary that is
        # hundreds of millions of parameters not spent twice.
        layer = f"{_embedding_variable(ctx)}_layer"
        return (
            f"TiedLMHead({vocab}, {layer}, softcap={_lit(softcap)}, "
            f"name={_lit(ctx.name)})({ctx.first})"
        )
    if softcap:
        return f"SoftcappedDense({vocab}, {_lit(softcap)}, name={_lit(ctx.name)})({ctx.first})"
    # Bias-free, as every published causal LM head is: the softmax that follows
    # is shift-invariant, so a per-token bias buys nothing and costs one
    # parameter per vocabulary entry.
    return f"tf.keras.layers.Dense({vocab}, use_bias=False, name={_lit(ctx.name)})({ctx.first})"


def _embedding_variable(ctx: EmitContext) -> str:
    """The embedding layer whose table a tied head reuses.

    Tying only means anything if there is exactly one embedding upstream; with
    none, or several, the user has to say which, so the graph is rejected
    rather than guessed at.
    """

    candidates = ctx.embedding_vars
    if len(candidates) != 1:
        raise EmitError(
            f"Node {ctx.node.id!r}: tied embeddings need exactly one Embedding node in the "
            f"graph, but this one has {len(candidates)}."
        )
    return candidates[0]


def _emit_custom_layer(ctx: EmitContext) -> str:
    """Instantiate the user's hoisted class.

    The class body itself is emitted above `build_model` by
    `_custom_definitions`; this is only the call site.
    """

    class_name = _class_name(ctx.params.get("class_name"), ctx.params.get("source"))
    if not class_name:
        raise EmitError(
            f"Node {ctx.node.id!r}: give the custom layer a class name matching its source."
        )
    return f"{class_name}(name={_lit(ctx.name)})({ctx.first})"


def _emit_custom_function(ctx: EmitContext) -> str:
    expression = str(ctx.params.get("expression") or "").strip()
    if not expression:
        raise EmitError(f"Node {ctx.node.id!r}: the custom function has no expression.")
    if "\n" in expression:
        raise EmitError(
            f"Node {ctx.node.id!r}: a custom function must be a single expression. "
            "Use a Custom layer for multi-line logic."
        )
    return f"tf.keras.layers.Lambda(lambda x: {expression}, name={_lit(ctx.name)})({ctx.first})"


def _class_name(declared: Any, source: Any) -> str:
    """The class to instantiate: what the user declared, else the first one defined."""

    name = str(declared or "").strip()
    if name:
        return name.splitlines()[0].strip()
    match = re.search(r"^\s*class\s+([A-Za-z_]\w*)", str(source or ""), re.MULTILINE)
    return match.group(1) if match else ""


def _custom_definitions(resolved: ResolvedGraph) -> list[str]:
    """Hoist each custom layer's source above `build_model`, once per class.

    Two nodes may share a class; emitting it twice would be a redefinition, so
    the first definition wins and later nodes reuse it.
    """

    seen: set[str] = set()
    blocks: list[str] = []
    for node_id in resolved.order:
        node = resolved.nodes[node_id]
        if node.type != "custom_layer":
            continue
        source = str(node.params.get("source") or "").strip()
        if not source:
            raise EmitError(f"Node {node.id!r}: the custom layer has no source.")
        class_name = _class_name(node.params.get("class_name"), source)
        if class_name in seen:
            continue
        seen.add(class_name)
        blocks.extend([source, "", ""])
    return blocks


# --- new primitives ---------------------------------------------------------


def _emit_depthwise_conv(ctx: EmitContext) -> str:
    strides = int(ctx.params["strides"])
    return _layer(
        "DepthwiseConv2D",
        int(ctx.params["kernel_size"]),
        strides=strides if strides != 1 else None,
        padding=ctx.params["padding"],
        depth_multiplier=int(ctx.params["depth_multiplier"]) or None,
        activation=_activation_or_none(ctx.params["activation"]),
        name=ctx.name,
        applied_to=ctx.first,
    )


def _emit_squeeze_excite(ctx: EmitContext) -> str:
    return (
        f"SqueezeExcite({int(ctx.params['ratio'])}, gate={_lit(ctx.params['gate'])}, "
        f"name={_lit(ctx.name)})({ctx.first})"
    )


def _emit_patch_embedding(ctx: EmitContext) -> str:
    return (
        f"PatchEmbedding({int(ctx.params['patch_size'])}, {int(ctx.params['embed_dim'])}, "
        f"class_token={_lit(bool(ctx.params['class_token']))}, name={_lit(ctx.name)})({ctx.first})"
    )


def _emit_geglu(ctx: EmitContext) -> str:
    return (
        f"GatedFeedForward({int(ctx.params['hidden_dim'])}, \"gelu\", "
        f"{_lit(float(ctx.params['dropout']))}, name={_lit(ctx.name)})({ctx.first})"
    )


# --- NLP emitters -----------------------------------------------------------


def _emit_sinusoidal(ctx: EmitContext) -> str:
    return (
        f"SinusoidalPositionEncoding(base={_lit(float(ctx.params['base']))}, "
        f"name={_lit(ctx.name)})({ctx.first})"
    )


def _emit_cross_attention(ctx: EmitContext) -> str:
    """Query first, context second — the wire order the node's help states."""

    if len(ctx.inputs) < 2:
        raise EmitError(
            f"Node {ctx.node.id!r}: cross-attention needs a query and a context, in that order."
        )
    return (
        f"tf.keras.layers.MultiHeadAttention("
        f"num_heads={_lit(int(ctx.params['num_heads']))}, "
        f"key_dim={_lit(int(ctx.params['key_dim']))}, "
        f"dropout={_lit(float(ctx.params['dropout']))}, "
        f"name={_lit(ctx.name)})({ctx.inputs[0]}, {ctx.inputs[1]})"
    )


def _emit_sequence_pool(ctx: EmitContext) -> str:
    return (
        f"SequencePooling({_lit(str(ctx.params['mode']))}, "
        f"hidden_dim={int(ctx.params['hidden_dim'])}, name={_lit(ctx.name)})({ctx.first})"
    )


def _emit_span_head(ctx: EmitContext) -> str:
    # Two logits per token, start and end. The softmax is the loss's job, not
    # the layer's, so the head stays linear.
    return f"tf.keras.layers.Dense(2, name={_lit(ctx.name)})({ctx.first})"


def _kernel_sizes(ctx: EmitContext) -> list[int]:
    sizes = parse_int_list(ctx.params.get("kernel_sizes"))
    if not sizes:
        raise EmitError(
            f"Node {ctx.node.id!r}: kernel widths must be a comma-separated list of "
            "positive integers, such as “3,4,5”."
        )
    return sizes


def _emit_text_cnn(ctx: EmitContext) -> str:
    return (
        f"text_cnn({ctx.first}, "
        f"filters={_lit(int(ctx.params['filters']))}, "
        f"kernel_sizes={tuple(_kernel_sizes(ctx))!r}, "
        f"activation={_lit(ctx.params['activation'])}, "
        f"dropout={_lit(float(ctx.params['dropout']))}, "
        f"name={_lit(ctx.name)})"
    )


def _emit_bilstm_encoder(ctx: EmitContext) -> str:
    return (
        f"bi_rnn_encoder({ctx.first}, "
        f"cell={_lit(str(ctx.params['cell']))}, "
        f"units={_lit(int(ctx.params['units']))}, "
        f"layers={_lit(int(ctx.params['layers']))}, "
        f"dropout={_lit(float(ctx.params['dropout']))}, "
        f"return_sequences={_lit(bool(ctx.params['return_sequences']))}, "
        f"name={_lit(ctx.name)})"
    )


def _emit_mla(ctx: EmitContext) -> str:
    constructor = _call(
        "LatentAttention",
        [
            str(int(ctx.params["num_heads"])),
            str(int(ctx.params["kv_lora_rank"])),
            str(int(ctx.params["qk_nope_head_dim"])),
            str(int(ctx.params["qk_rope_head_dim"])),
            str(int(ctx.params["v_head_dim"])),
        ],
        {
            "q_lora_rank": int(ctx.params["q_lora_rank"]),
            "rope_theta": float(ctx.params["rope_theta"]),
            "causal": bool(ctx.params["causal"]),
            "dropout": float(ctx.params["dropout"]),
            "name": ctx.name,
        },
    )
    return f"{constructor}({ctx.first})"


# --- vision blocks ----------------------------------------------------------


def _emit_resnet_block(ctx: EmitContext) -> str:
    return _call(
        "resnet_stage",
        [ctx.first],
        {
            "filters": int(ctx.params["filters"]),
            "blocks": int(ctx.params["blocks"]),
            "stride": int(ctx.params["stride"]),
            "variant": str(ctx.params["variant"]),
            "expansion": int(ctx.params["expansion"]),
            "name": ctx.name,
        },
    )


def _emit_inverted_residual(ctx: EmitContext) -> str:
    return _call(
        "inverted_residual",
        [ctx.first],
        {
            "filters": int(ctx.params["filters"]),
            "expand_ratio": int(ctx.params["expand_ratio"]),
            "kernel_size": int(ctx.params["kernel_size"]),
            "stride": int(ctx.params["stride"]),
            "use_se": bool(ctx.params["use_se"]),
            "se_ratio": int(ctx.params["se_ratio"]),
            "activation": str(ctx.params["activation"]),
            "blocks": int(ctx.params["blocks"]),
            "name": ctx.name,
        },
    )


def _emit_dense_block(ctx: EmitContext) -> str:
    return _call(
        "dense_block",
        [ctx.first],
        {
            "growth_rate": int(ctx.params["growth_rate"]),
            "layers": int(ctx.params["layers"]),
            "bottleneck_ratio": int(ctx.params["bottleneck_ratio"]),
            "name": ctx.name,
        },
    )


def _emit_inception_block(ctx: EmitContext) -> str:
    return _call(
        "inception_module",
        [ctx.first],
        {
            "filters_1x1": int(ctx.params["filters_1x1"]),
            "reduce_3x3": int(ctx.params["reduce_3x3"]),
            "filters_3x3": int(ctx.params["filters_3x3"]),
            "reduce_5x5": int(ctx.params["reduce_5x5"]),
            "filters_5x5": int(ctx.params["filters_5x5"]),
            "filters_pool": int(ctx.params["filters_pool"]),
            "name": ctx.name,
        },
    )


def _emit_convnext_block(ctx: EmitContext) -> str:
    return _call(
        "convnext_stage",
        [ctx.first],
        {
            "filters": int(ctx.params["filters"]),
            "blocks": int(ctx.params["blocks"]),
            "kernel_size": int(ctx.params["kernel_size"]),
            "expand_ratio": int(ctx.params["expand_ratio"]),
            "layer_scale": float(ctx.params["layer_scale"]),
            "name": ctx.name,
        },
    )


def _emit_vit_block(ctx: EmitContext) -> str:
    return _call(
        "vit_encoder",
        [ctx.first],
        {
            "layers": int(ctx.params["layers"]),
            "num_heads": int(ctx.params["num_heads"]),
            "head_dim": int(ctx.params["head_dim"]),
            "mlp_dim": int(ctx.params["mlp_dim"]),
            "dropout": float(ctx.params["dropout"]),
            "name": ctx.name,
        },
    )


# --- named LLM family blocks ------------------------------------------------

# Every parameter is written out in full, even where it equals `llm_block`'s own
# default: the generated file is the record of what the architecture is, and a
# reader should not have to open the helper to learn whether a block uses
# QK-norm. The int/float/bool/text split lives in `blocks.py` so the torch
# emitter renders each parameter the same way.


def _emit_llm_block(ctx: EmitContext) -> str:
    def value(key: str) -> Any:
        return ctx.params.get(key, LLM_BLOCK_DEFAULTS[key])

    kwargs: dict[str, Any] = {}
    for key in LLM_BLOCK_INT_PARAMS:
        kwargs[key] = int(value(key) or 0)
    for key in LLM_BLOCK_FLOAT_PARAMS:
        kwargs[key] = float(value(key) or 0.0)
    for key in LLM_BLOCK_BOOL_PARAMS:
        kwargs[key] = bool(value(key))
    for key in LLM_BLOCK_TEXT_PARAMS:
        kwargs[key] = str(value(key))
    kwargs["name"] = ctx.name
    return _call("llm_block", [ctx.first], kwargs)


def _call(function: str, positional: list[str], kwargs: dict[str, Any]) -> str:
    """`function(a, b, key=value, ...)` with every value rendered as a literal.

    Positional arguments arrive pre-rendered (they are usually variable names);
    keyword values are literals. A `None` keyword is dropped so the helper's own
    default applies.
    """

    rendered = list(positional)
    rendered += [f"{key}={_lit(item)}" for key, item in kwargs.items() if item is not None]
    return f"{function}({', '.join(rendered)})"


def _activation_or_none(value: Any) -> Any:
    return None if str(value) == "linear" else value


EMITTERS: dict[str, Callable[[EmitContext], str]] = {
    "depthwise_conv2d": _emit_depthwise_conv,
    "squeeze_excite": _emit_squeeze_excite,
    "patch_embedding": _emit_patch_embedding,
    "geglu": _emit_geglu,
    "mla_attention": _emit_mla,
    "resnet_block": _emit_resnet_block,
    "inverted_residual_block": _emit_inverted_residual,
    "dense_block": _emit_dense_block,
    "inception_block": _emit_inception_block,
    "convnext_block": _emit_convnext_block,
    "vit_block": _emit_vit_block,
    **{family.type: _emit_llm_block for family in LLM_FAMILIES},
    "input": _emit_input,
    # `emit_module` short-circuits Output nodes before reaching this table —
    # they alias their input rather than emitting a layer. The entry exists so
    # the catalog/shapes/emitters key sets stay identical, which is the check
    # that stops a node from being addable but unbuildable.
    "output": lambda ctx: ctx.first,
    "dense": _emit_dense,
    "activation": _emit_activation,
    "flatten": _emit_flatten,
    "reshape": _emit_reshape,
    "embedding": _emit_embedding,
    "conv1d": _emit_conv("Conv1D"),
    "conv2d": _emit_conv("Conv2D"),
    "separable_conv2d": _emit_conv("SeparableConv2D"),
    "conv2d_transpose": _emit_conv("Conv2DTranspose"),
    "max_pool2d": _emit_pool("MaxPooling2D"),
    "avg_pool2d": _emit_pool("AveragePooling2D"),
    "global_avg_pool2d": lambda ctx: _layer(
        "GlobalAveragePooling2D", name=ctx.name, applied_to=ctx.first
    ),
    "global_max_pool2d": lambda ctx: _layer(
        "GlobalMaxPooling2D", name=ctx.name, applied_to=ctx.first
    ),
    "up_sampling2d": lambda ctx: _layer(
        "UpSampling2D",
        (int(ctx.params["size"]), int(ctx.params["size"])),
        name=ctx.name,
        applied_to=ctx.first,
    ),
    "zero_padding2d": lambda ctx: _layer(
        "ZeroPadding2D",
        (int(ctx.params["padding"]), int(ctx.params["padding"])),
        name=ctx.name,
        applied_to=ctx.first,
    ),
    "batch_norm": _identity_layer("BatchNormalization", momentum=None, epsilon=None),
    "layer_norm": _identity_layer("LayerNormalization", epsilon=None),
    "group_norm": _identity_layer("GroupNormalization", groups=None, epsilon=None),
    "lstm": _emit_recurrent("LSTM"),
    "gru": _emit_recurrent("GRU"),
    "add": _emit_merge("Add"),
    "multiply": _emit_merge("Multiply"),
    "average": _emit_merge("Average"),
    "subtract": _emit_merge("Subtract"),
    "concatenate": _emit_merge("Concatenate"),
    "random_flip": _identity_layer("RandomFlip", mode=None),
    "random_rotation": _identity_layer("RandomRotation", factor=None),
    "random_zoom": _emit_random_zoom,
    "random_contrast": _emit_random_contrast,
    "dropout": _identity_layer("Dropout", rate=None),
    "spatial_dropout2d": _identity_layer("SpatialDropout2D", rate=None),
    "gaussian_noise": _identity_layer("GaussianNoise", stddev=None),
    "activity_regularization": _identity_layer("ActivityRegularization", l1=None, l2=None),
    "pretrained_backbone": _emit_backbone,
    "positional_embedding": _emit_positional,
    "rotary_embedding": _emit_rope,
    "multi_head_attention": _emit_attention,
    "rms_norm": _emit_rms_norm,
    "swiglu": _emit_swiglu,
    "feed_forward": _emit_feed_forward,
    "transformer_block": _emit_transformer_block,
    "lm_head": _emit_lm_head,
    "global_avg_pool1d": lambda ctx: _layer(
        "GlobalAveragePooling1D", name=ctx.name, applied_to=ctx.first
    ),
    "grouped_query_attention": _emit_gqa,
    "moe_feed_forward": _emit_moe,
    "sinusoidal_position_encoding": _emit_sinusoidal,
    "cross_attention": _emit_cross_attention,
    "sequence_pool": _emit_sequence_pool,
    "span_head": _emit_span_head,
    "text_cnn_block": _emit_text_cnn,
    "bilstm_encoder": _emit_bilstm_encoder,
    "custom_layer": _emit_custom_layer,
    "custom_function": _emit_custom_function,
}
