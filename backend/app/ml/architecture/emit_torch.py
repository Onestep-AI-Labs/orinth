"""PyTorch code generation: graph IR → an `nn.Module`.

The second emitter behind the same IR. Keras remains the compile target that
actually trains inside the platform (`emit_keras`); this one exists so a graph
can leave the studio as PyTorch for people whose stack is torch.

Two things differ structurally from the Keras path and are worth knowing:

- **PyTorch is NCHW.** The generated module takes images as
  `(batch, channels, height, width)`, not the `(batch, height, width, channels)`
  the canvas shows. Sequences are `(batch, length)` in both.
- **Layers must be constructed with their input width.** Keras infers it at
  call time; torch does not. Widths come from the analytic shape pass, which is
  why a graph must resolve its shapes before it can be exported to torch.

Nodes whose meaning is tied to Keras — a `tf.keras.applications` backbone, or a
custom layer written as a `tf.keras.layers.Layer` — cannot be translated and
raise a clear error rather than emitting something that looks right and is not.
"""

from __future__ import annotations

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
from app.ml.architecture.emit_keras import EmitError, _identifier, _lit
from app.ml.architecture.graph import OUTPUT_TYPE, ResolvedGraph, ResolvedNode, parse_shape
from app.ml.architecture.shapes import Shape
from app.ml.architecture.torch_helpers import (
    ATTENTION_MASK_HELPER,
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
    SPARSE_MOE_HELPER,
    SQUEEZE_EXCITE_HELPER,
    VIT_HELPER,
    llm_block_source,
)
from app.ml.architecture.torch_helpers import (
    ROPE_APPLY_HELPER as ROPE_APPLY_HELPER_TORCH,
)

TORCH_MODULE_FILENAME = "generated_model_torch.py"


@dataclass(frozen=True)
class TorchContext:
    node: ResolvedNode
    inputs: list[str]
    input_shapes: list[Shape]
    name: str
    # Variable names of every Embedding module, for a tied LM head.
    embedding_vars: tuple[str, ...] = ()

    @property
    def params(self) -> dict[str, Any]:
        return self.node.params

    @property
    def first(self) -> str:
        return self.inputs[0]

    def width(self, index: int = 0) -> int:
        """Feature width of an input, which torch layers need at construction."""

        shape = self.input_shapes[index] if index < len(self.input_shapes) else None
        if shape is None or not shape or shape[-1] is None:
            raise EmitError(
                f"Node {self.node.id!r}: PyTorch export needs a known input width here. "
                "Give the Input node a fully specified shape."
            )
        return int(shape[-1])


def emit_torch_module(
    resolved: ResolvedGraph,
    shapes: dict[str, Shape] | None = None,
    *,
    architecture_name: str = "model",
    architecture_id: str = "",
    version: int = 1,
    default_num_classes: int = 2,
) -> str:
    """Render a standalone `nn.Module` for a resolved graph."""

    if not resolved.ok:
        raise EmitError(
            "Cannot generate code for a graph with errors: "
            + "; ".join(issue.message for issue in resolved.errors())
        )
    if not resolved.order:
        raise EmitError("Cannot generate code for an empty graph.")

    shapes = shapes or {}
    class_name = _class_name(architecture_name)
    aliases: dict[str, str] = {}
    constructors: list[str] = []
    forward: list[str] = []
    helpers: set[str] = set()
    embedding_vars = tuple(
        resolved.nodes[node_id].var_name
        for node_id in resolved.order
        if resolved.nodes[node_id].type == "embedding"
    )

    for node_id in resolved.order:
        node = resolved.nodes[node_id]
        input_vars = [aliases.get(source, resolved.nodes[source].var_name) for source in node.inputs]
        if node.type == OUTPUT_TYPE:
            if not input_vars:
                raise EmitError("The Output node has nothing connected to it.")
            aliases[node_id] = input_vars[0]
            continue
        emitter = TORCH_EMITTERS.get(node.type)
        if emitter is None:
            raise EmitError(
                f"{node.spec.name} has no PyTorch equivalent, so this graph cannot be "
                "exported to torch. Switch the code panel to TensorFlow, or replace that node."
            )
        context = TorchContext(
            node=node,
            inputs=input_vars,
            input_shapes=[shapes.get(source) for source in node.inputs],
            name=node.var_name,
            embedding_vars=embedding_vars,
        )
        helpers.update(TORCH_HELPER_DEPENDENCIES.get(node.type, ()))
        # As on the Keras side: a family block's helpers follow its parameters,
        # so the two frameworks emit the same set of structures.
        if is_llm_block(node.type):
            # The set carries structural markers as well as helper keys —
            # `rms_norm`, `layer_norm`, `classic_ffn` — which `llm_block_source`
            # reads and `TORCH_HELPER_ORDER` simply does not contain, so they
            # steer the generated block without emitting anything of their own.
            # (torch's `nn.RMSNorm` is built in; Keras needs the helper class.)
            helpers.update(llm_block_structures(node.params))
        if node.type == "lm_head":
            if node.params.get("tie_embeddings"):
                helpers.add("tied_head")
            elif float(node.params.get("logit_softcap", 0.0) or 0.0):
                helpers.add("softcap")
        construction, expression = emitter(context)
        if construction:
            constructors.append(f"        self.{node.var_name} = {construction}")
        forward.append(f"        {node.var_name} = {expression}")

    input_ids = resolved.input_ids
    output_vars = [aliases[node_id] for node_id in resolved.output_ids if node_id in aliases]
    if not input_ids or not output_vars:
        raise EmitError("The graph needs a connected Input node and Output node.")

    arguments = ", ".join(resolved.nodes[node_id].var_name for node_id in input_ids)
    # `llm_block` is assembled rather than looked up: its branches are pruned to
    # the structures this graph reaches, so it never constructs a helper class
    # the module has not defined.
    helper_source = [
        llm_block_source(helpers) if key == "llm_block" else TORCH_HELPER_SOURCE[key]
        for key in TORCH_HELPER_ORDER
        if key in helpers
    ]

    return "\n".join(
        [
            *_header(architecture_name, architecture_id, version),
            "",
            "import torch",
            "from torch import nn",
            "import torch.nn.functional as F",
            "",
            "",
            *_joined(helper_source),
            f"class {class_name}(nn.Module):",
            f'    """{architecture_name} — generated from a visual graph."""',
            "",
            f"    def __init__(self, num_classes: int = {default_num_classes}) -> None:",
            "        super().__init__()",
            *(constructors or ["        pass"]),
            "",
            f"    def forward(self, {arguments}):",
            *forward,
            f"        return {output_vars[0]}",
            "",
            "",
            f"def build_model(num_classes: int = {default_num_classes}) -> nn.Module:",
            f"    return {class_name}(num_classes=num_classes)",
            "",
            "",
            'if __name__ == "__main__":',
            "    model = build_model()",
            "    print(model)",
            '    print("parameters:", sum(p.numel() for p in model.parameters()))',
            "",
        ]
    )


def _joined(blocks: list[str]) -> list[str]:
    lines: list[str] = []
    for block in blocks:
        lines.extend([block, "", ""])
    return lines


def _header(architecture_name: str, architecture_id: str, version: int) -> list[str]:
    origin = f"{architecture_name!r}"
    if architecture_id:
        origin += f" ({architecture_id}, v{version})"
    return [
        f"# Generated by Onestep AI Platform from architecture {origin}.",
        "# PyTorch export. Images arrive as (batch, channels, height, width) —",
        "# the canvas shows the Keras (height, width, channels) convention.",
        "# Regenerated from the graph on every export; edits here are not read back.",
    ]


# Class names the generated helpers occupy. An architecture called "ResNet
# stage" or "Dense block" would otherwise produce a module class that shadows
# the helper it is trying to call, and the failure is a confusing TypeError
# about an unexpected keyword argument rather than a name clash.
RESERVED_CLASS_NAMES = frozenset(
    {
        "RotaryEmbedding",
        "PositionalEmbedding",
        "SwiGLU",
        "GatedFeedForward",
        "SqueezeExcite",
        "PatchEmbedding",
        "LayerScale",
        "GroupedQueryAttention",
        "FamilyAttention",
        "LatentAttention",
        "MixtureOfExperts",
        "SparseMoE",
        "TiedLMHead",
        "SoftcappedLinear",
        "TransformerStack",
        "LlmBlockStack",
        "ResNetStage",
        "InvertedResidual",
        "DenseBlock",
        "InceptionModule",
        "ConvNeXtStage",
        "ViTEncoder",
    }
)


def _class_name(text: str) -> str:
    slug = _identifier(text) or "model"
    name = "".join(part.capitalize() for part in slug.split("_")) or "Model"
    return f"{name}Model" if name in RESERVED_CLASS_NAMES else name


# --- emitters ---------------------------------------------------------------
#
# Each returns `(constructor expression or "", forward expression)`. A layer
# with weights is constructed in __init__ and called in forward; a stateless op
# emits no constructor and inlines into forward.

Emitter = Callable[[TorchContext], tuple[str, str]]

ACTIVATIONS = {
    "relu": "F.relu({x})",
    "gelu": "F.gelu({x})",
    "swish": "F.silu({x})",
    "tanh": "torch.tanh({x})",
    "sigmoid": "torch.sigmoid({x})",
    "softmax": "F.softmax({x}, dim=-1)",
    "elu": "F.elu({x})",
    "selu": "F.selu({x})",
    "linear": "{x}",
}


def _activation_expression(name: str, value: str) -> str:
    template = ACTIVATIONS.get(str(name), "{x}")
    return template.format(x=value)


def _input(ctx: TorchContext) -> tuple[str, str]:
    # The forward signature already binds the input; this is an identity so the
    # variable exists in the same order as the Keras path.
    return "", ctx.name


def _dense(ctx: TorchContext) -> tuple[str, str]:
    units = "num_classes" if ctx.params.get("units_from_dataset") else str(int(ctx.params["units"]))
    bias = _lit(bool(ctx.params["use_bias"]))
    construction = f"nn.Linear({ctx.width()}, {units}, bias={bias})"
    return construction, _activation_expression(
        ctx.params["activation"], f"self.{ctx.name}({ctx.first})"
    )


def _conv(dimension: int) -> Emitter:
    def emit(ctx: TorchContext) -> tuple[str, str]:
        kernel = int(ctx.params["kernel_size"])
        stride = int(ctx.params["strides"])
        # torch accepts padding="same" only at stride 1; otherwise emit the
        # integer padding that reproduces Keras's behaviour most closely.
        if str(ctx.params["padding"]) == "same":
            padding = '"same"' if stride == 1 else str(kernel // 2)
        else:
            padding = "0"
        construction = (
            f"nn.Conv{dimension}d({ctx.width()}, {int(ctx.params['filters'])}, {kernel}, "
            f"stride={stride}, padding={padding})"
        )
        return construction, _activation_expression(
            ctx.params["activation"], f"self.{ctx.name}({ctx.first})"
        )

    return emit


def _conv_transpose(ctx: TorchContext) -> tuple[str, str]:
    kernel = int(ctx.params["kernel_size"])
    stride = int(ctx.params["strides"])
    padding = kernel // 2 if str(ctx.params["padding"]) == "same" else 0
    construction = (
        f"nn.ConvTranspose2d({ctx.width()}, {int(ctx.params['filters'])}, {kernel}, "
        f"stride={stride}, padding={padding}, output_padding={stride - 1})"
    )
    return construction, _activation_expression(
        ctx.params["activation"], f"self.{ctx.name}({ctx.first})"
    )


def _pool(kind: str) -> Emitter:
    def emit(ctx: TorchContext) -> tuple[str, str]:
        size = int(ctx.params["pool_size"])
        stride = int(ctx.params["strides"]) or size
        padding = size // 2 if str(ctx.params["padding"]) == "same" else 0
        return "", f"F.{kind}({ctx.first}, {size}, stride={stride}, padding={padding})"

    return emit


def _global_pool2d(kind: str) -> Emitter:
    def emit(ctx: TorchContext) -> tuple[str, str]:
        operation = "mean" if kind == "avg" else "amax"
        return "", f"{ctx.first}.{operation}(dim=(2, 3))"

    return emit


def _norm(module: str, argument: str = "") -> Emitter:
    def emit(ctx: TorchContext) -> tuple[str, str]:
        extra = f", {argument}" if argument else ""
        return f"nn.{module}({ctx.width()}{extra})", f"self.{ctx.name}({ctx.first})"

    return emit


def _batch_norm(ctx: TorchContext) -> tuple[str, str]:
    """BatchNorm2d, with the node's momentum and epsilon actually applied.

    Both were hardcoded before — `eps=1e-3` and torch's default momentum — so a
    momentum set on the canvas took effect in the Keras module and was silently
    dropped from the PyTorch one. Two emitters that disagree are describing two
    different models.

    The conventions are also inverted, which makes a naive hand-off worse than
    no hand-off at all. Keras `momentum` is the fraction of the *old* running
    statistic to keep; torch `momentum` is the fraction of the *new* batch
    statistic to take. Keras 0.9 and torch 0.1 are the same model.
    """

    keras_momentum = float(ctx.params["momentum"])
    epsilon = float(ctx.params["epsilon"])
    return (
        f"nn.BatchNorm2d({ctx.width()}, eps={epsilon!r}, momentum={1.0 - keras_momentum:.6g})",
        f"self.{ctx.name}({ctx.first})",
    )


def _recurrent(module: str) -> Emitter:
    def emit(ctx: TorchContext) -> tuple[str, str]:
        units = int(ctx.params["units"])
        bidirectional = bool(ctx.params["bidirectional"])
        construction = (
            f"nn.{module}({ctx.width()}, {units}, batch_first=True, "
            f"bidirectional={_lit(bidirectional)}, dropout={_lit(float(ctx.params['dropout']))})"
        )
        # torch returns `(sequence_output, state)`; Keras returns one or the
        # other depending on return_sequences, so select to match.
        tail = "[0]" if ctx.params.get("return_sequences") else "[0][:, -1]"
        return construction, f"self.{ctx.name}({ctx.first}){tail}"

    return emit


def _merge(operation: str) -> Emitter:
    def emit(ctx: TorchContext) -> tuple[str, str]:
        if operation == "cat":
            return "", f"torch.cat([{', '.join(ctx.inputs)}], dim={int(ctx.params['axis'])})"
        if operation == "sub":
            return "", f"{ctx.inputs[0]} - {ctx.inputs[1]}"
        if operation == "mean":
            return "", f"torch.stack([{', '.join(ctx.inputs)}]).mean(dim=0)"
        joiner = " + " if operation == "add" else " * "
        return "", joiner.join(ctx.inputs)

    return emit


def _attention(ctx: TorchContext) -> tuple[str, str]:
    heads = int(ctx.params["num_heads"])
    construction = (
        f"nn.MultiheadAttention({ctx.width()}, {heads}, "
        f"dropout={_lit(float(ctx.params['dropout']))}, batch_first=True)"
    )
    causal = bool(ctx.params["causal"])
    call = (
        f"self.{ctx.name}({ctx.first}, {ctx.first}, {ctx.first}, "
        f"is_causal={_lit(causal)}, need_weights=False)[0]"
    )
    return construction, call


def _grouped_attention(ctx: TorchContext) -> tuple[str, str]:
    construction = (
        f"GroupedQueryAttention({ctx.width()}, {int(ctx.params['num_heads'])}, "
        f"{int(ctx.params['num_kv_heads'])}, {int(ctx.params['key_dim'])}, "
        f"causal={_lit(bool(ctx.params['causal']))}, "
        f"dropout={_lit(float(ctx.params['dropout']))}, "
        f"qk_norm={_lit(bool(ctx.params['qk_norm']))}, "
        f"sliding_window={int(ctx.params['sliding_window'])})"
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _swiglu(ctx: TorchContext) -> tuple[str, str]:
    construction = (
        f"SwiGLU({ctx.width()}, {int(ctx.params['hidden_dim'])}, "
        f"dropout={_lit(float(ctx.params['dropout']))})"
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _feed_forward(ctx: TorchContext) -> tuple[str, str]:
    width = ctx.width()
    hidden = int(ctx.params["hidden_dim"])
    construction = (
        f"nn.Sequential(nn.Linear({width}, {hidden}), nn.GELU(), "
        f"nn.Dropout({_lit(float(ctx.params['dropout']))}), nn.Linear({hidden}, {width}))"
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _moe(ctx: TorchContext) -> tuple[str, str]:
    construction = (
        f"MixtureOfExperts({ctx.width()}, {int(ctx.params['num_experts'])}, "
        f"{int(ctx.params['experts_per_token'])}, {int(ctx.params['hidden_dim'])}, "
        f"dropout={_lit(float(ctx.params['dropout']))}, "
        f"shared_experts={int(ctx.params['shared_experts'])})"
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _transformer_block(ctx: TorchContext) -> tuple[str, str]:
    construction = (
        f"TransformerStack({ctx.width()}, layers={int(ctx.params['layers'])}, "
        f"num_heads={int(ctx.params['num_heads'])}, "
        f"num_kv_heads={int(ctx.params['num_kv_heads'])}, "
        f"key_dim={int(ctx.params['key_dim'])}, ffn_dim={int(ctx.params['ffn_dim'])}, "
        f"norm={_lit(ctx.params['norm'])}, ffn={_lit(ctx.params['ffn'])}, "
        f"num_experts={int(ctx.params['num_experts'])}, "
        f"experts_per_token={int(ctx.params['experts_per_token'])}, "
        f"shared_experts={int(ctx.params['shared_experts'])}, "
        f"qk_norm={_lit(bool(ctx.params['qk_norm']))}, "
        f"sliding_window={int(ctx.params['sliding_window'])}, "
        f"global_every={int(ctx.params['global_every'])}, "
        f"causal={_lit(bool(ctx.params['causal']))}, "
        f"dropout={_lit(float(ctx.params['dropout']))})"
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _embedding(ctx: TorchContext) -> tuple[str, str]:
    construction = (
        f"nn.Embedding({int(ctx.params['input_dim'])}, {int(ctx.params['output_dim'])}"
        + (", padding_idx=0" if ctx.params.get("mask_zero") else "")
        + ")"
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _lm_head(ctx: TorchContext) -> tuple[str, str]:
    vocab = (
        "num_classes"
        if ctx.params.get("vocab_from_dataset")
        else str(int(ctx.params["vocab_size"]))
    )
    softcap = float(ctx.params.get("logit_softcap", 0.0) or 0.0)
    if ctx.params.get("tie_embeddings"):
        if len(ctx.embedding_vars) != 1:
            raise EmitError(
                f"Node {ctx.node.id!r}: tied embeddings need exactly one Embedding node in "
                f"the graph, but this one has {len(ctx.embedding_vars)}."
            )
        # torch ties by sharing the parameter object itself, which is how
        # `tie_word_embeddings` works in transformers.
        return (
            f"TiedLMHead({vocab}, softcap={_lit(softcap)})",
            f"self.{ctx.name}({ctx.first}, self.{ctx.embedding_vars[0]}.weight)",
        )
    if softcap:
        return (
            f"SoftcappedLinear({ctx.width()}, {vocab}, {_lit(softcap)})",
            f"self.{ctx.name}({ctx.first})",
        )
    # Bias-free, matching every published causal LM head — the softmax that
    # follows is shift-invariant, so the bias buys nothing.
    return f"nn.Linear({ctx.width()}, {vocab}, bias=False)", f"self.{ctx.name}({ctx.first})"


def _reshape(ctx: TorchContext) -> tuple[str, str]:
    target = parse_shape(ctx.params.get("target_shape"))
    if target is None:
        raise EmitError(f"Node {ctx.node.id!r} has an unparseable target shape.")
    dimensions = ", ".join(str(dimension) for dimension in target)
    return "", f"{ctx.first}.reshape({ctx.first}.shape[0], {dimensions})"


def _custom_function(ctx: TorchContext) -> tuple[str, str]:
    expression = str(ctx.params.get("expression") or "").strip()
    if not expression:
        raise EmitError(f"Node {ctx.node.id!r}: the custom function has no expression.")
    if "\n" in expression:
        raise EmitError(f"Node {ctx.node.id!r}: a custom function must be a single expression.")
    if "tf." in expression:
        raise EmitError(
            f"Node {ctx.node.id!r}: this custom function calls TensorFlow, so it cannot be "
            "exported to PyTorch. Rewrite it with torch, or export as TensorFlow."
        )
    return "", f"(lambda x: {expression})({ctx.first})"


# --- new primitives ---------------------------------------------------------


def _depthwise_conv(ctx: TorchContext) -> tuple[str, str]:
    channels = ctx.width()
    kernel = int(ctx.params["kernel_size"])
    stride = int(ctx.params["strides"])
    multiplier = int(ctx.params["depth_multiplier"])
    padding = kernel // 2 if str(ctx.params["padding"]) == "same" else 0
    construction = (
        f"nn.Conv2d({channels}, {channels * multiplier}, {kernel}, stride={stride}, "
        f"padding={padding}, groups={channels})"
    )
    return construction, _activation_expression(
        ctx.params["activation"], f"self.{ctx.name}({ctx.first})"
    )


def _squeeze_excite(ctx: TorchContext) -> tuple[str, str]:
    construction = (
        f"SqueezeExcite({ctx.width()}, {int(ctx.params['ratio'])}, "
        f"gate={_lit(str(ctx.params['gate']))})"
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _patch_embedding(ctx: TorchContext) -> tuple[str, str]:
    construction = (
        f"PatchEmbedding({ctx.width()}, {int(ctx.params['patch_size'])}, "
        f"{int(ctx.params['embed_dim'])}, class_token={_lit(bool(ctx.params['class_token']))})"
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _geglu(ctx: TorchContext) -> tuple[str, str]:
    construction = (
        f"GatedFeedForward({ctx.width()}, {int(ctx.params['hidden_dim'])}, \"gelu\", "
        f"{_lit(float(ctx.params['dropout']))})"
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _latent_attention(ctx: TorchContext) -> tuple[str, str]:
    construction = _construct(
        "LatentAttention",
        [
            ctx.width(),
            int(ctx.params["num_heads"]),
            int(ctx.params["kv_lora_rank"]),
            int(ctx.params["qk_nope_head_dim"]),
            int(ctx.params["qk_rope_head_dim"]),
            int(ctx.params["v_head_dim"]),
        ],
        {
            "q_lora_rank": int(ctx.params["q_lora_rank"]),
            "rope_theta": float(ctx.params["rope_theta"]),
            "causal": bool(ctx.params["causal"]),
            "dropout": float(ctx.params["dropout"]),
        },
    )
    return construction, f"self.{ctx.name}({ctx.first})"


# --- vision blocks ----------------------------------------------------------


def _resnet_block(ctx: TorchContext) -> tuple[str, str]:
    construction = _construct(
        "ResNetStage",
        [ctx.width(), int(ctx.params["filters"]), int(ctx.params["blocks"])],
        {
            "stride": int(ctx.params["stride"]),
            "variant": str(ctx.params["variant"]),
            "expansion": int(ctx.params["expansion"]),
        },
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _inverted_residual(ctx: TorchContext) -> tuple[str, str]:
    construction = _construct(
        "InvertedResidual",
        [ctx.width(), int(ctx.params["filters"])],
        {
            "expand_ratio": int(ctx.params["expand_ratio"]),
            "kernel_size": int(ctx.params["kernel_size"]),
            "stride": int(ctx.params["stride"]),
            "use_se": bool(ctx.params["use_se"]),
            "se_ratio": int(ctx.params["se_ratio"]),
            "activation": str(ctx.params["activation"]),
            "blocks": int(ctx.params["blocks"]),
        },
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _dense_block(ctx: TorchContext) -> tuple[str, str]:
    construction = _construct(
        "DenseBlock",
        [ctx.width(), int(ctx.params["growth_rate"]), int(ctx.params["layers"])],
        {"bottleneck_ratio": int(ctx.params["bottleneck_ratio"])},
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _inception_block(ctx: TorchContext) -> tuple[str, str]:
    construction = _construct(
        "InceptionModule",
        [
            ctx.width(),
            int(ctx.params["filters_1x1"]),
            int(ctx.params["reduce_3x3"]),
            int(ctx.params["filters_3x3"]),
            int(ctx.params["reduce_5x5"]),
            int(ctx.params["filters_5x5"]),
            int(ctx.params["filters_pool"]),
        ],
        {},
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _convnext_block(ctx: TorchContext) -> tuple[str, str]:
    construction = _construct(
        "ConvNeXtStage",
        [ctx.width(), int(ctx.params["filters"]), int(ctx.params["blocks"])],
        {
            "kernel_size": int(ctx.params["kernel_size"]),
            "expand_ratio": int(ctx.params["expand_ratio"]),
            "layer_scale": float(ctx.params["layer_scale"]),
        },
    )
    return construction, f"self.{ctx.name}({ctx.first})"


def _vit_block(ctx: TorchContext) -> tuple[str, str]:
    construction = _construct(
        "ViTEncoder",
        [
            ctx.width(),
            int(ctx.params["layers"]),
            int(ctx.params["num_heads"]),
            int(ctx.params["head_dim"]),
            int(ctx.params["mlp_dim"]),
        ],
        {"dropout": float(ctx.params["dropout"])},
    )
    return construction, f"self.{ctx.name}({ctx.first})"


# --- named LLM family blocks ------------------------------------------------


def _llm_block(ctx: TorchContext) -> tuple[str, str]:
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
    return _construct("LlmBlockStack", [ctx.width()], kwargs), f"self.{ctx.name}({ctx.first})"


def _construct(class_name: str, positional: list[Any], kwargs: dict[str, Any]) -> str:
    rendered = [str(item) for item in positional]
    rendered += [f"{key}={_lit(item)}" for key, item in kwargs.items()]
    return f"{class_name}({', '.join(rendered)})"


TORCH_EMITTERS: dict[str, Emitter] = {
    "depthwise_conv2d": _depthwise_conv,
    "squeeze_excite": _squeeze_excite,
    "patch_embedding": _patch_embedding,
    "geglu": _geglu,
    "mla_attention": _latent_attention,
    "resnet_block": _resnet_block,
    "inverted_residual_block": _inverted_residual,
    "dense_block": _dense_block,
    "inception_block": _inception_block,
    "convnext_block": _convnext_block,
    "vit_block": _vit_block,
    **{family.type: _llm_block for family in LLM_FAMILIES},
    "input": _input,
    "output": lambda ctx: ("", ctx.first),
    "dense": _dense,
    "activation": lambda ctx: ("", _activation_expression(ctx.params["activation"], ctx.first)),
    "flatten": lambda ctx: ("", f"{ctx.first}.flatten(start_dim=1)"),
    "reshape": _reshape,
    "embedding": _embedding,
    "conv1d": _conv(1),
    "conv2d": _conv(2),
    "separable_conv2d": _conv(2),
    "conv2d_transpose": _conv_transpose,
    "max_pool2d": _pool("max_pool2d"),
    "avg_pool2d": _pool("avg_pool2d"),
    "global_avg_pool2d": _global_pool2d("avg"),
    "global_max_pool2d": _global_pool2d("max"),
    "global_avg_pool1d": lambda ctx: ("", f"{ctx.first}.mean(dim=1)"),
    "up_sampling2d": lambda ctx: (
        "",
        f"F.interpolate({ctx.first}, scale_factor={int(ctx.params['size'])}, mode='nearest')",
    ),
    "zero_padding2d": lambda ctx: (
        "",
        f"F.pad({ctx.first}, ({int(ctx.params['padding'])},) * 4)",
    ),
    "batch_norm": _batch_norm,
    "layer_norm": _norm("LayerNorm"),
    "group_norm": lambda ctx: (
        f"nn.GroupNorm({int(ctx.params['groups'])}, {ctx.width()})",
        f"self.{ctx.name}({ctx.first})",
    ),
    "rms_norm": _norm("RMSNorm"),
    "lstm": _recurrent("LSTM"),
    "gru": _recurrent("GRU"),
    "add": _merge("add"),
    "multiply": _merge("mul"),
    "average": _merge("mean"),
    "subtract": _merge("sub"),
    "concatenate": _merge("cat"),
    "dropout": lambda ctx: (
        f"nn.Dropout({_lit(float(ctx.params['rate']))})",
        f"self.{ctx.name}({ctx.first})",
    ),
    "spatial_dropout2d": lambda ctx: (
        f"nn.Dropout2d({_lit(float(ctx.params['rate']))})",
        f"self.{ctx.name}({ctx.first})",
    ),
    "gaussian_noise": lambda ctx: (
        "",
        f"({ctx.first} + torch.randn_like({ctx.first}) * "
        f"{_lit(float(ctx.params['stddev']))} if self.training else {ctx.first})",
    ),
    "activity_regularization": lambda ctx: ("", ctx.first),
    "positional_embedding": lambda ctx: (
        f"PositionalEmbedding({int(ctx.params['max_length'])}, {ctx.width()})",
        f"self.{ctx.name}({ctx.first})",
    ),
    "rotary_embedding": lambda ctx: (
        f"RotaryEmbedding(base={_lit(float(ctx.params['base']))})",
        f"self.{ctx.name}({ctx.first})",
    ),
    "multi_head_attention": _attention,
    "grouped_query_attention": _grouped_attention,
    "swiglu": _swiglu,
    "feed_forward": _feed_forward,
    "moe_feed_forward": _moe,
    "transformer_block": _transformer_block,
    "lm_head": _lm_head,
    "custom_function": _custom_function,
}


# --- generated helper modules ----------------------------------------------

ROPE_HELPER = '''class RotaryEmbedding(nn.Module):
    """Rotary position embedding over the final dimension."""

    def __init__(self, base: float = 10000.0) -> None:
        super().__init__()
        self.base = base

    def forward(self, x):
        length, width = x.shape[1], x.shape[2]
        half = width // 2
        positions = torch.arange(length, device=x.device, dtype=x.dtype)[:, None]
        frequencies = self.base ** (
            -torch.arange(half, device=x.device, dtype=x.dtype) * 2.0 / width
        )[None, :]
        angles = positions * frequencies
        cos, sin = angles.cos(), angles.sin()
        first, second = x[..., :half], x[..., half:]
        return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)'''

POSITIONAL_HELPER = '''class PositionalEmbedding(nn.Module):
    """Learned absolute position vectors added to the sequence."""

    def __init__(self, max_length: int, width: int) -> None:
        super().__init__()
        self.positions = nn.Parameter(torch.randn(max_length, width) * 0.02)

    def forward(self, x):
        return x + self.positions[: x.shape[1]]'''

SWIGLU_HELPER = '''class SwiGLU(nn.Module):
    """Gated feed-forward: silu(gate(x)) * up(x), projected back down."""

    def __init__(self, width: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden_dim)
        self.up = nn.Linear(width, hidden_dim)
        self.down = nn.Linear(hidden_dim, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))'''

GQA_HELPER = '''class GroupedQueryAttention(nn.Module):
    """Attention where several query heads share one key/value head."""

    def __init__(self, width, num_heads, num_kv_heads, key_dim, causal=True, dropout=0.0,
                 qk_norm=False, sliding_window=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.key_dim = key_dim
        self.causal = causal
        self.sliding_window = sliding_window
        # QK-norm: RMSNorm over each head's vector before the dot product,
        # as Qwen3 and Gemma 3+ apply it.
        self.q_norm = nn.RMSNorm(key_dim) if qk_norm else None
        self.k_norm = nn.RMSNorm(key_dim) if qk_norm else None
        self.q = nn.Linear(width, num_heads * key_dim)
        self.k = nn.Linear(width, num_kv_heads * key_dim)
        self.v = nn.Linear(width, num_kv_heads * key_dim)
        self.o = nn.Linear(num_heads * key_dim, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, _ = x.shape
        groups = self.num_heads // self.num_kv_heads

        def split(t, heads):
            return t.view(batch, length, heads, self.key_dim).transpose(1, 2)

        query = split(self.q(x), self.num_heads)
        key = split(self.k(x), self.num_kv_heads)
        value = split(self.v(x), self.num_kv_heads)
        if self.q_norm is not None:
            query, key = self.q_norm(query), self.k_norm(key)
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

        dropout_p = self.drop.p if self.training else 0.0
        if self.sliding_window:
            # Local attention: build the band mask explicitly, since
            # `is_causal` alone cannot express a window.
            positions = torch.arange(length, device=x.device)
            distance = positions[:, None] - positions[None, :]
            allowed = (distance >= 0) & (distance < self.sliding_window)
            if not self.causal:
                allowed = distance.abs() < self.sliding_window
            context = F.scaled_dot_product_attention(
                query, key, value, attn_mask=allowed, dropout_p=dropout_p
            )
        else:
            context = F.scaled_dot_product_attention(
                query, key, value, is_causal=self.causal, dropout_p=dropout_p
            )
        context = context.transpose(1, 2).reshape(batch, length, self.num_heads * self.key_dim)
        return self.o(context)'''

MOE_HELPER = '''class MixtureOfExperts(nn.Module):
    """Sparse feed-forward: a router sends each token to its top-k experts.

    A readable dense-gather implementation — every expert runs on every token
    and unselected outputs are masked. Correct and slow, not a production
    kernel.
    """

    def __init__(self, width, num_experts, experts_per_token, hidden_dim, dropout=0.0,
                 shared_experts=0):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = min(experts_per_token, num_experts)
        self.router = nn.Linear(width, num_experts)
        # Gated experts, matching the Keras emitter and production MoE designs.
        self.experts = nn.ModuleList(
            SwiGLU(width, hidden_dim, dropout) for _ in range(num_experts)
        )
        # Always-on experts every token passes through, unrouted — the
        # DeepSeek design, where a shared expert absorbs what all tokens need.
        self.shared = nn.ModuleList(
            SwiGLU(width, hidden_dim, dropout) for _ in range(shared_experts)
        )

    def forward(self, x):
        logits = self.router(x)
        values, indices = logits.topk(self.experts_per_token, dim=-1)
        gates = values.softmax(dim=-1)
        weights = torch.zeros_like(logits).scatter(-1, indices, gates)
        stacked = torch.stack([expert(x) for expert in self.experts], dim=-2)
        out = (stacked * weights.unsqueeze(-1)).sum(dim=-2)
        for expert in self.shared:
            out = out + expert(x)
        return out'''

TRANSFORMER_STACK_HELPER = '''class TransformerStack(nn.Module):
    """A stack of pre-norm transformer layers, each with its own weights."""

    def __init__(
        self,
        width,
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
    ):
        super().__init__()
        self.causal = causal
        make_norm = (lambda: nn.RMSNorm(width)) if norm == "rms" else (lambda: nn.LayerNorm(width))
        self.attn_norms = nn.ModuleList(make_norm() for _ in range(layers))
        self.ffn_norms = nn.ModuleList(make_norm() for _ in range(layers))

        def make_attention(index):
            # Hybrid attention: most layers local, every `global_every`-th full.
            is_global = bool(global_every) and (index + 1) % global_every == 0
            window = 0 if (is_global or not sliding_window) else sliding_window
            return GroupedQueryAttention(
                width,
                num_heads,
                num_kv_heads or num_heads,
                key_dim,
                causal=causal,
                dropout=dropout,
                qk_norm=qk_norm,
                sliding_window=window,
            )

        def make_ffn():
            if ffn == "moe":
                return MixtureOfExperts(
                    width, num_experts, experts_per_token, ffn_dim, dropout, shared_experts
                )
            if ffn == "swiglu":
                return SwiGLU(width, ffn_dim, dropout)
            return nn.Sequential(
                nn.Linear(width, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, width)
            )

        self.attentions = nn.ModuleList(make_attention(index) for index in range(layers))
        self.ffns = nn.ModuleList(make_ffn() for _ in range(layers))

    def forward(self, x):
        for attn_norm, attention, ffn_norm, ffn in zip(
            self.attn_norms, self.attentions, self.ffn_norms, self.ffns
        ):
            x = x + attention(attn_norm(x))
            x = x + ffn(ffn_norm(x))
        return x'''

TIED_HEAD_HELPER = '''class TiedLMHead(nn.Module):
    """Vocabulary projection that reuses the input embedding table.

    Shares the embedding parameter rather than learning a second matrix —
    `tie_word_embeddings` in transformers. At a 262k vocabulary this removes
    hundreds of millions of parameters.
    """

    def __init__(self, vocab_size: int, softcap: float = 0.0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.softcap = softcap

    def forward(self, x, embedding_weight):
        logits = F.linear(x, embedding_weight[: self.vocab_size])
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits'''

SOFTCAP_HELPER = '''class SoftcappedLinear(nn.Module):
    """Vocabulary projection whose logits are bounded by tanh.

    `tanh(logit / cap) * cap`, matching Gemma's final_logit_softcapping.
    """

    def __init__(self, width: int, vocab_size: int, softcap: float) -> None:
        super().__init__()
        self.projection = nn.Linear(width, vocab_size)
        self.softcap = softcap

    def forward(self, x):
        logits = self.projection(x)
        return torch.tanh(logits / self.softcap) * self.softcap'''

TORCH_HELPER_SOURCE: dict[str, str] = {
    "rope": ROPE_HELPER,
    "rope_apply": ROPE_APPLY_HELPER_TORCH,
    "attention_mask": ATTENTION_MASK_HELPER,
    "positional": POSITIONAL_HELPER,
    "swiglu": SWIGLU_HELPER,
    "gated_ffn": GATED_FFN_HELPER,
    "squeeze_excite": SQUEEZE_EXCITE_HELPER,
    "patch_embedding": PATCH_EMBEDDING_HELPER,
    "layer_scale": LAYER_SCALE_HELPER,
    "gqa": GQA_HELPER,
    "family_attention": FAMILY_ATTENTION_HELPER,
    "latent_attention": LATENT_ATTENTION_HELPER,
    "moe": MOE_HELPER,
    "sparse_moe": SPARSE_MOE_HELPER,
    "stack": TRANSFORMER_STACK_HELPER,
    # `llm_block` is built by `llm_block_source` from the required set rather
    # than looked up here; the key still appears in TORCH_HELPER_ORDER.
    "tied_head": TIED_HEAD_HELPER,
    "softcap": SOFTCAP_HELPER,
    "resnet": RESNET_HELPER,
    "inverted_residual": INVERTED_RESIDUAL_HELPER,
    "dense_block": DENSE_BLOCK_HELPER,
    "inception": INCEPTION_HELPER,
    "convnext": CONVNEXT_HELPER,
    "vit": VIT_HELPER,
}
# Emission order, so a helper never references one defined below it.
TORCH_HELPER_ORDER = [
    "rope",
    "rope_apply",
    "attention_mask",
    "positional",
    "swiglu",
    "gated_ffn",
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
    "stack",
    "llm_block",
    "resnet",
    "inverted_residual",
    "dense_block",
    "inception",
    "convnext",
    "vit",
]

# Family blocks are absent here: their helpers depend on how they are
# configured, so `emit_torch_module` asks `blocks.llm_block_structures` per node
# — the same function the Keras emitter uses, so the two files carry the same
# structures.
TORCH_HELPER_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "rotary_embedding": ("rope",),
    "positional_embedding": ("positional",),
    "swiglu": ("swiglu",),
    "grouped_query_attention": ("gqa",),
    "moe_feed_forward": ("swiglu", "moe"),
    "transformer_block": ("swiglu", "gqa", "moe", "stack"),
    "squeeze_excite": ("squeeze_excite",),
    "patch_embedding": ("patch_embedding",),
    "geglu": ("gated_ffn",),
    "mla_attention": ("rope_apply", "attention_mask", "latent_attention"),
    "resnet_block": ("resnet",),
    "inverted_residual_block": ("squeeze_excite", "inverted_residual"),
    "dense_block": ("dense_block",),
    "inception_block": ("inception",),
    "convnext_block": ("layer_scale", "convnext"),
    "vit_block": ("vit",),
}
