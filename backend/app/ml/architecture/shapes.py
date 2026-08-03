"""Analytic shape inference and parameter estimation.

Runs on the request path, so it must not import TensorFlow — it reimplements
the shape arithmetic of the layers the catalog exposes. It is deliberately an
approximation: `None` means "cannot be determined without building the model",
and the `/compile` subprocess is the authoritative answer.

Rules are keyed by the same `type` string as `catalog.NODE_SPECS` and
`emit_keras.EMITTERS`; a test asserts the three key sets match.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from app.ml.architecture.blocks import LLM_FAMILIES
from app.ml.architecture.graph import ResolvedGraph, ResolvedNode, parse_shape
from app.schemas import ArchitectureIssue

# A shape excludes the batch dimension. `None` for the whole shape means
# unknown; `None` for one dimension means that axis is dynamic.
Shape = list[int | None] | None


class ShapeError(ValueError):
    """A shape the user must fix — reported against the node that raised it."""


# Final channel count of each tf.keras.applications backbone with
# include_top=False. Needed because the analytic pass cannot introspect a
# backbone it never builds.
BACKBONE_FEATURES: dict[str, int] = {
    "MobileNetV2": 1280,
    "EfficientNetB0": 1280,
    "EfficientNetB1": 1280,
    "EfficientNetB2": 1408,
    "EfficientNetB3": 1536,
    "EfficientNetB4": 1792,
    "EfficientNetB5": 2048,
    "EfficientNetB6": 2304,
    "EfficientNetB7": 2560,
    "EfficientNetV2B0": 1280,
    "EfficientNetV2B3": 1536,
    "EfficientNetV2S": 1280,
    "ResNet50": 2048,
    "Xception": 2048,
    "InceptionV3": 2048,
    "DenseNet121": 1024,
    "ConvNeXtTiny": 768,
}
# Every backbone above reduces spatial dimensions by this factor overall.
BACKBONE_STRIDE = 32


def infer_shapes(
    resolved: ResolvedGraph, num_classes: int | None = None
) -> tuple[dict[str, Shape], int | None, list[ArchitectureIssue]]:
    """Walk the graph in topological order, resolving each node's output shape.

    Returns `(shapes, total_params_estimate, issues)`. A node that raises
    `ShapeError` gets an error issue and an unknown shape, and its successors
    degrade to unknown rather than cascading errors — one broken wire should
    produce one message, not ten.
    """

    shapes: dict[str, Shape] = {}
    issues: list[ArchitectureIssue] = []
    params_total = 0
    params_known = True

    for node_id in resolved.order:
        node = resolved.nodes[node_id]
        input_shapes = [shapes.get(source) for source in node.inputs]
        rule = SHAPE_RULES.get(node.type)
        if rule is None:
            shapes[node_id] = None
            params_known = False
            continue
        try:
            shapes[node_id] = rule(node, input_shapes, num_classes)
        except ShapeError as error:
            shapes[node_id] = None
            issues.append(
                ArchitectureIssue(severity="error", node_id=node_id, message=str(error))
            )
            continue

        counter = PARAM_RULES.get(node.type)
        if counter is None:
            continue
        count = counter(node, input_shapes, shapes[node_id], num_classes)
        if count is None:
            params_known = False
        else:
            params_total += count

    return shapes, (params_total if params_known else None), issues


# --- helpers ---------------------------------------------------------------


def _only(inputs: list[Shape]) -> Shape:
    return inputs[0] if inputs else None


def _require_rank(shape: Shape, rank: int, name: str) -> list[int | None]:
    if shape is None:
        raise _Unknown()
    if len(shape) != rank:
        raise ShapeError(
            f"{name} expects a rank-{rank} input (excluding batch) but got {_fmt(shape)}."
        )
    return shape


class _Unknown(Exception):
    """Internal: an input shape is unknown, so the output is too."""


def _fmt(shape: Shape) -> str:
    if shape is None:
        return "unknown"
    return "(" + ", ".join("?" if dim is None else str(dim) for dim in shape) + ")"


def _conv_out(size: int | None, kernel: int, stride: int, padding: str) -> int | None:
    if size is None:
        return None
    if padding == "same":
        return math.ceil(size / stride)
    out = size - kernel + 1
    if out <= 0:
        raise ShapeError(
            f"A {kernel}×{kernel} kernel with 'valid' padding does not fit a size-{size} axis."
        )
    return math.ceil(out / stride)


def _unknown_safe(rule: Callable) -> Callable:
    """Turn an internal `_Unknown` into an unknown output shape."""

    def wrapped(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
        try:
            return rule(node, inputs, num_classes)
        except _Unknown:
            return None

    return wrapped


# --- rules -----------------------------------------------------------------


def _input(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = parse_shape(node.params.get("shape"))
    if shape is None:
        raise ShapeError(
            "Input shape must be comma-separated whole numbers, for example 224,224,3."
        )
    if any(dim is not None and dim <= 0 for dim in shape):
        raise ShapeError("Input dimensions must be positive.")
    return shape


def _identity(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    return _only(inputs)


def _dense(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _only(inputs)
    if shape is None:
        raise _Unknown()
    if not shape:
        raise ShapeError("Dense needs an input with at least one dimension.")
    return [*shape[:-1], _dense_units(node, num_classes)]


def _dense_units(node: ResolvedNode, num_classes: int | None) -> int | None:
    if node.params.get("units_from_dataset"):
        return num_classes
    return int(node.params.get("units", 1))


def _flatten(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _only(inputs)
    if shape is None:
        raise _Unknown()
    if any(dim is None for dim in shape):
        return [None]
    total = 1
    for dim in shape:
        total *= int(dim)  # type: ignore[arg-type]
    return [total]


def _reshape(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    target = parse_shape(node.params.get("target_shape"))
    if target is None:
        raise ShapeError("Target shape must be comma-separated whole numbers.")
    source = _only(inputs)
    if source is not None and not any(dim is None for dim in source) and not any(
        dim is None for dim in target
    ):
        before = math.prod(int(dim) for dim in source)  # type: ignore[arg-type]
        after = math.prod(int(dim) for dim in target)  # type: ignore[arg-type]
        if before != after:
            raise ShapeError(
                f"Cannot reshape {_fmt(source)} ({before} values) into "
                f"{_fmt(target)} ({after} values)."
            )
    return target


def _embedding(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _only(inputs)
    if shape is None:
        raise _Unknown()
    return [*shape, int(node.params.get("output_dim", 1))]


def _conv1d(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 2, "Conv1D")
    length = _conv_out(
        shape[0],
        int(node.params.get("kernel_size", 1)),
        int(node.params.get("strides", 1)),
        str(node.params.get("padding", "same")),
    )
    return [length, int(node.params.get("filters", 1))]


def _conv2d(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 3, node.spec.name)
    kernel = int(node.params.get("kernel_size", 1))
    stride = int(node.params.get("strides", 1))
    padding = str(node.params.get("padding", "same"))
    return [
        _conv_out(shape[0], kernel, stride, padding),
        _conv_out(shape[1], kernel, stride, padding),
        int(node.params.get("filters", 1)),
    ]


def _conv2d_transpose(
    node: ResolvedNode, inputs: list[Shape], num_classes: int | None
) -> Shape:
    shape = _require_rank(_only(inputs), 3, "Conv2DTranspose")
    kernel = int(node.params.get("kernel_size", 1))
    stride = int(node.params.get("strides", 1))
    padding = str(node.params.get("padding", "same"))

    def out(size: int | None) -> int | None:
        if size is None:
            return None
        return size * stride if padding == "same" else (size - 1) * stride + kernel

    return [out(shape[0]), out(shape[1]), int(node.params.get("filters", 1))]


def _pool2d(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 3, node.spec.name)
    pool = int(node.params.get("pool_size", 2))
    # Keras defaults `strides` to the pool size; the catalog spells that as 0
    # because AdvancedParameterSpec has no "unset" value for a number.
    stride = int(node.params.get("strides", 0)) or pool
    padding = str(node.params.get("padding", "valid"))
    return [
        _conv_out(shape[0], pool, stride, padding),
        _conv_out(shape[1], pool, stride, padding),
        shape[2],
    ]


def _global_pool2d(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 3, node.spec.name)
    return [shape[2]]


def _up_sampling2d(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 3, "UpSampling2D")
    size = int(node.params.get("size", 2))
    return [
        None if shape[0] is None else shape[0] * size,
        None if shape[1] is None else shape[1] * size,
        shape[2],
    ]


def _zero_padding2d(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 3, "ZeroPadding2D")
    pad = int(node.params.get("padding", 0))
    return [
        None if shape[0] is None else shape[0] + 2 * pad,
        None if shape[1] is None else shape[1] + 2 * pad,
        shape[2],
    ]


def _group_norm(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _only(inputs)
    if shape is None:
        raise _Unknown()
    channels = shape[-1]
    groups = int(node.params.get("groups", 1))
    if channels is not None and groups > 0 and channels % groups != 0:
        raise ShapeError(
            f"GroupNormalization needs the channel count ({channels}) to divide evenly "
            f"into {groups} groups."
        )
    return shape


def _recurrent(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 2, node.spec.name)
    units = int(node.params.get("units", 1))
    if node.params.get("bidirectional"):
        units *= 2
    if node.params.get("return_sequences"):
        return [shape[0], units]
    return [units]


def _merge(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    known = [shape for shape in inputs if shape is not None]
    if not known:
        raise _Unknown()
    reference = known[0]
    for other in known[1:]:
        if len(other) != len(reference) or not _dims_compatible(reference, other):
            raise ShapeError(
                f"{node.spec.name} needs matching input shapes but got "
                f"{_fmt(reference)} and {_fmt(other)}."
            )
    return list(reference)


def _dims_compatible(left: list[int | None], right: list[int | None]) -> bool:
    return all(a is None or b is None or a == b for a, b in zip(left, right, strict=True))


def _concatenate(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    known = [shape for shape in inputs if shape is not None]
    if len(known) != len(inputs) or not known:
        raise _Unknown()
    rank = len(known[0])
    if any(len(shape) != rank for shape in known):
        raise ShapeError(
            "Concatenate needs every input to have the same rank; got "
            + ", ".join(_fmt(shape) for shape in known)
            + "."
        )
    axis = int(node.params.get("axis", -1))
    index = axis + rank if axis < 0 else axis
    if not 0 <= index < rank:
        raise ShapeError(f"Axis {axis} is out of range for a rank-{rank} input.")
    for position in range(rank):
        if position == index:
            continue
        values = {shape[position] for shape in known if shape[position] is not None}
        if len(values) > 1:
            raise ShapeError(
                f"Concatenate on axis {axis} needs every other dimension to match, but "
                f"axis {position} differs: {sorted(values)}."
            )
    sizes = [shape[index] for shape in known]
    total = None if any(size is None for size in sizes) else sum(size for size in sizes)  # type: ignore[misc]
    return [*known[0][:index], total, *known[0][index + 1 :]]


def _backbone(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 3, "Pretrained backbone")
    application = str(node.params.get("application", ""))
    features = BACKBONE_FEATURES.get(application)
    pooling = str(node.params.get("pooling", "avg"))
    if pooling in {"avg", "max"}:
        return [features]
    return [
        None if shape[0] is None else max(1, shape[0] // BACKBONE_STRIDE),
        None if shape[1] is None else max(1, shape[1] // BACKBONE_STRIDE),
        features,
    ]


# --- transformer and custom rules ------------------------------------------


def _sequence_identity(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    """Norms, RoPE, and positional embeddings preserve `(sequence, features)`."""

    return _require_rank(_only(inputs), 2, node.spec.name)


def _attention(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    """Keras MultiHeadAttention defaults its output width to the query's."""

    shape = _require_rank(_only(inputs), 2, "Multi-head attention")
    return list(shape)


def _feed_forward(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    """Both feed-forward kinds project back to the model width they were given."""

    return _require_rank(_only(inputs), 2, node.spec.name)


def _transformer_block(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 2, "Transformer block")
    model_dim = shape[1]
    heads = int(node.params.get("num_heads", 1))
    if model_dim is not None and model_dim % heads != 0:
        raise ShapeError(
            f"A transformer block needs its model width ({model_dim}) to divide evenly "
            f"into {heads} heads."
        )
    kv_heads = int(node.params.get("num_kv_heads", 0) or heads)
    if kv_heads > heads:
        raise ShapeError(
            f"Grouped-query attention needs at most as many key/value heads as query heads "
            f"({kv_heads} > {heads})."
        )
    if heads % kv_heads != 0:
        raise ShapeError(
            f"Query heads ({heads}) must divide evenly into key/value head groups ({kv_heads})."
        )
    if str(node.params.get("ffn")) == "moe":
        experts = int(node.params.get("num_experts", 1))
        active = int(node.params.get("experts_per_token", 1))
        if active > experts:
            raise ShapeError(
                f"A mixture of experts cannot route each token to {active} of only {experts} experts."
            )
    if int(node.params.get("global_every", 0) or 0) and not int(
        node.params.get("sliding_window", 0) or 0
    ):
        raise ShapeError(
            "“Global layer every” only means something alongside a sliding window — "
            "without one, every layer is already global."
        )
    return list(shape)


def _grouped_attention(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 2, "Grouped-query attention")
    heads = int(node.params.get("num_heads", 1))
    kv_heads = int(node.params.get("num_kv_heads", 1))
    if kv_heads > heads:
        raise ShapeError(
            f"Grouped-query attention needs at most as many key/value heads as query heads "
            f"({kv_heads} > {heads})."
        )
    if kv_heads > 0 and heads % kv_heads != 0:
        raise ShapeError(
            f"Query heads ({heads}) must divide evenly into key/value head groups ({kv_heads})."
        )
    return list(shape)


def _lm_head(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 2, "LM head")
    return [shape[0], _lm_vocab(node, num_classes)]


def _lm_vocab(node: ResolvedNode, num_classes: int | None) -> int | None:
    # The runner passes the corpus vocabulary through the same `num_classes`
    # channel a classifier head uses, so one signature covers both.
    if node.params.get("vocab_from_dataset"):
        return num_classes
    return int(node.params.get("vocab_size", 2))


def _global_pool1d(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 2, "GlobalAvgPool1D")
    return [shape[1]]


def _declared_or_unknown(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    """Custom nodes: honour a declared output shape, else pass the input through.

    A blank `output_shape` means "this does not change the shape", which is
    true of most custom layers and keeps downstream nodes resolvable. Anything
    else the user must declare, because the analytic pass cannot read Python.
    """

    declared = parse_shape(node.params.get("output_shape") or None)
    if declared is not None:
        return declared
    return _only(inputs)


# --- new primitive rules ----------------------------------------------------


def _depthwise_conv2d(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    height, width, channels = _require_rank(_only(inputs), 3, "DepthwiseConv2D")
    kernel = int(node.params.get("kernel_size", 1))
    stride = int(node.params.get("strides", 1))
    padding = str(node.params.get("padding", "same"))
    multiplier = int(node.params.get("depth_multiplier", 1))
    return [
        _conv_out(height, kernel, stride, padding),
        _conv_out(width, kernel, stride, padding),
        None if channels is None else channels * multiplier,
    ]


def _squeeze_excite(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    # Rescales channels in place, so the shape is untouched.
    return list(_require_rank(_only(inputs), 3, "Squeeze-and-excite"))


def _patch_embedding(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    height, width, _ = _require_rank(_only(inputs), 3, "Patch embedding")
    patch = int(node.params.get("patch_size", 1))
    embed = int(node.params.get("embed_dim", 1))
    if height is not None and height % patch:
        raise ShapeError(
            f"A patch size of {patch} does not divide a height of {height} evenly. "
            "ViT needs whole patches; resize the input or change the patch size."
        )
    if width is not None and width % patch:
        raise ShapeError(
            f"A patch size of {patch} does not divide a width of {width} evenly."
        )
    if height is None or width is None:
        return [None, embed]
    tokens = (height // patch) * (width // patch)
    return [tokens + (1 if node.params.get("class_token") else 0), embed]


def _mla_attention(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 2, "Multi-head latent attention")
    # The output projection maps back to the residual stream's width.
    return list(shape)


# --- vision block rules -----------------------------------------------------


def _resnet_block(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    height, width, _ = _require_rank(_only(inputs), 3, "ResNet stage")
    stride = int(node.params.get("stride", 1))
    filters = int(node.params.get("filters", 1))
    if str(node.params.get("variant", "basic")) == "bottleneck":
        filters *= int(node.params.get("expansion", 4))
    return [
        None if height is None else math.ceil(height / stride),
        None if width is None else math.ceil(width / stride),
        filters,
    ]


def _inverted_residual_block(
    node: ResolvedNode, inputs: list[Shape], num_classes: int | None
) -> Shape:
    height, width, _ = _require_rank(_only(inputs), 3, "Inverted residual")
    # Only the first block in a repeat carries the stride; the rest are 1, the
    # same convention torchvision's _mobilenet_extractor uses.
    stride = int(node.params.get("stride", 1))
    return [
        None if height is None else math.ceil(height / stride),
        None if width is None else math.ceil(width / stride),
        int(node.params.get("filters", 1)),
    ]


def _dense_block(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    height, width, channels = _require_rank(_only(inputs), 3, "DenseNet block")
    growth = int(node.params.get("growth_rate", 1))
    layers = int(node.params.get("layers", 1))
    return [height, width, None if channels is None else channels + layers * growth]


def _inception_block(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    height, width, _ = _require_rank(_only(inputs), 3, "Inception module")
    total = sum(
        int(node.params.get(key, 0))
        for key in ("filters_1x1", "filters_3x3", "filters_5x5", "filters_pool")
    )
    return [height, width, total]


def _convnext_block(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    height, width, _ = _require_rank(_only(inputs), 3, "ConvNeXt block")
    return [height, width, int(node.params.get("filters", 1))]


def _vit_block(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    shape = _require_rank(_only(inputs), 2, "ViT encoder block")
    width = shape[1]
    heads = int(node.params.get("num_heads", 1) or 1)
    head_dim = int(node.params.get("head_dim", 1) or 1)
    if width is not None and heads * head_dim != width:
        raise ShapeError(
            f"A ViT block's heads × head dim must equal the token width: "
            f"{heads} × {head_dim} = {heads * head_dim}, but patches are {width} wide. "
            f"ViT-Base is 12 heads of 64 against 768."
        )
    return list(shape)


# --- named LLM family blocks ------------------------------------------------


def _llm_block(node: ResolvedNode, inputs: list[Shape], num_classes: int | None) -> Shape:
    """Every family block preserves `(sequence, width)` and validates its config.

    The checks here are the ones that would otherwise surface as an opaque
    reshape failure inside the training subprocess twenty minutes later.
    """

    shape = _require_rank(_only(inputs), 2, node.spec.name)
    params = node.params
    attention = str(params.get("attention", "gqa"))
    heads = int(params.get("num_heads", 1) or 1)

    if attention != "mla":
        kv_heads = int(params.get("num_kv_heads", 0) or heads)
        if kv_heads > heads:
            raise ShapeError(
                f"{node.spec.name} needs at most as many key/value heads as query heads "
                f"({kv_heads} > {heads})."
            )
        if heads % kv_heads:
            raise ShapeError(
                f"Query heads ({heads}) must divide evenly into key/value head groups "
                f"({kv_heads})."
            )
    else:
        rope_dim = int(params.get("qk_rope_head_dim", 0) or 0)
        if rope_dim % 2:
            raise ShapeError(
                f"Latent attention rotates feature pairs, so the RoPE head dim must be even "
                f"(got {rope_dim})."
            )

    if str(params.get("ffn")) == "moe":
        experts = int(params.get("num_experts", 1) or 1)
        active = int(params.get("experts_per_token", 1) or 1)
        if active > experts:
            raise ShapeError(
                f"Each token cannot be routed to {active} of only {experts} experts."
            )
        if int(params.get("dense_layers", 0) or 0) > int(params.get("layers", 1) or 1):
            raise ShapeError(
                "More leading dense layers than there are layers in the stack."
            )
    if int(params.get("global_every", 0) or 0) and not int(params.get("sliding_window", 0) or 0):
        raise ShapeError(
            "“Global layer every” only means something alongside a sliding window — "
            "without one, every layer is already global."
        )
    return list(shape)


SHAPE_RULES: dict[str, Callable[[ResolvedNode, list[Shape], int | None], Shape]] = {
    node_type: _unknown_safe(rule)
    for node_type, rule in {
        "depthwise_conv2d": _depthwise_conv2d,
        "squeeze_excite": _squeeze_excite,
        "patch_embedding": _patch_embedding,
        "geglu": _feed_forward,
        "mla_attention": _mla_attention,
        "resnet_block": _resnet_block,
        "inverted_residual_block": _inverted_residual_block,
        "dense_block": _dense_block,
        "inception_block": _inception_block,
        "convnext_block": _convnext_block,
        "vit_block": _vit_block,
        **{family.type: _llm_block for family in LLM_FAMILIES},
        "input": _input,
        "output": _identity,
        "dense": _dense,
        "activation": _identity,
        "flatten": _flatten,
        "reshape": _reshape,
        "embedding": _embedding,
        "conv1d": _conv1d,
        "conv2d": _conv2d,
        "separable_conv2d": _conv2d,
        "conv2d_transpose": _conv2d_transpose,
        "max_pool2d": _pool2d,
        "avg_pool2d": _pool2d,
        "global_avg_pool2d": _global_pool2d,
        "global_max_pool2d": _global_pool2d,
        "up_sampling2d": _up_sampling2d,
        "zero_padding2d": _zero_padding2d,
        "batch_norm": _identity,
        "layer_norm": _identity,
        "group_norm": _group_norm,
        "lstm": _recurrent,
        "gru": _recurrent,
        "add": _merge,
        "multiply": _merge,
        "average": _merge,
        "subtract": _merge,
        "concatenate": _concatenate,
        # Augmentation resamples pixels; it never changes the image's shape,
        # and it carries no weights so PARAM_RULES leaves it at zero.
        "random_flip": _identity,
        "random_rotation": _identity,
        "random_zoom": _identity,
        "random_contrast": _identity,
        "dropout": _identity,
        "spatial_dropout2d": _identity,
        "gaussian_noise": _identity,
        "activity_regularization": _identity,
        "pretrained_backbone": _backbone,
        "positional_embedding": _sequence_identity,
        "rotary_embedding": _sequence_identity,
        "multi_head_attention": _attention,
        "rms_norm": _sequence_identity,
        "swiglu": _feed_forward,
        "feed_forward": _feed_forward,
        "transformer_block": _transformer_block,
        "lm_head": _lm_head,
        "global_avg_pool1d": _global_pool1d,
        "moe_feed_forward": _feed_forward,
        "grouped_query_attention": _grouped_attention,
        "custom_layer": _declared_or_unknown,
        "custom_function": _declared_or_unknown,
    }.items()
}


# --- parameter estimation --------------------------------------------------
#
# Only nodes with weights appear here; anything absent contributes nothing.
# A rule returning `None` means "unknown", which makes the whole estimate
# `None` rather than quietly reporting a total that is missing a backbone's
# several million parameters.

ParamRule = Callable[[ResolvedNode, list[Shape], Shape, int | None], int | None]


def _dense_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    units = _dense_units(node, num_classes)
    if shape is None or not shape or shape[-1] is None or units is None:
        return None
    weights = int(shape[-1]) * units
    return weights + (units if node.params.get("use_bias", True) else 0)


def _conv_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or shape[-1] is None:
        return None
    channels = int(shape[-1])
    kernel = int(node.params.get("kernel_size", 1))
    filters = int(node.params.get("filters", 1))
    window = kernel if node.type == "conv1d" else kernel * kernel
    if node.type == "separable_conv2d":
        return window * channels + channels * filters + filters
    return window * channels * filters + filters


def _embedding_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    return int(node.params.get("input_dim", 0)) * int(node.params.get("output_dim", 0))


def _norm_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or shape[-1] is None:
        return None
    # BatchNormalization also carries non-trainable moving mean and variance.
    return int(shape[-1]) * (4 if node.type == "batch_norm" else 2)


def _recurrent_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 2 or shape[-1] is None:
        return None
    features = int(shape[-1])
    units = int(node.params.get("units", 1))
    if node.type == "lstm":
        per_direction = 4 * ((features + units) * units + units)
    else:
        # Keras GRU defaults to reset_after=True, which carries two bias vectors.
        per_direction = 3 * ((features + units) * units + 2 * units)
    return per_direction * (2 if node.params.get("bidirectional") else 1)


def _backbone_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    # Deliberately unknown: the analytic pass never builds the backbone, and
    # guessing would be worse than saying so. `/compile` reports the real count.
    return None


def _positional_embedding_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 2 or shape[1] is None:
        return None
    return int(node.params.get("max_length", 0)) * int(shape[1])


def _attention_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 2 or shape[1] is None:
        return None
    heads = int(node.params.get("num_heads", 1))
    return _attention_projection_params(
        int(shape[1]), heads, heads, int(node.params.get("key_dim", 1))
    )


def _norm_scale_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or shape[-1] is None:
        return None
    # RMSNorm carries a scale but no bias.
    return int(shape[-1])


def _feed_forward_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 2 or shape[1] is None:
        return None
    model_dim = int(shape[1])
    hidden = int(node.params.get("hidden_dim", 1))
    if node.type == "geglu":
        # Gate, up, down — and no biases, which is what Gemma's MLP and every
        # other published gated feed-forward ships. The named family blocks use
        # the same bias-free `GatedFeedForward`.
        return 3 * model_dim * hidden
    if node.type == "swiglu":
        # The older standalone SwiGLU layer keeps its biases so saved graphs and
        # the generic transformer block keep their counts. Prefer `geglu` or a
        # family block when the goal is matching a published model exactly.
        return 2 * (model_dim * hidden + hidden) + hidden * model_dim + model_dim
    return model_dim * hidden + hidden + hidden * model_dim + model_dim


def _transformer_block_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 2 or shape[1] is None:
        return None
    model_dim = int(shape[1])
    layers = int(node.params.get("layers", 1))
    heads = int(node.params.get("num_heads", 1))
    key_dim = int(node.params.get("key_dim", 1))
    hidden = int(node.params.get("ffn_dim", 1))
    uses_rms = str(node.params.get("norm", "rms")) == "rms"

    kv_heads = int(node.params.get("num_kv_heads", 0) or heads)
    attention = _attention_projection_params(
        model_dim, heads, kv_heads, key_dim, bool(node.params.get("qk_norm"))
    )
    kind = str(node.params.get("ffn", "swiglu"))
    if kind == "moe":
        experts = int(node.params.get("num_experts", 1))
        shared = int(node.params.get("shared_experts", 0) or 0)
        expert = 2 * (model_dim * hidden + hidden) + hidden * model_dim + model_dim
        ffn = (experts + shared) * expert + model_dim * experts + experts
    elif kind == "swiglu":
        ffn = 2 * (model_dim * hidden + hidden) + hidden * model_dim + model_dim
    else:
        ffn = model_dim * hidden + hidden + hidden * model_dim + model_dim
    norms = 2 * (model_dim if uses_rms else 2 * model_dim)
    return layers * (attention + ffn + norms)


def _moe_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    """Total parameters, not active ones.

    A mixture of experts holds every expert's weights; only a few run per
    token. The canvas reports the total because that is what has to fit in
    memory.
    """

    shape = _only(inputs)
    if shape is None or len(shape) != 2 or shape[1] is None:
        return None
    model_dim = int(shape[1])
    experts = int(node.params.get("num_experts", 1))
    hidden = int(node.params.get("hidden_dim", 1))
    # A gated (SwiGLU) expert: gate and up projections, then down.
    expert = 2 * (model_dim * hidden + hidden) + hidden * model_dim + model_dim
    router = model_dim * experts + experts
    # Shared experts are always-on and unrouted, so they add weights without
    # widening the router — the DeepSeek design.
    shared = int(node.params.get("shared_experts", 0) or 0)
    return (experts + shared) * expert + router


def _grouped_attention_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 2 or shape[1] is None:
        return None
    return _attention_projection_params(
        int(shape[1]),
        int(node.params.get("num_heads", 1)),
        int(node.params.get("num_kv_heads", 1)),
        int(node.params.get("key_dim", 1)),
        bool(node.params.get("qk_norm")),
    )


def _attention_projection_params(
    model_dim: int,
    heads: int,
    kv_heads: int,
    key_dim: int,
    qk_norm: bool = False,
    bias: bool = True,
) -> int:
    """Q/K/V/O projections, with K and V sized by the key-value head count.

    Shrinking the KV heads is the whole point of grouped-query attention, so
    the estimate has to reflect it or a GQA model looks the same size as an
    MHA one.

    `bias` is False for every RoPE-era model — Llama, Qwen, Mistral, Gemma and
    DeepSeek all ship `attention_bias: false` — and True for GPT-2 and BERT.
    """

    query = model_dim * heads * key_dim + (heads * key_dim if bias else 0)
    key_value = 2 * (model_dim * kv_heads * key_dim + (kv_heads * key_dim if bias else 0))
    output = heads * key_dim * model_dim + (model_dim if bias else 0)
    # QK-norm is one RMSNorm scale vector per head dimension, on q and on k.
    norms = 2 * key_dim if qk_norm else 0
    return query + key_value + output + norms


def _lm_head_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    vocab = _lm_vocab(node, num_classes)
    if shape is None or len(shape) != 2 or shape[1] is None or vocab is None:
        return None
    if node.params.get("tie_embeddings"):
        # The projection reuses the embedding matrix, so it contributes no
        # weights of its own. On a large-vocabulary model this is the single
        # biggest saving available — Gemma 3's 262k vocabulary at width 2560
        # is 671M parameters that tying removes.
        return 0
    # No bias: the softmax that follows is shift-invariant, so every published
    # causal LM head omits it, and at these vocabularies it is not a rounding
    # error to count one anyway.
    return int(shape[1]) * vocab


def _unknown_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    # Custom Python is opaque to the analytic pass; counting it would be a guess.
    return None


# --- new primitive parameter rules ------------------------------------------
#
# Convolutions inside a block are emitted with `use_bias=False` where the
# reference implementation does — every BatchNormalization that follows has a
# learned shift, so the convolution's bias is redundant and torchvision, Keras
# Applications, and the original papers all omit it. Counting one anyway is a
# small error per layer and a large one across a ResNet-50.


def _bn(channels: int) -> int:
    """BatchNormalization: gamma, beta, moving mean, moving variance."""

    return 4 * channels


def _depthwise_conv_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or shape[-1] is None:
        return None
    channels = int(shape[-1])
    kernel = int(node.params.get("kernel_size", 1))
    multiplier = int(node.params.get("depth_multiplier", 1))
    filters = channels * multiplier
    return kernel * kernel * filters + filters


def _squeeze_excite_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or shape[-1] is None:
        return None
    channels = int(shape[-1])
    squeezed = max(1, channels // int(node.params.get("ratio", 4)))
    return channels * squeezed + squeezed + squeezed * channels + channels


def _patch_embedding_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 3 or shape[-1] is None:
        return None
    patch = int(node.params.get("patch_size", 1))
    embed = int(node.params.get("embed_dim", 1))
    # A strided convolution whose kernel equals its stride — the standard
    # "cut into patches and project" implementation.
    total = patch * patch * int(shape[-1]) * embed + embed
    return total + (embed if node.params.get("class_token") else 0)


def _mla_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 2 or shape[1] is None:
        return None
    return _latent_attention_params(int(shape[1]), node.params)


def _latent_attention_params(model_dim: int, params: dict) -> int:
    """DeepSeek's MLA projections, bias-free as its config states.

    Queries optionally pass through a rank-`q_lora_rank` bottleneck; keys and
    values always pass through a rank-`kv_lora_rank` one that also carries a
    single shared rotary key. That latent is what the cache stores, and it is
    why a 61-layer 128-head model is servable at all.
    """

    heads = int(params.get("num_heads", 1) or 1)
    q_rank = int(params.get("q_lora_rank", 0) or 0)
    kv_rank = int(params.get("kv_lora_rank", 1) or 1)
    rope_dim = int(params.get("qk_rope_head_dim", 0) or 0)
    nope_dim = int(params.get("qk_nope_head_dim", 1) or 1)
    v_dim = int(params.get("v_head_dim", 1) or 1)

    qk_dim = nope_dim + rope_dim
    if q_rank:
        # q_a_proj, its RMSNorm, then q_b_proj.
        query = model_dim * q_rank + q_rank + q_rank * heads * qk_dim
    else:
        query = model_dim * heads * qk_dim
    # kv_a_proj_with_mqa emits the latent plus the one shared rotary key.
    key_value = model_dim * (kv_rank + rope_dim) + kv_rank
    key_value += kv_rank * heads * (nope_dim + v_dim)
    output = heads * v_dim * model_dim
    return query + key_value + output


# --- vision block parameter rules -------------------------------------------


def _resnet_block_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 3 or shape[-1] is None:
        return None
    channels = int(shape[-1])
    filters = int(node.params.get("filters", 1))
    blocks = int(node.params.get("blocks", 1))
    stride = int(node.params.get("stride", 1))
    bottleneck = str(node.params.get("variant", "basic")) == "bottleneck"
    expansion = int(node.params.get("expansion", 4)) if bottleneck else 1
    out_channels = filters * expansion

    total = 0
    for index in range(blocks):
        source = channels if index == 0 else out_channels
        block_stride = stride if index == 0 else 1
        if bottleneck:
            total += source * filters + _bn(filters)
            total += 9 * filters * filters + _bn(filters)
            total += filters * out_channels + _bn(out_channels)
        else:
            total += 9 * source * filters + _bn(filters)
            total += 9 * filters * filters + _bn(filters)
        # The projection shortcut, only where the shape actually changes.
        if source != out_channels or block_stride != 1:
            total += source * out_channels + _bn(out_channels)
    return total


def _inverted_residual_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 3 or shape[-1] is None:
        return None
    channels = int(shape[-1])
    filters = int(node.params.get("filters", 1))
    ratio = int(node.params.get("expand_ratio", 1))
    kernel = int(node.params.get("kernel_size", 3))
    blocks = int(node.params.get("blocks", 1))
    use_se = bool(node.params.get("use_se"))
    se_ratio = max(1, int(node.params.get("se_ratio", 4)))

    total = 0
    for index in range(blocks):
        source = channels if index == 0 else filters
        hidden = source * ratio
        if ratio != 1:
            total += source * hidden + _bn(hidden)
        total += kernel * kernel * hidden + _bn(hidden)
        if use_se:
            squeezed = max(1, hidden // se_ratio)
            total += hidden * squeezed + squeezed + squeezed * hidden + hidden
        total += hidden * filters + _bn(filters)
    return total


def _dense_block_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 3 or shape[-1] is None:
        return None
    channels = int(shape[-1])
    growth = int(node.params.get("growth_rate", 1))
    layers = int(node.params.get("layers", 1))
    inner = growth * int(node.params.get("bottleneck_ratio", 4))

    total = 0
    for index in range(layers):
        width = channels + index * growth
        total += _bn(width) + width * inner
        total += _bn(inner) + 9 * inner * growth
    return total


def _inception_block_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 3 or shape[-1] is None:
        return None
    channels = int(shape[-1])

    def conv(source: int, filters: int, kernel: int) -> int:
        return kernel * kernel * source * filters + filters

    reduce_3 = int(node.params.get("reduce_3x3", 1))
    reduce_5 = int(node.params.get("reduce_5x5", 1))
    return (
        conv(channels, int(node.params.get("filters_1x1", 1)), 1)
        + conv(channels, reduce_3, 1)
        + conv(reduce_3, int(node.params.get("filters_3x3", 1)), 3)
        + conv(channels, reduce_5, 1)
        + conv(reduce_5, int(node.params.get("filters_5x5", 1)), 5)
        + conv(channels, int(node.params.get("filters_pool", 1)), 1)
    )


def _convnext_block_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 3 or shape[-1] is None:
        return None
    channels = int(shape[-1])
    filters = int(node.params.get("filters", 1))
    blocks = int(node.params.get("blocks", 1))
    kernel = int(node.params.get("kernel_size", 7))
    hidden = filters * int(node.params.get("expand_ratio", 4))
    scaled = float(node.params.get("layer_scale", 0.0) or 0.0) > 0

    # A 1x1 projection when the stage changes width, mirroring ConvNeXt's
    # downsample layer without the stride.
    total = (channels * filters + filters) if channels != filters else 0
    for _ in range(blocks):
        total += kernel * kernel * filters + filters  # depthwise
        total += 2 * filters  # LayerNorm
        total += filters * hidden + hidden
        total += hidden * filters + filters
        if scaled:
            total += filters
    return total


def _vit_block_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    shape = _only(inputs)
    if shape is None or len(shape) != 2 or shape[1] is None:
        return None
    model_dim = int(shape[1])
    layers = int(node.params.get("layers", 1))
    heads = int(node.params.get("num_heads", 1))
    head_dim = int(node.params.get("head_dim", 1))
    mlp = int(node.params.get("mlp_dim", 1))
    attention = _attention_projection_params(model_dim, heads, heads, head_dim)
    feed_forward = model_dim * mlp + mlp + mlp * model_dim + model_dim
    return layers * (attention + feed_forward + 4 * model_dim)


# --- named LLM family blocks -------------------------------------------------


def _llm_block_params(
    node: ResolvedNode, inputs: list[Shape], output: Shape, num_classes: int | None
) -> int | None:
    """Exact weight count for a stack of one family's blocks.

    Walks the stack rather than multiplying, because two things vary by layer
    index: DeepSeek keeps its first `dense_layers` feed-forwards dense, and a
    hybrid attention pattern promotes every Nth layer to global. Only the first
    changes the count, but writing the loop keeps that visible.
    """

    shape = _only(inputs)
    if shape is None or len(shape) != 2 or shape[1] is None:
        return None
    model_dim = int(shape[1])
    params = node.params
    layers = int(params.get("layers", 1) or 1)
    use_bias = bool(params.get("use_bias"))

    if str(params.get("attention", "gqa")) == "mla":
        attention = _latent_attention_params(model_dim, params)
    else:
        heads = int(params.get("num_heads", 1) or 1)
        attention = _attention_projection_params(
            model_dim,
            heads,
            int(params.get("num_kv_heads", 0) or heads),
            int(params.get("head_dim", 1) or 1),
            bool(params.get("qk_norm")),
            bias=use_bias,
        )

    norm_width = model_dim if str(params.get("norm", "rms")) == "rms" else 2 * model_dim
    # Sandwich placement normalizes each sub-layer's output as well as its
    # input, which is four norms a layer rather than two.
    norms = norm_width * (4 if str(params.get("norm_placement")) == "sandwich" else 2)

    kind = str(params.get("ffn", "swiglu"))
    dense_layers = min(int(params.get("dense_layers", 0) or 0), layers)
    hidden = int(params.get("ffn_dim", 1) or 1)

    def gated(width: int) -> int:
        # gate, up, down — three matrices, and no biases in any published
        # SwiGLU or GeGLU implementation.
        return 3 * model_dim * width

    def classic(width: int) -> int:
        return model_dim * width + width + width * model_dim + model_dim

    if kind == "moe":
        experts = int(params.get("num_experts", 1) or 1)
        shared = int(params.get("shared_experts", 0) or 0)
        # The router is one bias-free matrix. `sigmoid_bias` adds the learned
        # per-expert selection bias DeepSeek uses instead of an auxiliary loss.
        router = model_dim * experts
        if str(params.get("router")) == "sigmoid_bias":
            router += experts
        sparse = (experts + shared) * gated(hidden) + router
        dense = gated(int(params.get("dense_ffn_dim", 1) or 1))
        feed_forward = dense_layers * dense + (layers - dense_layers) * sparse
    elif kind in {"swiglu", "geglu"}:
        feed_forward = layers * gated(hidden)
    else:
        feed_forward = layers * classic(hidden)

    return layers * (attention + norms) + feed_forward


PARAM_RULES: dict[str, ParamRule] = {
    "depthwise_conv2d": _depthwise_conv_params,
    "squeeze_excite": _squeeze_excite_params,
    "patch_embedding": _patch_embedding_params,
    "geglu": _feed_forward_params,
    "mla_attention": _mla_params,
    "resnet_block": _resnet_block_params,
    "inverted_residual_block": _inverted_residual_params,
    "dense_block": _dense_block_params,
    "inception_block": _inception_block_params,
    "convnext_block": _convnext_block_params,
    "vit_block": _vit_block_params,
    **{family.type: _llm_block_params for family in LLM_FAMILIES},
    "dense": _dense_params,
    "conv1d": _conv_params,
    "conv2d": _conv_params,
    "separable_conv2d": _conv_params,
    "conv2d_transpose": _conv_params,
    "embedding": _embedding_params,
    "batch_norm": _norm_params,
    "layer_norm": _norm_params,
    "group_norm": _norm_params,
    "lstm": _recurrent_params,
    "gru": _recurrent_params,
    "pretrained_backbone": _backbone_params,
    "positional_embedding": _positional_embedding_params,
    "multi_head_attention": _attention_params,
    "rms_norm": _norm_scale_params,
    "swiglu": _feed_forward_params,
    "feed_forward": _feed_forward_params,
    "transformer_block": _transformer_block_params,
    "lm_head": _lm_head_params,
    "moe_feed_forward": _moe_params,
    "grouped_query_attention": _grouped_attention_params,
    "custom_layer": _unknown_params,
    "custom_function": _unknown_params,
}
