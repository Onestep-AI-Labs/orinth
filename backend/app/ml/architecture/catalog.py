"""Node palette for the architecture studio.

Every node type is declared exactly once, here, as a serializable `NodeSpec`.
Its `params` are `AdvancedParameterSpec`s — the same type the training form's
advanced accordion already renders — so the studio's node inspector is that
generic field renderer pointed at a spec, and a new node type needs no
frontend change.

The behavior that cannot be expressed declaratively lives in two sibling
registries keyed by the same `type` string: `shapes.SHAPE_RULES` and
`emit_keras.EMITTERS`. `test_architecture_catalog.py` asserts all three share
one key set, which is what keeps a node from being addable but unbuildable.
"""

from typing import Any

from app.ml.architecture.blocks import (
    GEGLU,
    GELU,
    GQA,
    LLM_BLOCK_DEFAULTS,
    LLM_FAMILIES,
    MHA,
    MLA,
    MOE,
    NLP_BLOCKS,
    POST,
    PRE,
    ROUTER_SIGMOID_BIAS,
    ROUTER_SOFTMAX,
    SANDWICH,
    SWIGLU,
    VISION_BLOCKS,
    BlockFamily,
)
from app.ml.common.advanced import code, number, select, toggle
from app.ml.vision.keras_classification.catalog import KERAS_APPLICATION_OPTIONS
from app.schemas import AdvancedParameterSpec, NodePortSpec, NodeSpec, TaskType

# Palette categories, in display order.
IO = "Input & Output"
AUGMENTATION = "Augmentation"
VISION_BLOCK = "Vision blocks"
LLM_BLOCK = "LLM blocks"
NLP = "NLP"
CORE = "Core"
CONVOLUTION = "Convolution"
NORMALIZATION = "Normalization"
RECURRENT = "Recurrent"
MERGE = "Merge"
REGULARIZATION = "Regularization"
BACKBONE = "Backbone"
TRANSFORMER = "Transformer"
CUSTOM = "Custom"

CATEGORY_ORDER = [
    IO,
    # Directly after Input, which is where these belong on the canvas and the
    # only place they do anything: they read the raw image.
    AUGMENTATION,
    VISION_BLOCK,
    LLM_BLOCK,
    NLP,
    CORE,
    CONVOLUTION,
    NORMALIZATION,
    RECURRENT,
    TRANSFORMER,
    MERGE,
    REGULARIZATION,
    BACKBONE,
    CUSTOM,
]

# Param groups inside the node inspector.
SHAPE = "Shape"
LAYER = "Layer"
CODE = "Code"
STACK = "Stack"
ATTENTION = "Attention"
LATENT_ATTENTION = "Latent attention"
FEED_FORWARD = "Feed-forward"
EXPERTS = "Experts"
NORM = "Normalization"
REGULARIZATION_GROUP = "Regularization"

ACTIVATIONS = ["linear", "relu", "gelu", "swish", "tanh", "sigmoid", "softmax", "elu", "selu"]
PADDINGS = ["same", "valid"]

BACKBONE_APPLICATIONS = [option["app_name"] for option in KERAS_APPLICATION_OPTIONS]

# What a freshly dropped custom-layer node starts with: a working identity
# layer, so the graph still compiles before the user has written anything.
CUSTOM_LAYER_TEMPLATE = """class MyLayer(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        return {**super().get_config(), "scale": self.scale}
"""


def text(
    key: str, label: str, *, default: str, group: str = LAYER, help: str | None = None
) -> AdvancedParameterSpec:
    return AdvancedParameterSpec(
        key=key, label=label, type="text", default=default, help=help, group=group
    )


def activation_param(default: str = "linear") -> AdvancedParameterSpec:
    return select("activation", "Activation", options=ACTIVATIONS, default=default, group=LAYER)


def _ports(keys: list[str]) -> list[NodePortSpec]:
    return [NodePortSpec(key=key, label=key.title()) for key in keys]


def _spec(
    node_type: str,
    name: str,
    category: str,
    description: str,
    *,
    params: list[AdvancedParameterSpec] | None = None,
    min_inputs: int = 1,
    max_inputs: int = 1,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    task_types: list[TaskType] | None = None,
    kind: str = "layer",
    source: str = "",
) -> NodeSpec:
    return NodeSpec(
        type=node_type,
        name=name,
        category=category,
        description=description,
        params=params or [],
        inputs=_ports(["in"] if inputs is None else inputs),
        outputs=_ports(["out"] if outputs is None else outputs),
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        task_types=task_types or [],
        kind=kind,  # type: ignore[arg-type]
        source=source,
    )


def _llm_block_params(defaults: dict[str, Any]) -> list[AdvancedParameterSpec]:
    """The one parameter set every named LLM block shares.

    A family node differs from its siblings only in these values, which is the
    point: switching `ffn` from swiglu to moe genuinely turns a Mistral block
    into a Mixtral one, and the generated code follows.
    """

    def value(key: str) -> Any:
        return defaults.get(key, LLM_BLOCK_DEFAULTS[key])

    return [
        number(
            "layers", "Layers", default=value("layers"), group=STACK,
            minimum=1, maximum=256, integer=True,
            help="How many of these blocks to stack. Each gets its own weights.",
        ),
        select(
            "attention", "Attention", options=[MHA, GQA, MLA], default=value("attention"),
            group=ATTENTION,
            help=(
                "mha shares nothing; gqa gives several query heads one key/value head; "
                "mla compresses key/value through a low-rank latent, which is DeepSeek's."
            ),
        ),
        number("num_heads", "Query heads", default=value("num_heads"), group=ATTENTION, minimum=1, maximum=256, integer=True),
        number(
            "num_kv_heads", "Key/value heads", default=value("num_kv_heads"), group=ATTENTION,
            minimum=0, maximum=256, integer=True,
            help="0 matches the query head count. Ignored when attention is mla.",
        ),
        number(
            "head_dim", "Head dim", default=value("head_dim"), group=ATTENTION,
            minimum=1, maximum=4096, integer=True,
            help="Stated rather than derived: Gemma 3 runs 8 heads of 256 against a 2560-wide stream.",
        ),
        toggle(
            "qk_norm", "QK-norm", default=value("qk_norm"), group=ATTENTION,
            help="RMSNorm over each query and key head vector before the dot product. Qwen3 and Gemma 3.",
        ),
        number(
            "query_scale", "Query pre-attention scalar", default=value("query_scale"), group=ATTENTION,
            minimum=0, maximum=8192, integer=True,
            help="Divide scores by sqrt of this instead of the head dim. 0 uses the head dim. Gemma 3 states 256.",
        ),
        number(
            "rope_theta", "RoPE theta", default=value("rope_theta"), group=ATTENTION,
            minimum=0.0, maximum=10_000_000.0, step=10000.0,
            help="Rotary base, applied to queries and keys inside attention. 0 disables rotary entirely — use a Positional embedding node instead, as GPT-2 and BERT do.",
        ),
        number(
            "sliding_window", "Sliding window", default=value("sliding_window"), group=ATTENTION,
            minimum=0, maximum=1_000_000, integer=True,
            help="Each token attends only this far back. 0 is full attention.",
        ),
        number(
            "global_every", "Global layer every", default=value("global_every"), group=ATTENTION,
            minimum=0, maximum=64, integer=True,
            help="With a window set, every Nth layer attends globally instead. Gemma 3 uses 6.",
        ),
        toggle("causal", "Causal mask", default=value("causal"), group=ATTENTION),
        toggle(
            "use_bias", "Projection biases", default=value("use_bias"), group=ATTENTION,
            help="GPT-2 and BERT carry a bias on every projection. Every RoPE-era model dropped them — Llama, Qwen, Mistral, Gemma and DeepSeek all set attention_bias false.",
        ),
        number(
            "q_lora_rank", "Query LoRA rank", default=value("q_lora_rank"), group=LATENT_ATTENTION,
            minimum=0, maximum=16384, integer=True,
            help="MLA only. 0 projects queries directly; DeepSeek and Kimi both compress through 1536.",
        ),
        number(
            "kv_lora_rank", "Key/value LoRA rank", default=value("kv_lora_rank"), group=LATENT_ATTENTION,
            minimum=1, maximum=16384, integer=True,
            help="MLA only. The width of the latent the KV cache actually stores.",
        ),
        number("qk_rope_head_dim", "RoPE head dim", default=value("qk_rope_head_dim"), group=LATENT_ATTENTION, minimum=0, maximum=1024, integer=True),
        number("qk_nope_head_dim", "Non-RoPE head dim", default=value("qk_nope_head_dim"), group=LATENT_ATTENTION, minimum=1, maximum=1024, integer=True),
        number("v_head_dim", "Value head dim", default=value("v_head_dim"), group=LATENT_ATTENTION, minimum=1, maximum=1024, integer=True),
        select(
            "ffn", "Feed-forward", options=[SWIGLU, GEGLU, GELU, MOE], default=value("ffn"),
            group=FEED_FORWARD,
            help="swiglu is Llama/Qwen/Mistral; geglu is Gemma; gelu is the classic two-layer MLP; moe routes to experts.",
        ),
        number("ffn_dim", "Feed-forward dim", default=value("ffn_dim"), group=FEED_FORWARD, minimum=1, maximum=1_000_000, integer=True, help="For an MoE block this is one expert's hidden width."),
        number("num_experts", "Routed experts", default=value("num_experts"), group=EXPERTS, minimum=1, maximum=2048, integer=True),
        number("experts_per_token", "Active per token", default=value("experts_per_token"), group=EXPERTS, minimum=1, maximum=64, integer=True),
        number(
            "shared_experts", "Shared experts", default=value("shared_experts"), group=EXPERTS,
            minimum=0, maximum=16, integer=True,
            help="Always-on, unrouted. DeepSeek and Kimi both use 1, so the routed experts can specialize.",
        ),
        number(
            "dense_layers", "Leading dense layers", default=value("dense_layers"), group=EXPERTS,
            minimum=0, maximum=64, integer=True,
            help="The first N layers keep a plain feed-forward instead of a router — DeepSeek's first_k_dense_replace.",
        ),
        number("dense_ffn_dim", "Dense feed-forward dim", default=value("dense_ffn_dim"), group=EXPERTS, minimum=1, maximum=1_000_000, integer=True, help="Width of the feed-forward in those leading dense layers."),
        select(
            "router", "Router", options=[ROUTER_SOFTMAX, ROUTER_SIGMOID_BIAS], default=value("router"),
            group=EXPERTS,
            help=(
                "softmax scores all experts, takes the top-k, renormalizes (Mixtral). "
                "sigmoid_bias scores each expert independently and adds a learned selection "
                "bias that never enters the gate value — DeepSeek's aux-loss-free balancing."
            ),
        ),
        number("routed_scaling", "Routed scaling factor", default=value("routed_scaling"), group=EXPERTS, minimum=0.1, maximum=16.0, step=0.001, help="Multiplies the routed experts' summed output. DeepSeek 2.5, Kimi 2.827."),
        select("norm", "Normalization", options=["rms", "layer"], default=value("norm"), group=NORM),
        select(
            "norm_placement", "Norm placement", options=[PRE, SANDWICH, POST], default=value("norm_placement"),
            group=NORM,
            help=(
                "pre normalizes before each sub-layer; sandwich also normalizes the "
                "sub-layer's output before the residual adds, which is Gemma's four norms "
                "per layer; post is the original 2017 arrangement, still used by BERT."
            ),
        ),
        number("dropout", "Dropout", default=value("dropout"), group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
    ]


def _llm_block_spec(family: BlockFamily) -> NodeSpec:
    return _spec(
        family.type,
        family.name,
        LLM_BLOCK,
        family.description,
        params=_llm_block_params(family.defaults),
        kind="block",
        source=family.source,
        task_types=["language_modeling", "text_classification"],
    )


_VISION_BLOCK_PARAMS: dict[str, list[AdvancedParameterSpec]] = {
    "resnet_block": [
        select("variant", "Variant", options=["basic", "bottleneck"], default="basic", group=LAYER, help="basic is two 3×3s (ResNet-18/34); bottleneck is 1×1 → 3×3 → 1×1 with a 4× expansion (ResNet-50 and up)."),
        number("filters", "Base filters", default=64, group=LAYER, minimum=1, maximum=4096, integer=True),
        number("blocks", "Blocks", default=2, group=STACK, minimum=1, maximum=64, integer=True),
        number("stride", "First-block stride", default=1, group=LAYER, minimum=1, maximum=4, integer=True, help="2 halves the feature map at the start of the stage, which is how ResNet changes resolution."),
        number("expansion", "Bottleneck expansion", default=4, group=LAYER, minimum=1, maximum=8, integer=True),
    ],
    "inverted_residual_block": [
        number("filters", "Output filters", default=32, group=LAYER, minimum=1, maximum=4096, integer=True),
        number("expand_ratio", "Expansion ratio", default=6, group=LAYER, minimum=1, maximum=12, integer=True, help="1 skips the expansion convolution entirely, which is what MobileNetV2's first block does."),
        number("kernel_size", "Depthwise kernel", default=3, group=LAYER, minimum=1, maximum=9, integer=True),
        number("stride", "Stride", default=1, group=LAYER, minimum=1, maximum=4, integer=True),
        toggle("use_se", "Squeeze-excite", default=False, group=LAYER, help="On for MobileNetV3 and EfficientNet; off for MobileNetV2."),
        number("se_ratio", "Squeeze ratio", default=4, group=LAYER, minimum=1, maximum=32, integer=True),
        select("activation", "Activation", options=["relu6", "relu", "hardswish", "swish"], default="relu6", group=LAYER),
        number("blocks", "Blocks", default=1, group=STACK, minimum=1, maximum=32, integer=True),
    ],
    "dense_block": [
        number("growth_rate", "Growth rate", default=32, group=LAYER, minimum=1, maximum=512, integer=True, help="Channels each layer contributes to the running concatenation."),
        number("layers", "Layers", default=6, group=STACK, minimum=1, maximum=64, integer=True),
        number("bottleneck_ratio", "Bottleneck ratio", default=4, group=LAYER, minimum=1, maximum=8, integer=True),
    ],
    "inception_block": [
        number("filters_1x1", "1×1 path", default=64, group=LAYER, minimum=1, maximum=2048, integer=True),
        number("reduce_3x3", "3×3 reduction", default=96, group=LAYER, minimum=1, maximum=2048, integer=True),
        number("filters_3x3", "3×3 path", default=128, group=LAYER, minimum=1, maximum=2048, integer=True),
        number("reduce_5x5", "5×5 reduction", default=16, group=LAYER, minimum=1, maximum=2048, integer=True),
        number("filters_5x5", "5×5 path", default=32, group=LAYER, minimum=1, maximum=2048, integer=True),
        number("filters_pool", "Pool path", default=32, group=LAYER, minimum=1, maximum=2048, integer=True),
    ],
    "convnext_block": [
        number("filters", "Channels", default=96, group=LAYER, minimum=1, maximum=4096, integer=True),
        number("blocks", "Blocks", default=3, group=STACK, minimum=1, maximum=64, integer=True),
        number("kernel_size", "Depthwise kernel", default=7, group=LAYER, minimum=1, maximum=15, integer=True),
        number("expand_ratio", "Pointwise expansion", default=4, group=LAYER, minimum=1, maximum=8, integer=True),
        number("layer_scale", "Layer scale init", default=1e-6, group=LAYER, minimum=0.0, maximum=1.0, step=1e-6, help="Initial value of the learned per-channel scale on the residual branch. 0 disables it."),
    ],
    "vit_block": [
        number("layers", "Layers", default=12, group=STACK, minimum=1, maximum=64, integer=True),
        number("num_heads", "Heads", default=12, group=ATTENTION, minimum=1, maximum=64, integer=True),
        number("head_dim", "Head dim", default=64, group=ATTENTION, minimum=1, maximum=1024, integer=True),
        number("mlp_dim", "MLP dim", default=3072, group=FEED_FORWARD, minimum=1, maximum=65536, integer=True),
        number("dropout", "Dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
    ],
}


def _vision_block_spec(block: BlockFamily) -> NodeSpec:
    return _spec(
        block.type,
        block.name,
        VISION_BLOCK,
        block.description,
        params=_VISION_BLOCK_PARAMS[block.type],
        kind="block",
        source=block.source,
    )


# Every task whose input is text. The NLP nodes that are specific to language —
# a Text CNN, a recurrent encoder, a span head — are offered only for these, so
# an image graph's palette stays about images.
TEXT_TASKS: list[TaskType] = [
    "text_classification",
    "summarization",
    "question_answering",
    "language_modeling",
]

_NLP_BLOCK_PARAMS: dict[str, list[AdvancedParameterSpec]] = {
    "text_cnn_block": [
        number("filters", "Filters per width", default=128, group=LAYER, minimum=1, maximum=4096, integer=True, help="Feature maps learned at each kernel width. The block emits this many times the number of widths."),
        text("kernel_sizes", "Kernel widths", default="3,4,5", group=LAYER, help="Comma-separated n-gram widths read in parallel. Kim's paper uses 3, 4 and 5."),
        select("activation", "Activation", options=ACTIVATIONS, default="relu", group=LAYER),
        number("dropout", "Dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05, help="Applied to the pooled features, which is where the original applies it."),
    ],
    "bilstm_encoder": [
        select("cell", "Cell", options=["lstm", "gru"], default="lstm", group=LAYER, help="GRU is cheaper by a third of the gates and usually as accurate on short text."),
        number("units", "Units per direction", default=128, group=LAYER, minimum=1, maximum=4096, integer=True, help="The output is twice this wide: the two directions are concatenated."),
        number("layers", "Layers", default=2, group=STACK, minimum=1, maximum=16, integer=True),
        toggle("return_sequences", "Return sequences", default=True, group=LAYER, help="On to keep one vector per token, for pooling or tagging; off to emit a single document vector."),
        number("dropout", "Dropout", default=0.2, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
    ],
}


def _nlp_block_spec(block: BlockFamily) -> NodeSpec:
    return _spec(
        block.type,
        block.name,
        NLP,
        block.description,
        params=_NLP_BLOCK_PARAMS[block.type],
        kind="block",
        source=block.source,
        task_types=TEXT_TASKS,
    )


_SPEC_LIST: list[NodeSpec] = [
    *(_vision_block_spec(block) for block in VISION_BLOCKS),
    *(_llm_block_spec(family) for family in LLM_FAMILIES),
    *(_nlp_block_spec(block) for block in NLP_BLOCKS),
    # --- Input & Output ----------------------------------------------------
    _spec(
        "input",
        "Input",
        IO,
        "Where data enters the model. Shape excludes the batch dimension.",
        params=[
            text(
                "shape",
                "Input shape",
                default="224,224,3",
                group=SHAPE,
                help="Comma-separated, no batch dimension. Images are height,width,channels; token sequences are just the length.",
            )
        ],
        min_inputs=0,
        max_inputs=0,
        inputs=[],
    ),
    _spec(
        "output",
        "Output",
        IO,
        "Marks where the model ends. Emits no layer of its own.",
        max_inputs=1,
        outputs=[],
    ),
    # --- Core --------------------------------------------------------------
    _spec(
        "dense",
        "Dense",
        CORE,
        "Fully connected layer.",
        params=[
            number("units", "Units", default=64, group=LAYER, minimum=1, maximum=65536, integer=True),
            toggle(
                "units_from_dataset",
                "Units = dataset class count",
                default=False,
                group=LAYER,
                help="For the output head. Emits the runner-supplied num_classes instead of a fixed number, so the same architecture retrains on a different dataset.",
            ),
            activation_param(),
            toggle("use_bias", "Use bias", default=True, group=LAYER),
        ],
    ),
    _spec(
        "activation",
        "Activation",
        CORE,
        "Standalone activation function.",
        params=[activation_param("relu")],
    ),
    _spec(
        "flatten",
        "Flatten",
        CORE,
        "Collapses all non-batch dimensions into one.",
    ),
    _spec(
        "reshape",
        "Reshape",
        CORE,
        "Reshapes the tensor without changing its element count.",
        params=[text("target_shape", "Target shape", default="7,7,64", group=SHAPE)],
    ),
    _spec(
        "embedding",
        "Embedding",
        CORE,
        "Maps integer token ids to dense vectors.",
        params=[
            number("input_dim", "Vocabulary size", default=10000, group=LAYER, minimum=2, maximum=2_000_000, integer=True),
            number("output_dim", "Embedding size", default=128, group=LAYER, minimum=1, maximum=32768, integer=True),
            toggle("mask_zero", "Mask zero padding", default=False, group=LAYER),
        ],
        task_types=["text_classification", "summarization", "question_answering"],
    ),
    # --- Convolution -------------------------------------------------------
    _spec(
        "conv1d",
        "Conv1D",
        CONVOLUTION,
        "1D convolution over a sequence.",
        params=[
            number("filters", "Filters", default=64, group=LAYER, minimum=1, maximum=8192, integer=True),
            number("kernel_size", "Kernel size", default=3, group=LAYER, minimum=1, maximum=31, integer=True),
            number("strides", "Strides", default=1, group=LAYER, minimum=1, maximum=8, integer=True),
            select("padding", "Padding", options=PADDINGS, default="same", group=LAYER),
            activation_param("relu"),
        ],
    ),
    _spec(
        "conv2d",
        "Conv2D",
        CONVOLUTION,
        "2D convolution over an image.",
        params=[
            number("filters", "Filters", default=32, group=LAYER, minimum=1, maximum=8192, integer=True),
            number("kernel_size", "Kernel size", default=3, group=LAYER, minimum=1, maximum=31, integer=True),
            number("strides", "Strides", default=1, group=LAYER, minimum=1, maximum=8, integer=True),
            select("padding", "Padding", options=PADDINGS, default="same", group=LAYER),
            activation_param("relu"),
        ],
    ),
    _spec(
        "separable_conv2d",
        "SeparableConv2D",
        CONVOLUTION,
        "Depthwise-separable convolution — far fewer parameters than Conv2D.",
        params=[
            number("filters", "Filters", default=32, group=LAYER, minimum=1, maximum=8192, integer=True),
            number("kernel_size", "Kernel size", default=3, group=LAYER, minimum=1, maximum=31, integer=True),
            number("strides", "Strides", default=1, group=LAYER, minimum=1, maximum=8, integer=True),
            select("padding", "Padding", options=PADDINGS, default="same", group=LAYER),
            activation_param("relu"),
        ],
    ),
    _spec(
        "conv2d_transpose",
        "Conv2DTranspose",
        CONVOLUTION,
        "Learned upsampling, for decoders and segmentation heads.",
        params=[
            number("filters", "Filters", default=32, group=LAYER, minimum=1, maximum=8192, integer=True),
            number("kernel_size", "Kernel size", default=3, group=LAYER, minimum=1, maximum=31, integer=True),
            number("strides", "Strides", default=2, group=LAYER, minimum=1, maximum=8, integer=True),
            select("padding", "Padding", options=PADDINGS, default="same", group=LAYER),
            activation_param("relu"),
        ],
    ),
    _spec(
        "max_pool2d",
        "MaxPool2D",
        CONVOLUTION,
        "Downsamples by taking the maximum in each window.",
        params=[
            number("pool_size", "Pool size", default=2, group=LAYER, minimum=1, maximum=16, integer=True),
            number("strides", "Strides", default=0, group=LAYER, minimum=0, maximum=16, integer=True, help="0 matches the pool size, which is the Keras default."),
            select("padding", "Padding", options=PADDINGS, default="valid", group=LAYER),
        ],
    ),
    _spec(
        "avg_pool2d",
        "AvgPool2D",
        CONVOLUTION,
        "Downsamples by averaging each window.",
        params=[
            number("pool_size", "Pool size", default=2, group=LAYER, minimum=1, maximum=16, integer=True),
            number("strides", "Strides", default=0, group=LAYER, minimum=0, maximum=16, integer=True, help="0 matches the pool size, which is the Keras default."),
            select("padding", "Padding", options=PADDINGS, default="valid", group=LAYER),
        ],
    ),
    _spec(
        "depthwise_conv2d",
        "DepthwiseConv2D",
        CONVOLUTION,
        "One filter per input channel, with no mixing between them. The first half of a separable convolution, and the core of every mobile architecture.",
        params=[
            number("kernel_size", "Kernel size", default=3, group=LAYER, minimum=1, maximum=31, integer=True),
            number("strides", "Strides", default=1, group=LAYER, minimum=1, maximum=8, integer=True),
            select("padding", "Padding", options=PADDINGS, default="same", group=LAYER),
            number("depth_multiplier", "Depth multiplier", default=1, group=LAYER, minimum=1, maximum=16, integer=True, help="Filters per input channel. 1 keeps the channel count unchanged."),
            activation_param(),
        ],
    ),
    _spec(
        "squeeze_excite",
        "Squeeze-and-excite",
        CONVOLUTION,
        "Pools each feature map to one number, learns a gate from those, and rescales the channels. Adds almost no parameters and is in EfficientNet, MobileNetV3, and SENet.",
        params=[
            number("ratio", "Squeeze ratio", default=4, group=LAYER, minimum=1, maximum=64, integer=True, help="The bottleneck is channels ÷ ratio wide."),
            select("gate", "Gate activation", options=["sigmoid", "hardsigmoid"], default="sigmoid", group=LAYER, help="MobileNetV3 uses hardsigmoid; SENet and EfficientNet use sigmoid."),
        ],
    ),
    _spec(
        "global_avg_pool2d",
        "GlobalAvgPool2D",
        CONVOLUTION,
        "Averages each feature map to a single value. The usual bridge from convolutions to a Dense head.",
    ),
    _spec(
        "global_max_pool2d",
        "GlobalMaxPool2D",
        CONVOLUTION,
        "Takes the maximum of each feature map.",
    ),
    _spec(
        "up_sampling2d",
        "UpSampling2D",
        CONVOLUTION,
        "Repeats rows and columns. No learned parameters.",
        params=[number("size", "Scale", default=2, group=LAYER, minimum=1, maximum=8, integer=True)],
    ),
    _spec(
        "zero_padding2d",
        "ZeroPadding2D",
        CONVOLUTION,
        "Pads the spatial dimensions with zeros.",
        params=[number("padding", "Padding", default=1, group=LAYER, minimum=0, maximum=16, integer=True)],
    ),
    # --- Normalization -----------------------------------------------------
    _spec(
        "batch_norm",
        "BatchNormalization",
        NORMALIZATION,
        "Normalizes across the batch. Standard between a convolution and its activation.",
        params=[
            number(
                "momentum", "Momentum", default=0.9, group=LAYER,
                minimum=0.0, maximum=1.0, step=0.01,
                help=(
                    "How much of the old running mean/variance to keep per step. These "
                    "statistics are used at inference but not during training, so a value "
                    "too close to 1 makes a model that trains well and validates at chance. "
                    "Keras defaults to 0.99, which assumes thousands of steps per epoch; "
                    "0.9 matches PyTorch's default and converges on datasets this size."
                ),
            ),
            number("epsilon", "Epsilon", default=0.001, group=LAYER, minimum=1e-7, maximum=0.1, step=1e-4),
        ],
    ),
    _spec(
        "layer_norm",
        "LayerNormalization",
        NORMALIZATION,
        "Normalizes across features per sample. Standard in sequence models.",
        params=[
            number("epsilon", "Epsilon", default=0.001, group=LAYER, minimum=1e-7, maximum=0.1, step=1e-4)
        ],
    ),
    _spec(
        "group_norm",
        "GroupNormalization",
        NORMALIZATION,
        "Normalizes within channel groups. Batch-size independent.",
        params=[
            number("groups", "Groups", default=32, group=LAYER, minimum=1, maximum=512, integer=True),
            number("epsilon", "Epsilon", default=0.001, group=LAYER, minimum=1e-7, maximum=0.1, step=1e-4),
        ],
    ),
    # --- Recurrent ---------------------------------------------------------
    _spec(
        "lstm",
        "LSTM",
        RECURRENT,
        "Long short-term memory over a sequence.",
        params=[
            number("units", "Units", default=64, group=LAYER, minimum=1, maximum=4096, integer=True),
            toggle("return_sequences", "Return sequences", default=False, group=LAYER, help="On to stack another recurrent layer; off to emit one vector per sample."),
            toggle("bidirectional", "Bidirectional", default=False, group=LAYER, help="Wraps the layer in Bidirectional, doubling the output width."),
            number("dropout", "Dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
        ],
    ),
    _spec(
        "gru",
        "GRU",
        RECURRENT,
        "Gated recurrent unit — cheaper than LSTM, often as accurate.",
        params=[
            number("units", "Units", default=64, group=LAYER, minimum=1, maximum=4096, integer=True),
            toggle("return_sequences", "Return sequences", default=False, group=LAYER),
            toggle("bidirectional", "Bidirectional", default=False, group=LAYER),
            number("dropout", "Dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
        ],
    ),
    # --- Merge -------------------------------------------------------------
    _spec("add", "Add", MERGE, "Element-wise sum. The residual connection.", min_inputs=2, max_inputs=-1),
    _spec("multiply", "Multiply", MERGE, "Element-wise product.", min_inputs=2, max_inputs=-1),
    _spec("average", "Average", MERGE, "Element-wise mean.", min_inputs=2, max_inputs=-1),
    _spec("subtract", "Subtract", MERGE, "Element-wise difference of exactly two inputs.", min_inputs=2, max_inputs=2),
    _spec(
        "concatenate",
        "Concatenate",
        MERGE,
        "Joins inputs along one axis. All other dimensions must match.",
        params=[
            number("axis", "Axis", default=-1, group=SHAPE, minimum=-4, maximum=4, integer=True)
        ],
        min_inputs=2,
        max_inputs=-1,
    ),
    # --- Augmentation ------------------------------------------------------
    # These are layers, not a training-form toggle, because the graph is the
    # single source of truth for what the model is: a run is reproducible from
    # the canvas alone, and the augmentation travels with an export or import
    # instead of living in a hyperparameter dict beside it.
    #
    # All four are active only while training. Keras drives that off the
    # `training` flag it already threads through `fit`, so a saved model
    # evaluates and serves deterministically with no bypass wiring — the same
    # reason they are safe to leave in the graph at inference time.
    _spec(
        "random_flip",
        "RandomFlip",
        AUGMENTATION,
        "Mirrors the image at random while training. Identity at inference.",
        params=[
            select(
                "mode",
                "Mode",
                options=["horizontal", "vertical", "horizontal_and_vertical"],
                default="horizontal",
                group=LAYER,
                help="Horizontal suits most photographs; vertical rarely does.",
            )
        ],
        task_types=["classification", "object_detection", "segmentation"],
    ),
    _spec(
        "random_rotation",
        "RandomRotation",
        AUGMENTATION,
        "Rotates the image at random while training. Identity at inference.",
        params=[
            number(
                "factor",
                "Factor",
                default=0.1,
                group=LAYER,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                help="Fraction of a full turn, so 0.1 is ±36°.",
            )
        ],
        task_types=["classification", "object_detection", "segmentation"],
    ),
    _spec(
        "random_zoom",
        "RandomZoom",
        AUGMENTATION,
        "Zooms the image in or out at random while training. Identity at inference.",
        params=[
            number(
                "factor",
                "Factor",
                default=0.1,
                group=LAYER,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                help="Fraction of the height/width, so 0.1 is ±10%.",
            )
        ],
        task_types=["classification", "object_detection", "segmentation"],
    ),
    _spec(
        "random_contrast",
        "RandomContrast",
        AUGMENTATION,
        "Varies image contrast at random while training. Identity at inference.",
        params=[
            number(
                "factor",
                "Factor",
                default=0.1,
                group=LAYER,
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                help="Contrast is scaled by a factor drawn from [1-f, 1+f].",
            )
        ],
        task_types=["classification", "object_detection", "segmentation"],
    ),
    # --- Regularization ----------------------------------------------------
    _spec(
        "dropout",
        "Dropout",
        REGULARIZATION,
        "Randomly zeroes activations during training.",
        params=[
            number("rate", "Rate", default=0.2, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05)
        ],
    ),
    _spec(
        "spatial_dropout2d",
        "SpatialDropout2D",
        REGULARIZATION,
        "Drops entire feature maps rather than individual activations.",
        params=[
            number("rate", "Rate", default=0.2, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05)
        ],
    ),
    _spec(
        "gaussian_noise",
        "GaussianNoise",
        REGULARIZATION,
        "Adds zero-centred Gaussian noise during training.",
        params=[
            number("stddev", "Std deviation", default=0.1, group=REGULARIZATION_GROUP, minimum=0.0, maximum=5.0, step=0.05)
        ],
    ),
    _spec(
        "activity_regularization",
        "ActivityRegularization",
        REGULARIZATION,
        "Penalizes large activations.",
        params=[
            number("l1", "L1", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=1.0, step=0.001),
            number("l2", "L2", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=1.0, step=0.001),
        ],
    ),
    # --- Backbone ----------------------------------------------------------
    # --- Transformer -------------------------------------------------------
    _spec(
        "patch_embedding",
        "Patch embedding",
        TRANSFORMER,
        "Cuts an image into non-overlapping squares and projects each to a vector, turning (H, W, C) into a token sequence. This is how a Vision Transformer reads pixels.",
        params=[
            number("patch_size", "Patch size", default=16, group=LAYER, minimum=1, maximum=64, integer=True),
            number("embed_dim", "Embedding size", default=768, group=LAYER, minimum=1, maximum=16384, integer=True),
            toggle("class_token", "Prepend class token", default=False, group=LAYER, help="Adds one learned token whose final state is the image representation. ViT classifies from it; later models pool instead."),
        ],
    ),
    _spec(
        "geglu",
        "GeGLU feed-forward",
        TRANSFORMER,
        "Gated feed-forward with a GELU gate rather than SwiGLU's SiLU. Gemma's MLP.",
        params=[
            number("hidden_dim", "Hidden dim", default=256, group=LAYER, minimum=1, maximum=1_000_000, integer=True),
            number("dropout", "Dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
        ],
    ),
    _spec(
        "mla_attention",
        "Multi-head latent attention",
        TRANSFORMER,
        "DeepSeek's attention: keys and values are compressed through a shared low-rank latent, so the cache stores one narrow vector per token instead of every head's K and V.",
        params=[
            number("num_heads", "Heads", default=16, group=ATTENTION, minimum=1, maximum=256, integer=True),
            number("q_lora_rank", "Query LoRA rank", default=0, group=LATENT_ATTENTION, minimum=0, maximum=16384, integer=True, help="0 projects queries directly from the residual stream. DeepSeek compresses through 1536."),
            number("kv_lora_rank", "Key/value LoRA rank", default=512, group=LATENT_ATTENTION, minimum=1, maximum=16384, integer=True, help="The width of the latent the KV cache holds — the whole point of the design."),
            number("qk_rope_head_dim", "RoPE head dim", default=64, group=LATENT_ATTENTION, minimum=0, maximum=1024, integer=True, help="The part of each head that carries rotary position. Shared across heads on the key side."),
            number("qk_nope_head_dim", "Non-RoPE head dim", default=128, group=LATENT_ATTENTION, minimum=1, maximum=1024, integer=True),
            number("v_head_dim", "Value head dim", default=128, group=LATENT_ATTENTION, minimum=1, maximum=1024, integer=True),
            number("rope_theta", "RoPE theta", default=10000.0, group=ATTENTION, minimum=0.0, maximum=10_000_000.0, step=10000.0),
            toggle("causal", "Causal mask", default=True, group=ATTENTION),
            number("dropout", "Attention dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
        ],
    ),
    _spec(
        "positional_embedding",
        "Positional embedding",
        TRANSFORMER,
        "Learned position vectors added to token embeddings.",
        params=[
            number("max_length", "Max sequence length", default=256, group=LAYER, minimum=1, maximum=32768, integer=True)
        ],
    ),
    _spec(
        "rotary_embedding",
        "Rotary embedding (RoPE)",
        TRANSFORMER,
        "Rotates query/key features by position. The positional scheme used by most modern LLMs.",
        params=[
            number("base", "Theta base", default=10000.0, group=LAYER, minimum=100.0, maximum=1_000_000.0, step=1000.0)
        ],
    ),
    _spec(
        "multi_head_attention",
        "Multi-head attention",
        TRANSFORMER,
        "Self-attention over a sequence. Turn on Causal for a decoder-only language model.",
        params=[
            number("num_heads", "Heads", default=4, group=LAYER, minimum=1, maximum=256, integer=True),
            number("key_dim", "Key dim per head", default=32, group=LAYER, minimum=1, maximum=1024, integer=True),
            toggle("causal", "Causal mask", default=True, group=LAYER, help="Each position may only attend to earlier ones — required for next-token prediction."),
            number("dropout", "Attention dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
        ],
    ),
    _spec(
        "rms_norm",
        "RMSNorm",
        TRANSFORMER,
        "Root-mean-square normalization. Cheaper than LayerNorm and the modern default.",
        params=[
            number("epsilon", "Epsilon", default=1e-6, group=LAYER, minimum=1e-8, maximum=0.1, step=1e-6)
        ],
    ),
    _spec(
        "swiglu",
        "SwiGLU feed-forward",
        TRANSFORMER,
        "Gated feed-forward block. The MLP half of a modern transformer layer.",
        params=[
            number("hidden_dim", "Hidden dim", default=256, group=LAYER, minimum=1, maximum=65536, integer=True),
            number("dropout", "Dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
        ],
    ),
    _spec(
        "feed_forward",
        "Feed-forward",
        TRANSFORMER,
        "Classic two-layer MLP block with an activation between.",
        params=[
            number("hidden_dim", "Hidden dim", default=256, group=LAYER, minimum=1, maximum=65536, integer=True),
            select("activation", "Activation", options=ACTIVATIONS, default="gelu", group=LAYER),
            number("dropout", "Dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
        ],
    ),
    _spec(
        "transformer_block",
        "Generic transformer block",
        LLM_BLOCK,
        "A complete pre-norm block — norm, attention, residual, norm, feed-forward, residual — stackable N deep. Start from a named family block instead when you want a specific model's structure.",
        kind="block",
        params=[
            number("layers", "Repeat", default=4, group=STACK, minimum=1, maximum=256, integer=True, help="Emits this many identical blocks in sequence, each with its own weights."),
            number("num_heads", "Heads", default=4, group=LAYER, minimum=1, maximum=256, integer=True),
            number("key_dim", "Key dim per head", default=32, group=LAYER, minimum=1, maximum=4096, integer=True),
            number("num_kv_heads", "Key/value heads", default=0, group=LAYER, minimum=0, maximum=256, integer=True, help="Grouped-query attention: fewer KV heads than query heads shrinks the KV cache. 0 matches the head count (standard multi-head)."),
            toggle("qk_norm", "QK-norm", default=False, group=LAYER, help="RMSNorm on the query and key head dimensions before attention. Qwen3 and Gemma 3+ use this to keep attention logits stable at depth."),
            number("sliding_window", "Sliding window", default=0, group=LAYER, minimum=0, maximum=1_000_000, integer=True, help="Each token attends only this far back. 0 is full attention. Gemma 4 uses 512."),
            number("global_every", "Global layer every", default=0, group=LAYER, minimum=0, maximum=64, integer=True, help="With a sliding window set, every Nth layer uses full attention instead. Gemma 4 alternates 5 local to 1 global."),
            number("ffn_dim", "Feed-forward dim", default=256, group=LAYER, minimum=1, maximum=1_000_000, integer=True),
            select("norm", "Normalization", options=["rms", "layer"], default="rms", group=LAYER),
            select("ffn", "Feed-forward kind", options=["swiglu", "gelu", "moe"], default="swiglu", group=LAYER, help="moe routes each token to a few of many experts — the sparse design behind Mixtral and DeepSeek."),
            number("num_experts", "Experts (MoE)", default=8, group=LAYER, minimum=1, maximum=1024, integer=True, help="Only used when the feed-forward kind is moe."),
            number("experts_per_token", "Active experts (MoE)", default=2, group=LAYER, minimum=1, maximum=64, integer=True, help="How many experts each token is routed to. Active parameters scale with this, total parameters with the expert count."),
            number("shared_experts", "Shared experts (MoE)", default=0, group=LAYER, minimum=0, maximum=16, integer=True, help="Always-on experts every token passes through, alongside the routed ones. DeepSeek V3 uses 1 shared plus 256 routed."),
            toggle("causal", "Causal mask", default=True, group=LAYER),
            number("dropout", "Dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
        ],
    ),
    _spec(
        "moe_feed_forward",
        "Mixture of experts",
        TRANSFORMER,
        "Routes each token to a few of many expert feed-forwards. Huge total capacity, small active cost.",
        params=[
            number("num_experts", "Experts", default=8, group=LAYER, minimum=1, maximum=1024, integer=True),
            number("experts_per_token", "Active per token", default=2, group=LAYER, minimum=1, maximum=64, integer=True),
            number("shared_experts", "Shared experts", default=0, group=LAYER, minimum=0, maximum=16, integer=True, help="Always-on experts every token passes through. DeepSeek V3 uses 1."),
            number("hidden_dim", "Expert hidden dim", default=256, group=LAYER, minimum=1, maximum=1_000_000, integer=True),
            number("dropout", "Dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
        ],
    ),
    _spec(
        "grouped_query_attention",
        "Grouped-query attention",
        TRANSFORMER,
        "Attention with fewer key/value heads than query heads — the memory trick every recent LLM uses.",
        params=[
            number("num_heads", "Query heads", default=8, group=LAYER, minimum=1, maximum=256, integer=True),
            number("num_kv_heads", "Key/value heads", default=2, group=LAYER, minimum=1, maximum=256, integer=True),
            number("key_dim", "Key dim per head", default=32, group=LAYER, minimum=1, maximum=4096, integer=True),
            toggle("qk_norm", "QK-norm", default=False, group=LAYER, help="RMSNorm on the query and key head dimensions before attention."),
            number("sliding_window", "Sliding window", default=0, group=LAYER, minimum=0, maximum=1_000_000, integer=True, help="Each token attends only this far back. 0 is full attention."),
            toggle("causal", "Causal mask", default=True, group=LAYER),
            number("dropout", "Attention dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
        ],
    ),
    _spec(
        "lm_head",
        "LM head",
        TRANSFORMER,
        "Projects each position to vocabulary logits for next-token prediction.",
        params=[
            number("vocab_size", "Vocabulary size", default=1000, group=LAYER, minimum=2, maximum=2_000_000, integer=True),
            toggle("vocab_from_dataset", "Vocab = dataset vocabulary", default=True, group=LAYER, help="Emits the runner-supplied vocabulary size, so the same graph retrains on a different corpus."),
            toggle("tie_embeddings", "Tie to input embedding", default=False, group=LAYER, help="Reuse the embedding matrix as the output projection instead of learning a second one. Gemma and small Qwen models do this; it removes vocab x width parameters."),
            number("logit_softcap", "Logit softcap", default=0.0, group=LAYER, minimum=0.0, maximum=1000.0, step=1.0, help="Scale logits by tanh(logit / cap) * cap to keep them bounded. Gemma applies this; 0 disables."),
        ],
    ),
    _spec(
        "global_avg_pool1d",
        "GlobalAvgPool1D",
        TRANSFORMER,
        "Averages a sequence into one vector. The bridge from a transformer to a classifier head.",
    ),
    # --- NLP ---------------------------------------------------------------
    # Sequence operations, grouped here because text is where they are reached
    # for. Only the ones that are specific to language declare `task_types`;
    # pooling, cross-attention and a fixed position signal are as useful to a
    # Vision Transformer, so they stay offered everywhere.
    _spec(
        "sinusoidal_position_encoding",
        "Sinusoidal position encoding",
        NLP,
        "The original transformer's fixed sine/cosine position signal. Carries no weights, so it costs nothing and still works past the longest sequence it was trained on — unlike a learned Positional embedding, which stops at its table.",
        params=[
            number("base", "Theta base", default=10000.0, group=LAYER, minimum=100.0, maximum=1_000_000.0, step=1000.0, help="Wavelength scale. 10000 is Vaswani et al.'s; larger values stretch the signal over longer contexts."),
        ],
    ),
    _spec(
        "cross_attention",
        "Cross-attention",
        NLP,
        "Attention from one sequence to another: the first wire is the query — the sequence being written — and the second is the context it reads. The link between an encoder and a decoder, and how a summarizer or a retrieval-augmented head consults its source.",
        params=[
            number("num_heads", "Heads", default=8, group=ATTENTION, minimum=1, maximum=256, integer=True),
            number("key_dim", "Key dim per head", default=64, group=ATTENTION, minimum=1, maximum=1024, integer=True),
            number("dropout", "Attention dropout", default=0.0, group=REGULARIZATION_GROUP, minimum=0.0, maximum=0.9, step=0.05),
        ],
        min_inputs=2,
        max_inputs=2,
    ),
    _spec(
        "sequence_pool",
        "Sequence pooling",
        NLP,
        "Collapses a token sequence to one vector, which is what a classifier head needs. cls takes the first position the way BERT does, mean and max pool every position, and attention learns which tokens matter.",
        params=[
            select(
                "mode",
                "Mode",
                options=["mean", "cls", "max", "attention"],
                default="mean",
                group=LAYER,
                help="cls only means something when position 0 is a class token or a sentence marker; otherwise it reads the first real token.",
            ),
            number(
                "hidden_dim",
                "Attention hidden dim",
                default=128,
                group=LAYER,
                minimum=1,
                maximum=8192,
                integer=True,
                help="Width of the scoring layer. Attention mode only; the other modes carry no weights.",
            ),
        ],
    ),
    _spec(
        "span_head",
        "Span head",
        NLP,
        "Projects every token to a start and an end logit — the extractive question-answering head. Softmax each column over the sequence to get the answer's boundaries.",
        task_types=["question_answering"],
    ),
    # --- Custom ------------------------------------------------------------
    _spec(
        "custom_layer",
        "Custom layer",
        CUSTOM,
        "Your own Keras layer, written in Python. Runs with full permissions in the training subprocess.",
        params=[
            code(
                "class_name",
                "Class name",
                default="MyLayer",
                group=CODE,
                help="Must match the class defined below.",
            ),
            code(
                "source",
                "Python source",
                default=CUSTOM_LAYER_TEMPLATE,
                group=CODE,
                help="A tf.keras.layers.Layer subclass. Hoisted above build_model() in the generated module.",
            ),
            text(
                "output_shape",
                "Output shape",
                default="",
                group=SHAPE,
                help="Optional, comma-separated and excluding batch. Tells the canvas what this layer emits; leave blank if it does not change the shape.",
            ),
        ],
    ),
    _spec(
        "custom_function",
        "Custom function",
        CUSTOM,
        "A one-expression transform over the incoming tensor `x`, wrapped in a Lambda.",
        params=[
            code(
                "expression",
                "Expression",
                default="tf.nn.gelu(x)",
                group=CODE,
                help="Any expression over `x` and `tf`. Evaluated inside a Lambda layer.",
            ),
            text(
                "output_shape",
                "Output shape",
                default="",
                group=SHAPE,
                help="Optional, comma-separated and excluding batch. Leave blank when the shape is unchanged.",
            ),
        ],
    ),
    _spec(
        "pretrained_backbone",
        "Pretrained backbone",
        BACKBONE,
        "A tf.keras.applications backbone with its classifier head removed. Start here for transfer learning.",
        params=[
            select("application", "Application", options=BACKBONE_APPLICATIONS, default="EfficientNetB0", group=LAYER),
            select("weights", "Weights", options=["imagenet", "none"], default="imagenet", group=LAYER),
            select("pooling", "Pooling", options=["avg", "max", "none"], default="avg", group=LAYER, help="avg or max reduce the feature map to a vector; none keeps it spatial."),
            toggle("trainable", "Trainable", default=False, group=LAYER, help="Off freezes the backbone so only layers after it train."),
        ],
    ),
]

NODE_SPECS: dict[str, NodeSpec] = {spec.type: spec for spec in _SPEC_LIST}


def node_spec(node_type: str) -> NodeSpec | None:
    return NODE_SPECS.get(node_type)


def node_catalog(task_type: str | None = None) -> list[NodeSpec]:
    """Palette entries, ordered by category then declaration.

    A node with an empty `task_types` is universal; one that names task types
    is only offered for those. Filtering is a hint for the palette, not a
    validation rule — a saved graph is never rejected for using a node the
    current task type would not have offered.
    """

    specs = list(_SPEC_LIST)
    if task_type:
        specs = [
            spec for spec in specs if not spec.task_types or task_type in spec.task_types
        ]
    return sorted(specs, key=lambda spec: CATEGORY_ORDER.index(spec.category))


def default_params(node_type: str) -> dict[str, Any]:
    """The params a freshly dropped node starts with."""

    spec = NODE_SPECS.get(node_type)
    if spec is None:
        return {}
    return {param.key: param.default for param in spec.params}
