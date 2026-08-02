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

from app.ml.common.advanced import code, number, select, toggle
from app.ml.vision.keras_classification.catalog import KERAS_APPLICATION_OPTIONS
from app.schemas import AdvancedParameterSpec, NodePortSpec, NodeSpec, TaskType

# Palette categories, in display order.
IO = "Input & Output"
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
    )


_SPEC_LIST: list[NodeSpec] = [
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
            number("momentum", "Momentum", default=0.99, group=LAYER, minimum=0.0, maximum=1.0, step=0.01),
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
        "Transformer block",
        TRANSFORMER,
        "A complete pre-norm block — norm, attention, residual, norm, feed-forward, residual — stackable N deep.",
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
