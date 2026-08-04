"""Starter graphs.

An empty canvas is a bad first experience for a node editor — the useful
question is "what does a working one look like?". Each template is a complete,
compilable graph the user can train as-is or take apart.

Positions come from `auto_layout`, so a template is declared as structure only
and never carries hand-tuned coordinates that would drift as it is edited.
"""

from __future__ import annotations

from typing import Any

from app.ml.architecture.catalog import default_params
from app.ml.architecture.layout import auto_layout
from app.ml.architecture.llm_presets import PRESETS, preset_graph
from app.schemas import (
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureNode,
    ArchitectureTemplate,
    TaskType,
)


def node(node_id: str, node_type: str, **params: Any) -> ArchitectureNode:
    """A node carrying its catalog defaults with `params` applied on top.

    Templates store the full resolved param set rather than only overrides so
    the inspector shows real values the moment a node is selected.
    """

    return ArchitectureNode(id=node_id, type=node_type, params={**default_params(node_type), **params})


def chain(*node_ids: str) -> list[ArchitectureEdge]:
    return [
        ArchitectureEdge(id=f"e_{source}_{target}", source=source, target=target)
        for source, target in zip(node_ids, node_ids[1:], strict=False)
    ]


def edge(source: str, target: str) -> ArchitectureEdge:
    return ArchitectureEdge(id=f"e_{source}_{target}", source=source, target=target)


def _graph(
    nodes: list[ArchitectureNode],
    edges: list[ArchitectureEdge],
    training_defaults: dict[str, Any] | None = None,
) -> ArchitectureGraph:
    return auto_layout(
        ArchitectureGraph(
            nodes=nodes, edges=edges, training_defaults=training_defaults or {}
        )
    )


def _small_cnn() -> ArchitectureGraph:
    return _graph(
        [
            node("input", "input", shape="64,64,3"),
            node("conv_1", "conv2d", filters=32, kernel_size=3, activation="linear"),
            node("bn_1", "batch_norm"),
            node("act_1", "activation", activation="relu"),
            node("pool_1", "max_pool2d", pool_size=2),
            node("conv_2", "conv2d", filters=64, kernel_size=3, activation="linear"),
            node("bn_2", "batch_norm"),
            node("act_2", "activation", activation="relu"),
            node("pool_2", "max_pool2d", pool_size=2),
            node("gap", "global_avg_pool2d"),
            node("drop", "dropout", rate=0.3),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain(
            "input", "conv_1", "bn_1", "act_1", "pool_1",
            "conv_2", "bn_2", "act_2", "pool_2",
            "gap", "drop", "head", "output",
        ),
        {"epochs": 15, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _residual_block() -> ArchitectureGraph:
    """A skip connection — the smallest graph that is not a straight line."""

    return _graph(
        [
            node("input", "input", shape="64,64,3"),
            node("stem", "conv2d", filters=32, kernel_size=3, activation="relu"),
            node("conv_1", "conv2d", filters=32, kernel_size=3, activation="linear"),
            node("bn_1", "batch_norm"),
            node("act_1", "activation", activation="relu"),
            node("conv_2", "conv2d", filters=32, kernel_size=3, activation="linear"),
            node("bn_2", "batch_norm"),
            node("residual", "add"),
            node("act_2", "activation", activation="relu"),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        [
            *chain("input", "stem", "conv_1", "bn_1", "act_1", "conv_2", "bn_2", "residual"),
            # The skip: the block's input bypasses both convolutions.
            edge("stem", "residual"),
            *chain("residual", "act_2", "gap", "head", "output"),
        ],
        {"epochs": 20, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _transfer_learning() -> ArchitectureGraph:
    return _graph(
        [
            node("input", "input", shape="224,224,3"),
            node("backbone", "pretrained_backbone", application="EfficientNetB0", pooling="avg", trainable=False),
            node("drop", "dropout", rate=0.2),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain("input", "backbone", "drop", "head", "output"),
        {"epochs": 10, "batch_size": 16, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _bilstm_text() -> ArchitectureGraph:
    return _graph(
        [
            node("input", "input", shape="200"),
            node("embed", "embedding", input_dim=20000, output_dim=128, mask_zero=True),
            node("lstm", "lstm", units=64, bidirectional=True, return_sequences=False, dropout=0.2),
            node("drop", "dropout", rate=0.3),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain("input", "embed", "lstm", "drop", "head", "output"),
        {"epochs": 10, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )



def _mini_gpt() -> ArchitectureGraph:
    """Decoder-only transformer — the shape every modern LLM shares."""

    return _graph(
        [
            node("input", "input", shape="128"),
            node("embed", "embedding", input_dim=2000, output_dim=128, mask_zero=False),
            node("pos", "positional_embedding", max_length=128),
            node("blocks", "transformer_block", layers=4, num_heads=4, key_dim=32, ffn_dim=256, norm="rms", ffn="swiglu", causal=True, dropout=0.1),
            node("final_norm", "rms_norm"),
            node("head", "lm_head", vocab_from_dataset=True),
            node("output", "output"),
        ],
        chain("input", "embed", "pos", "blocks", "final_norm", "head", "output"),
        {"epochs": 20, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adamw"},
    )


def _transformer_from_primitives() -> ArchitectureGraph:
    """One transformer layer wired by hand, so every piece is editable.

    The same computation as a single `transformer_block`, opened up: norm →
    attention → residual → norm → SwiGLU → residual. Copy the middle to go
    deeper, or swap any node for a custom one.
    """

    return _graph(
        [
            node("input", "input", shape="128"),
            node("embed", "embedding", input_dim=2000, output_dim=128),
            node("rope", "rotary_embedding", base=10000.0),
            node("attn_norm", "rms_norm"),
            node("attn", "multi_head_attention", num_heads=4, key_dim=32, causal=True, dropout=0.1),
            node("attn_residual", "add"),
            node("ffn_norm", "rms_norm"),
            node("ffn", "swiglu", hidden_dim=256, dropout=0.1),
            node("ffn_residual", "add"),
            node("final_norm", "rms_norm"),
            node("head", "lm_head", vocab_from_dataset=True),
            node("output", "output"),
        ],
        [
            *chain("input", "embed", "rope", "attn_norm", "attn", "attn_residual"),
            # The attention residual: the pre-norm input bypasses attention.
            edge("rope", "attn_residual"),
            *chain("attn_residual", "ffn_norm", "ffn", "ffn_residual"),
            edge("attn_residual", "ffn_residual"),
            *chain("ffn_residual", "final_norm", "head", "output"),
        ],
        {"epochs": 20, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adamw"},
    )


def _transformer_text_classifier() -> ArchitectureGraph:
    """Encoder-style transformer pooled into a classifier head."""

    return _graph(
        [
            node("input", "input", shape="128"),
            node("embed", "embedding", input_dim=20000, output_dim=128),
            node("pos", "positional_embedding", max_length=128),
            node("blocks", "transformer_block", layers=2, num_heads=4, key_dim=32, ffn_dim=256, causal=False, dropout=0.1),
            node("pool", "global_avg_pool1d"),
            node("drop", "dropout", rate=0.2),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain("input", "embed", "pos", "blocks", "pool", "drop", "head", "output"),
        {"epochs": 15, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adamw"},
    )


def _custom_layer_demo() -> ArchitectureGraph:
    """Shows the escape hatch: a hand-written Keras layer inside a real graph."""

    return _graph(
        [
            node("input", "input", shape="64,64,3"),
            node("conv", "conv2d", filters=32, kernel_size=3, activation="relu"),
            node(
                "custom",
                "custom_layer",
                class_name="ChannelGate",
                output_shape="",
                source=(
                    "class ChannelGate(tf.keras.layers.Layer):\n"
                    '    """Squeeze-and-excite: rescale channels by their global importance."""\n'
                    "\n"
                    "    def __init__(self, ratio=4, **kwargs):\n"
                    "        super().__init__(**kwargs)\n"
                    "        self.ratio = ratio\n"
                    "\n"
                    "    def build(self, input_shape):\n"
                    "        channels = int(input_shape[-1])\n"
                    "        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()\n"
                    "        self.down = tf.keras.layers.Dense(max(1, channels // self.ratio), activation='relu')\n"
                    "        self.up = tf.keras.layers.Dense(channels, activation='sigmoid')\n"
                    "        super().build(input_shape)\n"
                    "\n"
                    "    def call(self, inputs):\n"
                    "        weights = self.up(self.down(self.squeeze(inputs)))\n"
                    "        return inputs * weights[:, None, None, :]\n"
                    "\n"
                    "    def get_config(self):\n"
                    "        return {**super().get_config(), 'ratio': self.ratio}\n"
                ),
            ),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain("input", "conv", "custom", "gap", "head", "output"),
        {"epochs": 15, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _mlp() -> ArchitectureGraph:
    """The simplest thing that trains — a good first canvas to take apart."""

    return _graph(
        [
            node("input", "input", shape="64,64,3"),
            node("flatten", "flatten"),
            node("hidden_1", "dense", units=256, activation="relu"),
            node("drop_1", "dropout", rate=0.3),
            node("hidden_2", "dense", units=128, activation="relu"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain("input", "flatten", "hidden_1", "drop_1", "hidden_2", "head", "output"),
        {"epochs": 20, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _inception_module() -> ArchitectureGraph:
    """Parallel branches at different kernel sizes, concatenated.

    The clearest demonstration of why a canvas beats a list: this shape is
    awkward to describe in sequence and obvious as a picture.
    """

    return _graph(
        [
            node("input", "input", shape="64,64,3"),
            node("stem", "conv2d", filters=32, kernel_size=3, activation="relu"),
            node("branch_1x1", "conv2d", filters=16, kernel_size=1, activation="relu"),
            node("branch_3x3", "conv2d", filters=16, kernel_size=3, activation="relu"),
            node("branch_5x5", "conv2d", filters=16, kernel_size=5, activation="relu"),
            node("branch_pool", "avg_pool2d", pool_size=3, strides=1, padding="same"),
            node("merge", "concatenate", axis=-1),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        [
            *chain("input", "stem"),
            edge("stem", "branch_1x1"),
            edge("stem", "branch_3x3"),
            edge("stem", "branch_5x5"),
            edge("stem", "branch_pool"),
            edge("branch_1x1", "merge"),
            edge("branch_3x3", "merge"),
            edge("branch_5x5", "merge"),
            edge("branch_pool", "merge"),
            *chain("merge", "gap", "head", "output"),
        ],
        {"epochs": 20, "batch_size": 16, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _text_cnn() -> ArchitectureGraph:
    """1D convolutions over embeddings — fast, strong text-classification baseline."""

    return _graph(
        [
            node("input", "input", shape="200"),
            node("embed", "embedding", input_dim=20000, output_dim=128),
            node("conv", "conv1d", filters=128, kernel_size=5, activation="relu"),
            node("pool", "global_avg_pool1d"),
            node("drop", "dropout", rate=0.3),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain("input", "embed", "conv", "pool", "drop", "head", "output"),
        {"epochs": 12, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _kim_text_cnn() -> ArchitectureGraph:
    """The published sentence classifier: three n-gram widths read in parallel."""

    return _graph(
        [
            node("input", "input", shape="200"),
            node("embed", "embedding", input_dim=20000, output_dim=128),
            node("cnn", "text_cnn_block", filters=128, kernel_sizes="3,4,5", activation="relu", dropout=0.5),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain("input", "embed", "cnn", "head", "output"),
        {"epochs": 12, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _bilstm_attention_text() -> ArchitectureGraph:
    """A recurrent encoder that keeps its sequence, pooled by learned attention.

    The contrast with `bilstm_text` is the point: that one throws the sequence
    away at the last layer, this one weighs every token and lets the pooling
    decide which mattered.
    """

    return _graph(
        [
            node("input", "input", shape="200"),
            node("embed", "embedding", input_dim=20000, output_dim=128, mask_zero=True),
            node("encoder", "bilstm_encoder", cell="lstm", units=128, layers=2, dropout=0.2, return_sequences=True),
            node("pool", "sequence_pool", mode="attention", hidden_dim=128),
            node("drop", "dropout", rate=0.3),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain("input", "embed", "encoder", "pool", "drop", "head", "output"),
        {"epochs": 12, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _resnet18() -> ArchitectureGraph:
    """The four-stage ResNet-18: 7×7 stem, then 64/128/256/512 at halving resolution."""

    return _graph(
        [
            node("input", "input", shape="224,224,3"),
            node("stem", "conv2d", filters=64, kernel_size=7, strides=2, activation="relu"),
            node("stem_pool", "max_pool2d", pool_size=3, strides=2, padding="same"),
            node("stage_1", "resnet_block", variant="basic", filters=64, blocks=2, stride=1),
            node("stage_2", "resnet_block", variant="basic", filters=128, blocks=2, stride=2),
            node("stage_3", "resnet_block", variant="basic", filters=256, blocks=2, stride=2),
            node("stage_4", "resnet_block", variant="basic", filters=512, blocks=2, stride=2),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain(
            "input", "stem", "stem_pool",
            "stage_1", "stage_2", "stage_3", "stage_4",
            "gap", "head", "output",
        ),
        {"epochs": 30, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _resnet50() -> ArchitectureGraph:
    """ResNet-50: the same four stages built from 1×1 → 3×3 → 1×1 bottlenecks."""

    return _graph(
        [
            node("input", "input", shape="224,224,3"),
            node("stem", "conv2d", filters=64, kernel_size=7, strides=2, activation="relu"),
            node("stem_pool", "max_pool2d", pool_size=3, strides=2, padding="same"),
            node("stage_1", "resnet_block", variant="bottleneck", filters=64, blocks=3, stride=1),
            node("stage_2", "resnet_block", variant="bottleneck", filters=128, blocks=4, stride=2),
            node("stage_3", "resnet_block", variant="bottleneck", filters=256, blocks=6, stride=2),
            node("stage_4", "resnet_block", variant="bottleneck", filters=512, blocks=3, stride=2),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain(
            "input", "stem", "stem_pool",
            "stage_1", "stage_2", "stage_3", "stage_4",
            "gap", "head", "output",
        ),
        {"epochs": 30, "batch_size": 16, "learning_rate": 0.0005, "optimizer": "adam"},
    )


def _mobilenet_v2() -> ArchitectureGraph:
    """MobileNetV2's inverted-residual stack, at its published widths."""

    return _graph(
        [
            node("input", "input", shape="224,224,3"),
            node("stem", "conv2d", filters=32, kernel_size=3, strides=2, activation="relu"),
            # The first block does not expand — there is nothing yet to expand from.
            node("block_1", "inverted_residual_block", filters=16, expand_ratio=1, stride=1, blocks=1),
            node("block_2", "inverted_residual_block", filters=24, expand_ratio=6, stride=2, blocks=2),
            node("block_3", "inverted_residual_block", filters=32, expand_ratio=6, stride=2, blocks=3),
            node("block_4", "inverted_residual_block", filters=64, expand_ratio=6, stride=2, blocks=4),
            node("block_5", "inverted_residual_block", filters=96, expand_ratio=6, stride=1, blocks=3),
            node("block_6", "inverted_residual_block", filters=160, expand_ratio=6, stride=2, blocks=3),
            node("block_7", "inverted_residual_block", filters=320, expand_ratio=6, stride=1, blocks=1),
            node("last_conv", "conv2d", filters=1280, kernel_size=1, activation="relu"),
            node("gap", "global_avg_pool2d"),
            node("drop", "dropout", rate=0.2),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain(
            "input", "stem",
            "block_1", "block_2", "block_3", "block_4", "block_5", "block_6", "block_7",
            "last_conv", "gap", "drop", "head", "output",
        ),
        {"epochs": 25, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _mobilenet_v3() -> ArchitectureGraph:
    """MobileNetV3's additions on top of V2: squeeze-excite and hard-swish."""

    return _graph(
        [
            node("input", "input", shape="224,224,3"),
            node("stem", "conv2d", filters=16, kernel_size=3, strides=2, activation="relu"),
            node("block_1", "inverted_residual_block", filters=16, expand_ratio=1, stride=1, activation="relu"),
            node("block_2", "inverted_residual_block", filters=24, expand_ratio=4, stride=2, blocks=2, activation="relu"),
            node("block_3", "inverted_residual_block", filters=40, expand_ratio=3, kernel_size=5, stride=2, blocks=3, use_se=True, activation="relu"),
            node("block_4", "inverted_residual_block", filters=80, expand_ratio=6, stride=2, blocks=4, activation="hardswish"),
            node("block_5", "inverted_residual_block", filters=112, expand_ratio=6, stride=1, blocks=2, use_se=True, activation="hardswish"),
            node("block_6", "inverted_residual_block", filters=160, expand_ratio=6, kernel_size=5, stride=2, blocks=3, use_se=True, activation="hardswish"),
            node("last_conv", "conv2d", filters=960, kernel_size=1, activation="swish"),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain(
            "input", "stem",
            "block_1", "block_2", "block_3", "block_4", "block_5", "block_6",
            "last_conv", "gap", "head", "output",
        ),
        {"epochs": 25, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _densenet() -> ArchitectureGraph:
    """DenseNet-121's four dense blocks, with the compressing transitions between."""

    return _graph(
        [
            node("input", "input", shape="224,224,3"),
            node("stem", "conv2d", filters=64, kernel_size=7, strides=2, activation="relu"),
            node("stem_pool", "max_pool2d", pool_size=3, strides=2, padding="same"),
            node("dense_1", "dense_block", growth_rate=32, layers=6),
            node("transition_1", "conv2d", filters=128, kernel_size=1, activation="relu"),
            node("pool_1", "avg_pool2d", pool_size=2, strides=2),
            node("dense_2", "dense_block", growth_rate=32, layers=12),
            node("transition_2", "conv2d", filters=256, kernel_size=1, activation="relu"),
            node("pool_2", "avg_pool2d", pool_size=2, strides=2),
            node("dense_3", "dense_block", growth_rate=32, layers=24),
            node("transition_3", "conv2d", filters=512, kernel_size=1, activation="relu"),
            node("pool_3", "avg_pool2d", pool_size=2, strides=2),
            node("dense_4", "dense_block", growth_rate=32, layers=16),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain(
            "input", "stem", "stem_pool",
            "dense_1", "transition_1", "pool_1",
            "dense_2", "transition_2", "pool_2",
            "dense_3", "transition_3", "pool_3",
            "dense_4", "gap", "head", "output",
        ),
        {"epochs": 30, "batch_size": 16, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _convnext() -> ArchitectureGraph:
    """ConvNeXt-Tiny: a 4×4 patchify stem and four stages at 96/192/384/768."""

    return _graph(
        [
            node("input", "input", shape="224,224,3"),
            node("stem", "conv2d", filters=96, kernel_size=4, strides=4, activation="linear"),
            node("stage_1", "convnext_block", filters=96, blocks=3),
            node("down_1", "conv2d", filters=192, kernel_size=2, strides=2, activation="linear"),
            node("stage_2", "convnext_block", filters=192, blocks=3),
            node("down_2", "conv2d", filters=384, kernel_size=2, strides=2, activation="linear"),
            node("stage_3", "convnext_block", filters=384, blocks=9),
            node("down_3", "conv2d", filters=768, kernel_size=2, strides=2, activation="linear"),
            node("stage_4", "convnext_block", filters=768, blocks=3),
            node("gap", "global_avg_pool2d"),
            node("norm", "layer_norm"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain(
            "input", "stem",
            "stage_1", "down_1", "stage_2", "down_2", "stage_3", "down_3", "stage_4",
            "gap", "norm", "head", "output",
        ),
        {"epochs": 30, "batch_size": 16, "learning_rate": 0.0004, "optimizer": "adamw"},
    )


def _googlenet() -> ArchitectureGraph:
    """GoogLeNet's inception stack, using the block node rather than hand wiring."""

    return _graph(
        [
            node("input", "input", shape="224,224,3"),
            node("stem", "conv2d", filters=64, kernel_size=7, strides=2, activation="relu"),
            node("pool_1", "max_pool2d", pool_size=3, strides=2, padding="same"),
            node("reduce", "conv2d", filters=192, kernel_size=3, activation="relu"),
            node("pool_2", "max_pool2d", pool_size=3, strides=2, padding="same"),
            node("inception_3a", "inception_block", filters_1x1=64, reduce_3x3=96, filters_3x3=128, reduce_5x5=16, filters_5x5=32, filters_pool=32),
            node("inception_3b", "inception_block", filters_1x1=128, reduce_3x3=128, filters_3x3=192, reduce_5x5=32, filters_5x5=96, filters_pool=64),
            node("pool_3", "max_pool2d", pool_size=3, strides=2, padding="same"),
            node("inception_4a", "inception_block", filters_1x1=192, reduce_3x3=96, filters_3x3=208, reduce_5x5=16, filters_5x5=48, filters_pool=64),
            node("gap", "global_avg_pool2d"),
            node("drop", "dropout", rate=0.4),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain(
            "input", "stem", "pool_1", "reduce", "pool_2",
            "inception_3a", "inception_3b", "pool_3", "inception_4a",
            "gap", "drop", "head", "output",
        ),
        {"epochs": 25, "batch_size": 16, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _vgg() -> ArchitectureGraph:
    """VGG-16's five blocks of stacked 3×3 convolutions — no residuals anywhere.

    Worth having on the canvas next to ResNet: the two differ by exactly one
    idea, and the picture makes that obvious.
    """

    nodes = [node("input", "input", shape="224,224,3")]
    order = ["input"]
    for index, (filters, count) in enumerate(
        [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)], start=1
    ):
        for step in range(count):
            name = f"conv_{index}_{step + 1}"
            nodes.append(node(name, "conv2d", filters=filters, kernel_size=3, activation="relu"))
            order.append(name)
        pool = f"pool_{index}"
        nodes.append(node(pool, "max_pool2d", pool_size=2, strides=2))
        order.append(pool)
    nodes += [
        node("flatten", "flatten"),
        node("fc_1", "dense", units=4096, activation="relu"),
        node("drop_1", "dropout", rate=0.5),
        node("fc_2", "dense", units=4096, activation="relu"),
        node("drop_2", "dropout", rate=0.5),
        node("head", "dense", units_from_dataset=True, activation="softmax"),
        node("output", "output"),
    ]
    order += ["flatten", "fc_1", "drop_1", "fc_2", "drop_2", "head", "output"]
    return _graph(
        nodes, chain(*order),
        {"epochs": 30, "batch_size": 16, "learning_rate": 0.0001, "optimizer": "adam"},
    )


def _unet() -> ArchitectureGraph:
    """U-Net: a contracting path, an expanding path, and skips across the middle.

    The clearest thing on this canvas that a list of layers cannot express —
    every skip carries the encoder's full-resolution detail past the bottleneck,
    and the picture is the explanation.
    """

    return _graph(
        [
            node("input", "input", shape="256,256,3"),
            node("enc_1", "conv2d", filters=64, kernel_size=3, activation="relu"),
            node("pool_1", "max_pool2d", pool_size=2),
            node("enc_2", "conv2d", filters=128, kernel_size=3, activation="relu"),
            node("pool_2", "max_pool2d", pool_size=2),
            node("enc_3", "conv2d", filters=256, kernel_size=3, activation="relu"),
            node("pool_3", "max_pool2d", pool_size=2),
            node("bottleneck", "conv2d", filters=512, kernel_size=3, activation="relu"),
            node("up_3", "conv2d_transpose", filters=256, kernel_size=2, strides=2, activation="linear"),
            node("skip_3", "concatenate", axis=-1),
            node("dec_3", "conv2d", filters=256, kernel_size=3, activation="relu"),
            node("up_2", "conv2d_transpose", filters=128, kernel_size=2, strides=2, activation="linear"),
            node("skip_2", "concatenate", axis=-1),
            node("dec_2", "conv2d", filters=128, kernel_size=3, activation="relu"),
            node("up_1", "conv2d_transpose", filters=64, kernel_size=2, strides=2, activation="linear"),
            node("skip_1", "concatenate", axis=-1),
            node("dec_1", "conv2d", filters=64, kernel_size=3, activation="relu"),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        [
            *chain(
                "input", "enc_1", "pool_1", "enc_2", "pool_2", "enc_3", "pool_3",
                "bottleneck", "up_3", "skip_3", "dec_3", "up_2", "skip_2", "dec_2",
                "up_1", "skip_1", "dec_1", "gap", "head", "output",
            ),
            # The skips: each encoder stage's output crosses to its mirror.
            edge("enc_3", "skip_3"),
            edge("enc_2", "skip_2"),
            edge("enc_1", "skip_1"),
        ],
        {"epochs": 25, "batch_size": 8, "learning_rate": 0.0005, "optimizer": "adam"},
    )


def _vit() -> ArchitectureGraph:
    """ViT-Base/16: patchify to a sequence, then twelve encoder layers."""

    return _graph(
        [
            node("input", "input", shape="224,224,3"),
            node("patches", "patch_embedding", patch_size=16, embed_dim=768, class_token=True),
            node("pos", "positional_embedding", max_length=197),
            node("encoder", "vit_block", layers=12, num_heads=12, head_dim=64, mlp_dim=3072, dropout=0.0),
            node("norm", "layer_norm"),
            node("pool", "global_avg_pool1d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain("input", "patches", "pos", "encoder", "norm", "pool", "head", "output"),
        {"epochs": 30, "batch_size": 16, "learning_rate": 0.0001, "optimizer": "adamw"},
    )


def _squeeze_excite_cnn() -> ArchitectureGraph:
    """A small CNN with squeeze-excite gates — SE as a node rather than a custom layer."""

    return _graph(
        [
            node("input", "input", shape="96,96,3"),
            node("conv_1", "conv2d", filters=32, kernel_size=3, strides=2, activation="relu"),
            node("se_1", "squeeze_excite", ratio=4),
            node("dw", "depthwise_conv2d", kernel_size=3, strides=2, activation="relu"),
            node("conv_2", "conv2d", filters=64, kernel_size=1, activation="relu"),
            node("se_2", "squeeze_excite", ratio=4),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain("input", "conv_1", "se_1", "dw", "conv_2", "se_2", "gap", "head", "output"),
        {"epochs": 20, "batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    )


def _bert_classifier() -> ArchitectureGraph:
    """BERT-base's encoder, pooled into a classifier head.

    Bidirectional and post-norm, which is what separates it from every decoder
    on this canvas.
    """

    return _graph(
        [
            node("input", "input", shape="128"),
            node("embed", "embedding", input_dim=30522, output_dim=768),
            node("pos", "positional_embedding", max_length=512),
            node("encoder", "bert_block", layers=12, num_heads=12, num_kv_heads=12, head_dim=64, ffn_dim=3072, dropout=0.1),
            node("pool", "global_avg_pool1d"),
            node("drop", "dropout", rate=0.1),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        chain("input", "embed", "pos", "encoder", "pool", "drop", "head", "output"),
        {"epochs": 5, "batch_size": 16, "learning_rate": 2e-5, "optimizer": "adamw"},
    )


def _moe_layer_by_hand() -> ArchitectureGraph:
    """One sparse layer opened up, so both residual paths are visible.

    The same computation a `mixtral_block` performs, wired from primitives:
    norm → attention → residual, norm → router-over-experts → residual. Useful
    for seeing where a mixture of experts actually sits in a layer, which is
    the question the compressed block node cannot answer.
    """

    return _graph(
        [
            node("input", "input", shape="128"),
            node("embed", "embedding", input_dim=8000, output_dim=256),
            node("attn_norm", "rms_norm"),
            node("attn", "grouped_query_attention", num_heads=8, num_kv_heads=2, key_dim=32, causal=True),
            node("attn_residual", "add"),
            node("moe_norm", "rms_norm"),
            node("moe", "moe_feed_forward", num_experts=8, experts_per_token=2, shared_experts=1, hidden_dim=512),
            node("moe_residual", "add"),
            node("final_norm", "rms_norm"),
            node("head", "lm_head", vocab_from_dataset=True),
            node("output", "output"),
        ],
        [
            *chain("input", "embed", "attn_norm", "attn", "attn_residual"),
            edge("embed", "attn_residual"),
            *chain("attn_residual", "moe_norm", "moe", "moe_residual"),
            edge("attn_residual", "moe_residual"),
            *chain("moe_residual", "final_norm", "head", "output"),
        ],
        {"epochs": 20, "batch_size": 16, "learning_rate": 0.0003, "optimizer": "adamw"},
    )


_TEMPLATES: list[tuple[str, str, str, TaskType, ArchitectureGraph]] = [
    (
        "small_cnn",
        "Small CNN",
        "Two convolution blocks and a global-pooled head. A sane baseline to train from scratch on a small image dataset.",
        "classification",
        _small_cnn(),
    ),
    (
        "residual_block",
        "Residual block",
        "A skip connection around two convolutions, the building block of ResNet. Shows how merge nodes wire up.",
        "classification",
        _residual_block(),
    ),
    (
        "transfer_learning",
        "Transfer learning head",
        "A frozen EfficientNetB0 backbone with a fresh classifier head. Usually the strongest option on a small dataset.",
        "classification",
        _transfer_learning(),
    ),
    (
        "mlp",
        "Multilayer perceptron",
        "Flatten and two dense layers. The simplest graph that trains — a good one to take apart first.",
        "classification",
        _mlp(),
    ),
    (
        "inception_module",
        "Inception module",
        "Four parallel branches at different kernel sizes, concatenated. Shows why a canvas beats a list.",
        "classification",
        _inception_module(),
    ),
    (
        "bilstm_text",
        "BiLSTM text classifier",
        "An embedding layer into a bidirectional LSTM. The standard starting point for text classification.",
        "text_classification",
        _bilstm_text(),
    ),
    (
        "text_cnn",
        "Text CNN",
        "1D convolutions over word embeddings. Faster than a recurrent model and often just as accurate.",
        "text_classification",
        _text_cnn(),
    ),
    (
        "kim_text_cnn",
        "Text CNN (Kim)",
        "Kernel widths 3, 4 and 5 over the embeddings at once, each max-pooled over time. The published version of the template above, in one block.",
        "text_classification",
        _kim_text_cnn(),
    ),
    (
        "bilstm_attention_text",
        "BiLSTM + attention pooling",
        "A two-layer bidirectional encoder that keeps its sequence, pooled by a learned attention score instead of taking the last step.",
        "text_classification",
        _bilstm_attention_text(),
    ),
    (
        "transformer_text_classifier",
        "Transformer classifier",
        "A two-layer encoder pooled into a classifier head. Attention without the causal mask.",
        "text_classification",
        _transformer_text_classifier(),
    ),
    (
        "custom_layer_demo",
        "Custom layer (squeeze-excite)",
        "A hand-written Keras layer inside a working CNN. The escape hatch for anything the palette lacks.",
        "classification",
        _custom_layer_demo(),
    ),
    (
        "resnet18",
        "ResNet-18",
        "A 7×7 stem into four residual stages at 64/128/256/512. The reference image classifier, and the shortest path to a strong from-scratch baseline.",
        "classification",
        _resnet18(),
    ),
    (
        "resnet50",
        "ResNet-50",
        "The same four stages built from 1×1 → 3×3 → 1×1 bottlenecks with a 4× expansion. Deeper than ResNet-18 at a similar cost.",
        "classification",
        _resnet50(),
    ),
    (
        "vgg16",
        "VGG-16",
        "Five blocks of stacked 3×3 convolutions and three dense layers. No residuals anywhere — worth opening next to ResNet to see what one idea changed.",
        "classification",
        _vgg(),
    ),
    (
        "googlenet",
        "GoogLeNet (Inception v1)",
        "Inception modules stacked into a full network. Four receptive fields in parallel at every stage.",
        "classification",
        _googlenet(),
    ),
    (
        "mobilenet_v2",
        "MobileNetV2",
        "Inverted residuals with linear bottlenecks, at the published widths. The standard mobile-scale classifier.",
        "classification",
        _mobilenet_v2(),
    ),
    (
        "mobilenet_v3",
        "MobileNetV3",
        "MobileNetV2 plus squeeze-excite gates and hard-swish, which is most of what separates the two generations.",
        "classification",
        _mobilenet_v3(),
    ),
    (
        "densenet121",
        "DenseNet-121",
        "Four dense blocks where every layer reads every earlier output, with compressing transitions between them.",
        "classification",
        _densenet(),
    ),
    (
        "convnext_tiny",
        "ConvNeXt-Tiny",
        "A transformer's design decisions applied to a convolutional network: patchify stem, 7×7 depthwise kernels, LayerNorm, and one activation per block.",
        "classification",
        _convnext(),
    ),
    (
        "vit_b16",
        "Vision Transformer (ViT-B/16)",
        "Cut the image into 16×16 patches, treat them as a sequence, and run twelve encoder layers over it. No convolutions after the patch projection.",
        "classification",
        _vit(),
    ),
    (
        "unet",
        "U-Net",
        "A contracting encoder, an expanding decoder, and skip connections across the middle. The clearest thing here that a list of layers cannot express.",
        "segmentation",
        _unet(),
    ),
    (
        "squeeze_excite_cnn",
        "Squeeze-excite CNN",
        "A small depthwise network with channel-attention gates. A cheap accuracy win on top of any convolutional stack.",
        "classification",
        _squeeze_excite_cnn(),
    ),
    (
        "bert_classifier",
        "BERT-base classifier",
        "Twelve bidirectional post-norm encoder layers pooled into a classifier head. The arrangement every decoder on this canvas departed from.",
        "text_classification",
        _bert_classifier(),
    ),
    (
        "moe_layer",
        "Mixture-of-experts layer, wired by hand",
        "One sparse layer opened up: attention and router each with their own visible residual path. Shows where a mixture of experts actually sits.",
        "language_modeling",
        _moe_layer_by_hand(),
    ),
    (
        "mini_gpt",
        "Mini GPT",
        "A decoder-only transformer: embeddings, positional encoding, N causal blocks, and an LM head.",
        "language_modeling",
        _mini_gpt(),
    ),
    (
        "transformer_primitives",
        "Transformer, wired by hand",
        "One transformer layer opened up into RMSNorm, RoPE, attention, residuals, and SwiGLU — every piece editable.",
        "language_modeling",
        _transformer_from_primitives(),
    ),
]

TEMPLATES: dict[str, ArchitectureTemplate] = {
    template_id: ArchitectureTemplate(
        id=template_id, name=name, description=description, task_type=task_type, graph=graph
    )
    for template_id, name, description, task_type, graph in _TEMPLATES
}

# LLM blueprints at published scales. Built lazily-ish at import from one
# shared graph shape, so a 1B and a 70B preset cannot drift structurally —
# they differ only in the transformer block's parameters.
TEMPLATES.update(
    {
        preset.id: ArchitectureTemplate(
            id=preset.id,
            name=preset.name,
            description=preset.description,
            task_type="language_modeling",
            graph=preset_graph(preset),
        )
        for preset in PRESETS
    }
)


def templates(task_type: str | None = None) -> list[ArchitectureTemplate]:
    values = list(TEMPLATES.values())
    if task_type:
        values = [template for template in values if template.task_type == task_type]
    return values


def template(template_id: str) -> ArchitectureTemplate | None:
    return TEMPLATES.get(template_id)
