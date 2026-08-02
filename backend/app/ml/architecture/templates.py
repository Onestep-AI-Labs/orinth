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
