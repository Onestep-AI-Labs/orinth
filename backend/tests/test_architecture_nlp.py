"""The NLP palette: sequence ops, text blocks, and both emitters.

Fast coverage of shape arithmetic, parameter estimates, and generated source
for the nodes added in the NLP category. The `slow` counterparts — building
each of them under real TensorFlow and PyTorch and comparing the three
parameter counts — live beside the other blocks in
`test_architecture_families.py`.
"""

import pytest

from app.ml.architecture.blocks import NLP_BLOCKS
from app.ml.architecture.catalog import NODE_SPECS, node_catalog
from app.ml.architecture.emit_keras import EMITTERS, emit_module
from app.ml.architecture.emit_torch import TORCH_EMITTERS, emit_torch_module
from app.ml.architecture.graph import build_graph, parse_int_list
from app.ml.architecture.shapes import SHAPE_RULES, infer_shapes
from app.ml.architecture.templates import TEMPLATES, chain, edge, node
from app.schemas import ArchitectureGraph

NLP_NODES = {
    "sinusoidal_position_encoding",
    "cross_attention",
    "sequence_pool",
    "span_head",
    "text_cnn_block",
    "bilstm_encoder",
}


def resolve(graph: ArchitectureGraph, num_classes: int | None = 4):
    resolved = build_graph(graph)
    shapes, params, issues = infer_shapes(resolved, num_classes)
    return resolved, shapes, params, [*resolved.issues, *issues]


def errors(issues):
    return [issue.message for issue in issues if issue.severity == "error"]


def text_graph(*middle, length: int = 200, width: int = 64) -> ArchitectureGraph:
    """Embedding → the nodes under test → a classifier head."""

    ids = [f"n{index}" for index, _ in enumerate(middle)]
    return ArchitectureGraph(
        nodes=[
            node("input", "input", shape=str(length)),
            node("embed", "embedding", input_dim=500, output_dim=width),
            *(node(node_id, node_type, **params) for node_id, (node_type, params) in zip(ids, middle, strict=True)),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        edges=chain("input", "embed", *ids, "head", "output"),
    )


# --- catalog ----------------------------------------------------------------


def test_the_nlp_nodes_are_present_in_every_registry():
    assert NLP_NODES <= set(NODE_SPECS)
    assert set(NODE_SPECS) == set(SHAPE_RULES) == set(EMITTERS)
    # Nothing here is Keras-bound, so all six export to PyTorch too.
    assert NLP_NODES <= set(TORCH_EMITTERS)


def test_every_nlp_block_is_declared_as_a_block_with_a_source():
    for block in NLP_BLOCKS:
        spec = NODE_SPECS[block.type]
        assert spec.kind == "block"
        assert spec.category == "NLP"
        assert spec.source, block.type


def test_language_specific_nodes_are_offered_only_for_text_tasks():
    text_nodes = {spec.type for spec in node_catalog("text_classification")}
    image_nodes = {spec.type for spec in node_catalog("classification")}

    assert {"text_cnn_block", "bilstm_encoder"} <= text_nodes
    assert not {"text_cnn_block", "bilstm_encoder"} & image_nodes
    # Pooling, cross-attention and a fixed position signal are as useful to a
    # Vision Transformer, so they stay universal.
    assert {"sequence_pool", "cross_attention", "sinusoidal_position_encoding"} <= (
        text_nodes & image_nodes
    )
    # A span head only means something for extractive question answering.
    assert "span_head" not in text_nodes
    assert "span_head" in {spec.type for spec in node_catalog("question_answering")}


# --- shapes and parameter estimates -----------------------------------------


def test_a_sinusoidal_encoding_preserves_the_sequence_and_carries_no_weights():
    _resolved, shapes, params, issues = resolve(
        text_graph(("sinusoidal_position_encoding", {"base": 10000.0}),
                   ("sequence_pool", {"mode": "mean"}))
    )

    assert not errors(issues)
    assert shapes["n0"] == [200, 64]
    # 500 × 64 embedding, mean pooling, and a 4-class head — nothing else.
    assert params == 500 * 64 + (64 * 4 + 4)


def test_sequence_pooling_collapses_to_one_vector_per_mode():
    for mode in ("mean", "cls", "max", "attention"):
        _resolved, shapes, _params, issues = resolve(
            text_graph(("sequence_pool", {"mode": mode, "hidden_dim": 32}))
        )

        assert not errors(issues), mode
        assert shapes["n0"] == [64], mode


def test_only_attention_pooling_carries_weights():
    def pooled_params(mode: str) -> int:
        return resolve(text_graph(("sequence_pool", {"mode": mode, "hidden_dim": 32})))[2]

    baseline = pooled_params("mean")

    assert pooled_params("max") == pooled_params("cls") == baseline
    # A tanh scoring layer with a bias, then a bias-free score per token.
    assert pooled_params("attention") == baseline + (64 * 32 + 32 + 32)


def test_a_span_head_emits_two_logits_per_token():
    _resolved, shapes, _params, issues = resolve(
        ArchitectureGraph(
            nodes=[
                node("input", "input", shape="200"),
                node("embed", "embedding", input_dim=500, output_dim=64),
                node("span", "span_head"),
                node("output", "output"),
            ],
            edges=chain("input", "embed", "span", "output"),
        )
    )

    assert not errors(issues)
    assert shapes["span"] == [200, 2]


def test_a_text_cnn_emits_one_vector_of_filters_per_kernel_width():
    _resolved, shapes, params, issues = resolve(
        text_graph(("text_cnn_block", {"filters": 32, "kernel_sizes": "3,4,5"}))
    )

    assert not errors(issues)
    assert shapes["n0"] == [96]
    convolutions = sum(kernel * 64 * 32 + 32 for kernel in (3, 4, 5))
    assert params == 500 * 64 + convolutions + (96 * 4 + 4)


def test_unusable_kernel_widths_are_reported_against_the_node():
    _resolved, _shapes, _params, issues = resolve(
        text_graph(("text_cnn_block", {"filters": 32, "kernel_sizes": "3,four"}))
    )

    assert any("Kernel widths" in message for message in errors(issues))


def test_a_bilstm_encoder_doubles_its_units_and_can_drop_the_sequence():
    sequence = resolve(
        text_graph(("bilstm_encoder", {"units": 32, "layers": 2, "return_sequences": True}),
                   ("sequence_pool", {"mode": "mean"}))
    )
    vector = resolve(
        text_graph(("bilstm_encoder", {"units": 32, "layers": 2, "return_sequences": False}))
    )

    assert not errors(sequence[3])
    assert sequence[1]["n0"] == [200, 64]
    assert not errors(vector[3])
    assert vector[1]["n0"] == [64]


def test_a_stacked_encoder_counts_the_second_layer_reading_both_directions():
    def encoder_params(layers: int) -> int:
        graph = text_graph(("bilstm_encoder", {"units": 32, "layers": layers,
                                               "return_sequences": False}))
        return resolve(graph)[2]

    one = 2 * 4 * ((64 + 32) * 32 + 32)
    # The second layer reads 64 features — the first layer's two directions
    # concatenated — not the embedding's 64 by coincidence of size.
    two = one + 2 * 4 * ((64 + 32) * 32 + 32)

    assert encoder_params(2) - encoder_params(1) == two - one


def test_cross_attention_returns_the_query_sequence():
    graph = ArchitectureGraph(
        nodes=[
            node("input", "input", shape="32"),
            node("query", "embedding", input_dim=500, output_dim=64),
            node("context", "embedding", input_dim=500, output_dim=64),
            node("attend", "cross_attention", num_heads=4, key_dim=16),
            node("pool", "sequence_pool", mode="mean"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        edges=[
            edge("input", "query"),
            edge("input", "context"),
            edge("query", "attend"),
            edge("context", "attend"),
            *chain("attend", "pool", "head", "output"),
        ],
    )

    _resolved, shapes, params, issues = resolve(graph)

    assert not errors(issues)
    # The context is 32 long too here, but the output follows the query either way.
    assert shapes["attend"] == [32, 64]
    # Q, K, V and the output projection, all 64 × 64 with a bias.
    assert params - 2 * 500 * 64 - (64 * 4 + 4) == 4 * (64 * 64 + 64)


def test_cross_attention_needs_two_inputs():
    graph = ArchitectureGraph(
        nodes=[
            node("input", "input", shape="32"),
            node("query", "embedding", input_dim=500, output_dim=64),
            node("attend", "cross_attention", num_heads=4, key_dim=16),
            node("output", "output"),
        ],
        edges=chain("input", "query", "attend", "output"),
    )

    _resolved, _shapes, _params, issues = resolve(graph)

    assert errors(issues)


# --- generated source --------------------------------------------------------


def render(graph: ArchitectureGraph, num_classes: int = 4) -> tuple[str, str]:
    resolved, shapes, _params, issues = resolve(graph, num_classes)
    assert not errors(issues), errors(issues)
    return (
        emit_module(resolved, shapes, architecture_name="Text model",
                    default_num_classes=num_classes),
        emit_torch_module(resolved, shapes, architecture_name="Text model",
                          default_num_classes=num_classes),
    )


def test_the_text_cnn_block_emits_its_kernel_widths_to_both_frameworks():
    keras_code, torch_code = render(
        text_graph(("text_cnn_block", {"filters": 32, "kernel_sizes": "3,4,5",
                                       "activation": "relu", "dropout": 0.5}))
    )

    assert "kernel_sizes=(3, 4, 5)" in keras_code
    assert "def text_cnn(" in keras_code
    assert "TextCNN(64, 32, (3, 4, 5), activation=nn.ReLU()" in torch_code


def test_the_encoder_emits_the_chosen_cell_to_both_frameworks():
    keras_code, torch_code = render(
        text_graph(("bilstm_encoder", {"cell": "gru", "units": 32, "layers": 2,
                                       "return_sequences": False}))
    )

    assert 'cell="gru"' in keras_code
    assert "tf.keras.layers.Bidirectional" in keras_code
    assert 'cell="gru"' in torch_code
    assert "bidirectional=True" in torch_code


def test_cross_attention_emits_the_query_first_in_both_frameworks():
    graph = ArchitectureGraph(
        nodes=[
            node("input", "input", shape="32"),
            node("query", "embedding", input_dim=500, output_dim=64),
            node("context", "embedding", input_dim=500, output_dim=32),
            node("attend", "cross_attention", num_heads=4, key_dim=16),
            node("pool", "sequence_pool", mode="cls"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        edges=[
            edge("input", "query"),
            edge("input", "context"),
            edge("query", "attend"),
            edge("context", "attend"),
            *chain("attend", "pool", "head", "output"),
        ],
    )

    keras_code, torch_code = render(graph)

    # Variables are numbered in topological order, so the 32-wide context is
    # `x_embedding_1` and the 64-wide query is `x_embedding_2`. The query has to
    # come first at the call whatever the graph named it.
    assert (
        'MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.0, '
        'name="x_cross_attention_1")(x_embedding_2, x_embedding_1)'
    ) in keras_code
    assert (
        "self.x_cross_attention_1(x_embedding_2, x_embedding_1, x_embedding_1"
    ) in torch_code
    # torch needs the context's width stated; it differs from the query's here.
    assert "nn.MultiheadAttention(64, 4, dropout=0.0, kdim=32, vdim=32" in torch_code


def test_only_the_helpers_a_graph_reaches_are_emitted():
    keras_code, torch_code = render(text_graph(("sequence_pool", {"mode": "max"})))

    assert "class SequencePooling" in keras_code
    assert "class SequencePooling" in torch_code
    # A pooling-only graph has no business carrying a Text CNN.
    assert "def text_cnn(" not in keras_code
    assert "class TextCNN" not in torch_code


# --- templates ---------------------------------------------------------------


@pytest.mark.parametrize("template_id", ["kim_text_cnn", "bilstm_attention_text"])
def test_the_new_text_templates_resolve_and_generate(template_id):
    template = TEMPLATES[template_id]

    assert template.task_type == "text_classification"
    keras_code, torch_code = render(template.graph)
    assert "def build_model(" in keras_code
    assert "def build_model(" in torch_code


# --- parsing -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3,4,5", [3, 4, 5]),
        (" 3 , 4 ", [3, 4]),
        ([3, 4], [3, 4]),
        ("5", [5]),
        ("", None),
        ("3,0", None),
        ("3,-1", None),
        ("3,four", None),
        (None, None),
    ],
)
def test_kernel_width_parsing(raw, expected):
    assert parse_int_list(raw) == expected
