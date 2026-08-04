"""Transformer primitives, custom code nodes, and language-model training.

Phase 17 stages D and E. The fast tests cover shape arithmetic and generated
source; the `slow` tests build every transformer template under TensorFlow and
train a small language model end to end.
"""

import json
from pathlib import Path

import pytest

from app.ml.architecture.catalog import NODE_SPECS
from app.ml.architecture.emit_keras import EMITTERS, EmitError, emit_module
from app.ml.architecture.graph import build_graph
from app.ml.architecture.lm_catalog import LM_FAMILY, LM_OPTION_ID
from app.ml.architecture.shapes import SHAPE_RULES, infer_shapes
from app.ml.architecture.templates import TEMPLATES, chain, node
from app.ml.training_catalog import training_model_options
from app.schemas import ArchitectureGraph
from app.training.runners import architecture_lm_train as lm

# --- catalog ---------------------------------------------------------------


def test_the_new_node_types_are_present_in_all_three_registries():
    added = {
        "positional_embedding",
        "rotary_embedding",
        "multi_head_attention",
        "rms_norm",
        "swiglu",
        "feed_forward",
        "transformer_block",
        "lm_head",
        "global_avg_pool1d",
        "custom_layer",
        "custom_function",
    }

    assert added <= set(NODE_SPECS)
    assert set(NODE_SPECS) == set(SHAPE_RULES) == set(EMITTERS)


def test_the_language_model_option_is_offered_only_for_language_modeling():
    lm_options = {option.id for option in training_model_options("language_modeling")}
    image_options = {option.id for option in training_model_options("classification")}

    assert LM_OPTION_ID in lm_options
    assert LM_OPTION_ID not in image_options
    assert next(
        option for option in training_model_options() if option.id == LM_OPTION_ID
    ).family == LM_FAMILY


# --- shapes ----------------------------------------------------------------


def lm_graph(**overrides) -> ArchitectureGraph:
    block = {"layers": 2, "num_heads": 4, "key_dim": 16, "ffn_dim": 128, **overrides}
    return ArchitectureGraph(
        nodes=[
            node("in", "input", shape="64"),
            node("embed", "embedding", input_dim=500, output_dim=64),
            node("pos", "positional_embedding", max_length=64),
            node("blocks", "transformer_block", **block),
            node("norm", "rms_norm"),
            node("head", "lm_head", vocab_from_dataset=True),
            node("out", "output"),
        ],
        edges=chain("in", "embed", "pos", "blocks", "norm", "head", "out"),
    )


def resolve(graph: ArchitectureGraph, num_classes: int | None = 500):
    resolved = build_graph(graph)
    shapes, params, issues = infer_shapes(resolved, num_classes)
    return resolved, shapes, params, [*resolved.issues, *issues]


def errors(issues):
    return [issue.message for issue in issues if issue.severity == "error"]


def test_a_language_model_graph_resolves_to_sequence_by_vocabulary():
    _resolved, shapes, _params, issues = resolve(lm_graph())

    assert not errors(issues)
    assert shapes["embed"] == [64, 64]
    assert shapes["blocks"] == [64, 64]
    # One prediction per position, over the vocabulary.
    assert shapes["head"] == [64, 500]


def test_the_lm_head_uses_the_supplied_vocabulary_size():
    _resolved, shapes, _params, _issues = resolve(lm_graph(), num_classes=1234)

    assert shapes["head"] == [64, 1234]


def test_a_fixed_lm_head_ignores_the_dataset_vocabulary():
    graph = lm_graph()
    graph.nodes[5] = node("head", "lm_head", vocab_size=99, vocab_from_dataset=False)

    _resolved, shapes, _params, _issues = resolve(graph, num_classes=500)

    assert shapes["head"] == [64, 99]


def test_a_model_width_not_divisible_by_the_head_count_is_reported():
    graph = lm_graph(num_heads=5)

    _resolved, _shapes, _params, issues = resolve(graph)

    assert any("divide evenly" in message for message in errors(issues))


def test_attention_and_norms_preserve_the_sequence_shape():
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="32"),
            node("embed", "embedding", input_dim=100, output_dim=32),
            node("rope", "rotary_embedding"),
            node("norm", "rms_norm"),
            node("attn", "multi_head_attention", num_heads=2, key_dim=16),
            node("ffn", "swiglu", hidden_dim=64),
            node("pool", "global_avg_pool1d"),
            node("head", "dense", units=3),
            node("out", "output"),
        ],
        edges=chain("in", "embed", "rope", "norm", "attn", "ffn", "pool", "head", "out"),
    )

    _resolved, shapes, _params, issues = resolve(graph, num_classes=3)

    assert not errors(issues)
    for key in ("rope", "norm", "attn", "ffn"):
        assert shapes[key] == [32, 32], key
    assert shapes["pool"] == [32]
    assert shapes["head"] == [3]


def test_a_custom_node_passes_the_shape_through_when_it_declares_none():
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("custom", "custom_function", expression="x * 2.0", output_shape=""),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=chain("in", "custom", "gap", "out"),
    )

    _resolved, shapes, _params, issues = resolve(graph)

    assert not errors(issues)
    assert shapes["custom"] == [8, 8, 3]


def test_a_custom_node_honours_a_declared_output_shape():
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("custom", "custom_function", expression="tf.reduce_mean(x, axis=[1, 2])", output_shape="3"),
            node("head", "dense", units=2),
            node("out", "output"),
        ],
        edges=chain("in", "custom", "head", "out"),
    )

    _resolved, shapes, _params, issues = resolve(graph)

    assert not errors(issues)
    assert shapes["custom"] == [3]
    assert shapes["head"] == [2]


def test_custom_nodes_make_the_parameter_estimate_unknown():
    """Counting parameters inside opaque Python would be a guess, not an estimate."""

    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("custom", "custom_layer", class_name="MyLayer"),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=chain("in", "custom", "gap", "out"),
    )

    _resolved, _shapes, params, _issues = resolve(graph)

    assert params is None


# --- code generation -------------------------------------------------------


def render(graph: ArchitectureGraph, num_classes: int = 500) -> str:
    resolved, shapes, _params, issues = resolve(graph, num_classes)
    assert not errors(issues), errors(issues)
    return emit_module(resolved, shapes, architecture_name="Test", default_num_classes=num_classes)


def test_helper_classes_are_emitted_only_when_the_graph_uses_them():
    lm_code = render(lm_graph())
    cnn_code = render(TEMPLATES["small_cnn"].graph, num_classes=3)

    assert "class RMSNorm(" in lm_code
    assert "class SwiGLU(" in lm_code
    assert "def transformer_block(" in lm_code
    # A plain CNN must not carry transformer machinery it never calls.
    assert "class RMSNorm(" not in cnn_code
    assert "def transformer_block(" not in cnn_code


def test_rope_helper_is_emitted_only_for_graphs_that_use_rotary_embedding():
    assert "class RotaryEmbedding(" in render(TEMPLATES["transformer_primitives"].graph)
    assert "class RotaryEmbedding(" not in render(lm_graph())


def test_the_transformer_block_emits_a_repeat_count_rather_than_repeated_nodes():
    code = render(lm_graph(layers=6))

    assert "layers=6" in code
    # One call site, not six copies of a subgraph.
    assert code.count("transformer_block(x_positional_embedding_1") == 1


def test_attention_is_emitted_as_self_attention_with_the_causal_flag():
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="32"),
            node("embed", "embedding", input_dim=100, output_dim=32),
            node("attn", "multi_head_attention", num_heads=2, key_dim=16, causal=True),
            node("pool", "global_avg_pool1d"),
            node("head", "dense", units=2),
            node("out", "output"),
        ],
        edges=chain("in", "embed", "attn", "pool", "head", "out"),
    )

    code = render(graph, num_classes=2)

    assert "MultiHeadAttention(num_heads=2, key_dim=16" in code
    assert "(x_embedding_1, x_embedding_1, use_causal_mask=True)" in code


def test_the_lm_head_emits_num_classes_when_sized_from_the_dataset():
    assert "tf.keras.layers.Dense(num_classes," in render(lm_graph())


def test_a_custom_layer_is_hoisted_above_build_model_and_instantiated():
    source = "class Doubler(tf.keras.layers.Layer):\n    def call(self, inputs):\n        return inputs * 2\n"
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("custom", "custom_layer", class_name="Doubler", source=source),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=chain("in", "custom", "gap", "out"),
    )

    code = render(graph, num_classes=2)

    assert code.index("class Doubler(") < code.index("def build_model(")
    assert 'Doubler(name="x_custom_layer_1")' in code


def test_two_nodes_sharing_a_custom_class_emit_one_definition():
    """A duplicated class would be a redefinition, not a second layer."""

    source = "class Doubler(tf.keras.layers.Layer):\n    def call(self, inputs):\n        return inputs * 2\n"
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("a", "custom_layer", class_name="Doubler", source=source),
            node("b", "custom_layer", class_name="Doubler", source=source),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=chain("in", "a", "b", "gap", "out"),
    )

    code = render(graph, num_classes=2)

    assert code.count("class Doubler(") == 1
    assert 'Doubler(name="x_custom_layer_1")' in code
    assert 'Doubler(name="x_custom_layer_2")' in code


def test_a_custom_class_name_is_inferred_from_the_source_when_left_blank():
    source = "class Inferred(tf.keras.layers.Layer):\n    def call(self, inputs):\n        return inputs\n"
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("custom", "custom_layer", class_name="", source=source),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=chain("in", "custom", "gap", "out"),
    )

    assert 'Inferred(name="x_custom_layer_1")' in render(graph, num_classes=2)


def test_a_multi_line_custom_function_is_rejected_with_a_pointer_to_custom_layer():
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("custom", "custom_function", expression="y = x * 2\nreturn y"),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=chain("in", "custom", "gap", "out"),
    )
    resolved, shapes, _params, _issues = resolve(graph, 2)

    with pytest.raises(EmitError, match="single expression"):
        emit_module(resolved, shapes, architecture_name="Test")


def test_an_empty_custom_layer_source_is_rejected():
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("custom", "custom_layer", class_name="X", source=""),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=chain("in", "custom", "gap", "out"),
    )
    resolved, shapes, _params, _issues = resolve(graph, 2)

    with pytest.raises(EmitError, match="no source"):
        emit_module(resolved, shapes, architecture_name="Test")


def test_the_hand_wired_transformer_template_has_both_residual_connections():
    resolved = build_graph(TEMPLATES["transformer_primitives"].graph)

    assert resolved.ok
    assert set(resolved.nodes["attn_residual"].inputs) == {"attn", "rope"}
    assert set(resolved.nodes["ffn_residual"].inputs) == {"ffn", "attn_residual"}


# --- language-model runner helpers -----------------------------------------


@pytest.mark.parametrize(
    ("text", "mode", "expected"),
    [
        ("ab c", "char", ["a", "b", " ", "c"]),
        ("Hello, world", "word", ["hello", ",", "world"]),
        ("a  b", "word", ["a", "b"]),
    ],
)
def test_tokenize(text, mode, expected):
    assert lm.tokenize(text, mode) == expected


def test_vocabulary_reserves_pad_and_unknown_and_keeps_frequent_tokens():
    vocabulary = lm.build_vocabulary(["a"] * 5 + ["b"] * 3 + ["c"], "word", limit=4)

    assert vocabulary[0] == lm.PAD
    assert vocabulary[1] == lm.UNKNOWN
    assert vocabulary[2:] == ["a", "b"]


def test_encoding_maps_unseen_tokens_to_unknown():
    vocabulary = [lm.PAD, lm.UNKNOWN, "a", "b"]
    lookup = {token: index for index, token in enumerate(vocabulary)}

    assert lm.encode(["a", "z", "b"], lookup) == [2, 1, 3]


def test_windows_pair_each_input_with_the_next_token():
    inputs, targets = lm.windows([1, 2, 3, 4, 5, 6], length=3, stride=1)

    assert inputs[0].tolist() == [1, 2, 3]
    assert targets[0].tolist() == [2, 3, 4]
    assert len(inputs) == len(targets)


def test_windows_returns_nothing_when_the_corpus_is_shorter_than_one_window():
    inputs, targets = lm.windows([1, 2], length=8, stride=1)

    assert inputs is None and targets is None


def test_corpus_reading_concatenates_every_text_in_a_split(tmp_path: Path):
    texts = tmp_path / "train" / "texts"
    texts.mkdir(parents=True)
    (texts / "a.txt").write_text("first", encoding="utf-8")
    (texts / "b.txt").write_text("second", encoding="utf-8")

    assert lm.read_corpus(tmp_path, "train") == "first\nsecond"
    assert lm.read_corpus(tmp_path, "valid") == ""


# --- real TensorFlow builds ------------------------------------------------


TRANSFORMER_TEMPLATES = [
    "mini_gpt",
    "transformer_primitives",
    "transformer_text_classifier",
    "custom_layer_demo",
    "inception_module",
    "text_cnn",
    "mlp",
]


@pytest.mark.slow
@pytest.mark.parametrize("template_id", TRANSFORMER_TEMPLATES)
def test_new_templates_build_under_tensorflow_and_match_the_analytic_pass(
    template_id, tmp_path, build_generated
):
    template = TEMPLATES[template_id]
    resolved, shapes, estimate, issues = resolve(template.graph, num_classes=11)
    assert not errors(issues)

    code = emit_module(resolved, shapes, architecture_name=template.name, default_num_classes=11)
    model = build_generated(code, tmp_path, f"generated_{template_id}", num_classes=11)

    if estimate is not None:
        assert model.count_params() == estimate
    for node_id in resolved.order:
        predicted = shapes.get(node_id)
        if predicted is None:
            continue
        try:
            layer = model.get_layer(resolved.nodes[node_id].var_name)
        except ValueError:
            continue
        assert tuple(layer.output.shape[1:]) == tuple(predicted), node_id


@pytest.mark.slow
def test_a_language_model_trains_and_writes_its_tokenizer_and_samples(tmp_path: Path):
    """End to end: emit a transformer, train it on a corpus, generate from it."""

    import sys

    from app.ml.architecture.emit_keras import MODULE_FILENAME

    corpus = "the quick brown fox jumps over the lazy dog . " * 200
    for split in ("train", "valid"):
        texts = tmp_path / "dataset" / split / "texts"
        texts.mkdir(parents=True)
        (texts / "corpus.txt").write_text(corpus, encoding="utf-8")

    graph = lm_graph(layers=1)
    graph.nodes[0] = node("in", "input", shape="16")
    graph.nodes[2] = node("pos", "positional_embedding", max_length=16)
    resolved, shapes, _params, issues = resolve(graph, num_classes=60)
    assert not errors(issues)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    module_path = run_dir / MODULE_FILENAME
    module_path.write_text(
        emit_module(resolved, shapes, architecture_name="Tiny LM", default_num_classes=60),
        encoding="utf-8",
    )

    argv = sys.argv
    sys.argv = [
        "architecture_lm_train",
        "--run-dir", str(run_dir),
        "--dataset-root", str(tmp_path / "dataset"),
        "--model-file", str(module_path),
        "--epochs", "2",
        "--batch-size", "8",
        "--learning-rate", "0.005",
        "--advanced", json.dumps(
            {"tokenizer": "word", "vocab_size": 60, "seed": 1, "max_new_tokens": 6, "temperature": 0.0}
        ),
    ]
    try:
        lm.main()
    finally:
        sys.argv = argv

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["tokenizer"] == "word"
    assert metrics["sequence_length"] == 16
    assert metrics["val_perplexity"] > 0
    # A model that learned anything beats uniform guessing over the vocabulary.
    assert metrics["val_loss"] < 3.0

    tokenizer = json.loads((run_dir / "tokenizer.json").read_text(encoding="utf-8"))
    assert tokenizer["mode"] == "word"
    assert tokenizer["vocabulary"][:2] == [lm.PAD, lm.UNKNOWN]

    generations = json.loads((run_dir / "sample_generations.json").read_text(encoding="utf-8"))
    assert generations and generations[0]["generated"].strip()
    assert (run_dir / "best_model.keras").exists()
