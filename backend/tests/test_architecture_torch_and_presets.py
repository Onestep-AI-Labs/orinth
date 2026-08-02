"""PyTorch export, grouped-query attention, mixture of experts, and LLM presets.

The parameter assertions here are the point of the file. A published model's
size is a checkable fact, so a preset that claims to be "Mistral 7B shaped"
either reproduces 7.24B parameters or it is wrong — and the two emitters must
agree with each other and with the analytic estimate.
"""

import pytest

from app.ml.architecture.catalog import NODE_SPECS
from app.ml.architecture.emit_keras import EMITTERS, EmitError, emit_module
from app.ml.architecture.emit_torch import TORCH_EMITTERS, emit_torch_module
from app.ml.architecture.graph import build_graph
from app.ml.architecture.llm_presets import PRESETS, PRESETS_BY_ID
from app.ml.architecture.shapes import infer_shapes
from app.ml.architecture.templates import TEMPLATES, chain, node
from app.schemas import ArchitectureGraph


def resolve(graph: ArchitectureGraph, num_classes: int | None = 128):
    resolved = build_graph(graph)
    shapes, params, issues = infer_shapes(resolved, num_classes)
    return resolved, shapes, params, [*resolved.issues, *issues]


def errors(issues):
    return [issue.message for issue in issues if issue.severity == "error"]


# --- LLM presets -----------------------------------------------------------


# Published parameter counts for the configurations these presets reproduce.
# A preset that drifts from its namesake's size is a preset that lies.
EXPECTED_SIZES = {
    "llm_1b": 1_498_987_776,
    "llm_3b": 3_607_654_656,
    "llm_7b": 7_243_140_352,
    "llm_13b": 13_018_026_240,
    "llm_70b": 70_560_552_192,
    "llm_moe_8x7b": 46_711_541_248,
    # Named configurations, read from published config defaults. These land
    # within a percent of the sizes their model cards advertise, which is the
    # check that the architecture — not just the layer count — is right.
    "qwen3_0_6b": 596_393_984,
    "qwen3_8b": 8_192_288_128,
    "qwen3_30b_a3b": 30_554_644_864,
    "gemma4_sparse": 16_322_753_520,
    "gemma4_dense": 2_940_637_440,
    "deepseek_moe": 721_219_916_288,
}


@pytest.mark.parametrize(("preset_id", "expected"), sorted(EXPECTED_SIZES.items()))
def test_llm_presets_report_their_published_parameter_count(preset_id, expected):
    preset = PRESETS_BY_ID[preset_id]

    _resolved, _shapes, params, issues = resolve(TEMPLATES[preset_id].graph, preset.vocab)

    assert not errors(issues)
    assert params == expected


def test_no_preset_is_silently_clamped_by_a_catalog_maximum():
    """A clamped `layers` once turned the 70B preset into a 57B one, quietly."""

    for preset in PRESETS:
        resolved, _shapes, _params, issues = resolve(TEMPLATES[preset.id].graph, preset.vocab)
        clamps = [
            issue.message
            for issue in issues
            if "raised to the minimum" in issue.message or "lowered to the maximum" in issue.message
        ]
        assert not clamps, f"{preset.id}: {clamps}"
        assert resolved.ok


def test_every_preset_is_registered_as_a_language_modeling_template():
    for preset in PRESETS:
        assert TEMPLATES[preset.id].task_type == "language_modeling"


# --- architecture features, not just sizes ---------------------------------


@pytest.mark.parametrize(
    ("preset_id", "flag"),
    [
        ("qwen3_8b", "qk_norm"),
        ("qwen3_0_6b", "qk_norm"),
        ("qwen3_30b_a3b", "qk_norm"),
        ("gemma4_sparse", "sliding_window"),
        ("gemma4_dense", "sliding_window"),
    ],
)
def test_named_presets_carry_their_familys_signature_feature(preset_id, flag):
    """A preset that only matches a parameter count is a generic transformer.

    Qwen3's QK-norm and Gemma 4's sliding-window attention are what make those
    models what they are, so the graph has to actually switch them on.
    """

    graph = TEMPLATES[preset_id].graph
    block = next(node for node in graph.nodes if node.type == "transformer_block")

    assert block.params[flag]


def test_gemma_presets_tie_embeddings_and_softcap_logits():
    for preset_id in ("gemma4_sparse", "gemma4_dense"):
        head = next(
            node for node in TEMPLATES[preset_id].graph.nodes if node.type == "lm_head"
        )
        assert head.params["tie_embeddings"]
        assert head.params["logit_softcap"] > 0


def test_gemma_alternates_local_and_global_attention_layers():
    block = next(
        node for node in TEMPLATES["gemma4_sparse"].graph.nodes
        if node.type == "transformer_block"
    )

    assert block.params["sliding_window"] == 512
    assert block.params["global_every"] == 6


def test_deepseek_preset_uses_a_shared_expert_alongside_routed_ones():
    block = next(
        node for node in TEMPLATES["deepseek_moe"].graph.nodes
        if node.type == "transformer_block"
    )

    assert block.params["shared_experts"] == 1
    assert block.params["num_experts"] == 256
    assert block.params["experts_per_token"] == 8


def test_gemma_head_dimension_is_decoupled_from_the_residual_width():
    """Gemma sets head_dim independently of width / heads.

    Its 8 heads of 256 total 2048, against a 2304-wide residual stream — and
    deriving head_dim as width / heads would give 288, a different model. The
    preset therefore states it, and this asserts the decoupling survived.
    """

    graph = TEMPLATES["gemma4_sparse"].graph
    block = next(node for node in graph.nodes if node.type == "transformer_block")
    embedding = next(node for node in graph.nodes if node.type == "embedding")

    width = embedding.params["output_dim"]
    assert block.params["key_dim"] == 256
    assert block.params["num_heads"] * block.params["key_dim"] != width
    assert block.params["key_dim"] != width // block.params["num_heads"]


def test_tying_embeddings_removes_the_head_projection_from_the_count():
    tied = resolve(TEMPLATES["gemma4_dense"].graph, 262144)[2]

    graph = TEMPLATES["gemma4_dense"].graph.model_copy(deep=True)
    for graph_node in graph.nodes:
        if graph_node.type == "lm_head":
            graph_node.params["tie_embeddings"] = False
    untied = resolve(graph, 262144)[2]

    # A 262k vocabulary at width 2304 is ~604M parameters not spent twice.
    assert untied - tied == 262144 * 2304 + 262144


def test_qk_norm_adds_two_scale_vectors_per_attention():
    graph = TEMPLATES["qwen3_8b"].graph.model_copy(deep=True)
    with_norm = resolve(graph, 151936)[2]
    for graph_node in graph.nodes:
        if graph_node.type == "transformer_block":
            graph_node.params["qk_norm"] = False
    without = resolve(graph, 151936)[2]

    block = next(n for n in graph.nodes if n.type == "transformer_block")
    assert with_norm - without == block.params["layers"] * 2 * block.params["key_dim"]


def test_sparse_presets_hold_more_parameters_than_their_dense_twin():
    """The MoE trade: same width and depth, far more total capacity."""

    dense = resolve(TEMPLATES["llm_7b"].graph, PRESETS_BY_ID["llm_7b"].vocab)[2]
    sparse = resolve(TEMPLATES["llm_moe_8x7b"].graph, PRESETS_BY_ID["llm_moe_8x7b"].vocab)[2]

    assert sparse > dense * 5


# --- grouped-query attention ------------------------------------------------


def attention_graph(**overrides) -> ArchitectureGraph:
    params = {"num_heads": 8, "num_kv_heads": 2, "key_dim": 16, **overrides}
    return ArchitectureGraph(
        nodes=[
            node("in", "input", shape="32"),
            node("embed", "embedding", input_dim=100, output_dim=128),
            node("attn", "grouped_query_attention", **params),
            node("pool", "global_avg_pool1d"),
            node("head", "dense", units=4),
            node("out", "output"),
        ],
        edges=chain("in", "embed", "attn", "pool", "head", "out"),
    )


def test_grouped_query_attention_preserves_the_sequence_shape():
    _resolved, shapes, _params, issues = resolve(attention_graph(), 4)

    assert not errors(issues)
    assert shapes["attn"] == [32, 128]


def test_more_kv_heads_than_query_heads_is_rejected():
    _resolved, _shapes, _params, issues = resolve(attention_graph(num_heads=4, num_kv_heads=8), 4)

    assert any("at most as many key/value heads" in message for message in errors(issues))


def test_query_heads_must_divide_into_kv_head_groups():
    _resolved, _shapes, _params, issues = resolve(attention_graph(num_heads=6, num_kv_heads=4), 4)

    assert any("divide evenly" in message for message in errors(issues))


def test_fewer_kv_heads_means_fewer_parameters():
    """The whole point of GQA — the estimate has to show the saving."""

    grouped = resolve(attention_graph(num_heads=8, num_kv_heads=2), 4)[2]
    full = resolve(attention_graph(num_heads=8, num_kv_heads=8), 4)[2]

    assert grouped < full


# --- mixture of experts ------------------------------------------------------


def moe_graph(**overrides) -> ArchitectureGraph:
    params = {"num_experts": 8, "experts_per_token": 2, "hidden_dim": 256, **overrides}
    return ArchitectureGraph(
        nodes=[
            node("in", "input", shape="32"),
            node("embed", "embedding", input_dim=100, output_dim=64),
            node("moe", "moe_feed_forward", **params),
            node("pool", "global_avg_pool1d"),
            node("head", "dense", units=4),
            node("out", "output"),
        ],
        edges=chain("in", "embed", "moe", "pool", "head", "out"),
    )


def test_mixture_of_experts_preserves_the_sequence_shape():
    _resolved, shapes, _params, issues = resolve(moe_graph(), 4)

    assert not errors(issues)
    assert shapes["moe"] == [32, 64]


def test_total_parameters_scale_with_the_expert_count():
    two = resolve(moe_graph(num_experts=2), 4)[2]
    eight = resolve(moe_graph(num_experts=8), 4)[2]

    # Four times the experts, so roughly four times the feed-forward weights.
    assert eight > two * 3


def test_routing_to_more_experts_than_exist_is_rejected():
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="32"),
            node("embed", "embedding", input_dim=100, output_dim=64),
            node("blocks", "transformer_block", num_heads=4, key_dim=16, ffn="moe",
                 num_experts=2, experts_per_token=8),
            node("pool", "global_avg_pool1d"),
            node("head", "dense", units=4),
            node("out", "output"),
        ],
        edges=chain("in", "embed", "blocks", "pool", "head", "out"),
    )

    _resolved, _shapes, _params, issues = resolve(graph, 4)

    assert any("of only 2 experts" in message for message in errors(issues))


# --- PyTorch emitter ---------------------------------------------------------


def test_the_torch_emitter_covers_all_but_the_framework_bound_nodes():
    """Everything except nodes whose meaning is Keras-specific."""

    untranslatable = {"pretrained_backbone", "custom_layer"}

    assert set(TORCH_EMITTERS) == set(NODE_SPECS) - untranslatable
    assert untranslatable <= set(EMITTERS)


def render_torch(graph: ArchitectureGraph, num_classes: int = 7) -> str:
    resolved, shapes, _params, issues = resolve(graph, num_classes)
    assert not errors(issues), errors(issues)
    return emit_torch_module(
        resolved, shapes, architecture_name="Test model", default_num_classes=num_classes
    )


def test_torch_export_produces_a_module_with_a_build_model_entry_point():
    code = render_torch(TEMPLATES["small_cnn"].graph)

    assert "import torch" in code
    assert "class TestModel(nn.Module):" in code
    assert "def build_model(num_classes: int = 7) -> nn.Module:" in code
    assert "def forward(self" in code


def test_torch_layers_are_constructed_with_their_input_width():
    """torch cannot infer input width, so the shape pass has to supply it."""

    code = render_torch(TEMPLATES["small_cnn"].graph)

    # 3 input channels into 32 filters, then 32 into 64.
    assert "nn.Conv2d(3, 32, 3" in code
    assert "nn.Conv2d(32, 64, 3" in code


def test_torch_export_names_its_channel_order_in_the_header():
    """NCHW versus the canvas's NHWC is the one thing that will trip a user."""

    assert "(batch, channels, height, width)" in render_torch(TEMPLATES["small_cnn"].graph)


def test_torch_helpers_are_emitted_only_when_used():
    transformer = render_torch(TEMPLATES["mini_gpt"].graph)
    cnn = render_torch(TEMPLATES["small_cnn"].graph)

    assert "class TransformerStack(nn.Module):" in transformer
    assert "class SwiGLU(nn.Module):" in transformer
    assert "class TransformerStack" not in cnn


def test_a_pretrained_backbone_cannot_be_exported_to_torch():
    with pytest.raises(EmitError, match="no PyTorch equivalent"):
        render_torch(TEMPLATES["transfer_learning"].graph)


def test_a_tensorflow_custom_layer_cannot_be_exported_to_torch():
    with pytest.raises(EmitError, match="no PyTorch equivalent"):
        render_torch(TEMPLATES["custom_layer_demo"].graph)


def test_a_custom_function_calling_tensorflow_is_rejected_with_a_reason():
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("custom", "custom_function", expression="tf.nn.gelu(x)"),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=chain("in", "custom", "gap", "out"),
    )

    with pytest.raises(EmitError, match="calls TensorFlow"):
        render_torch(graph, 2)


def test_a_torch_safe_custom_function_exports_fine():
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("custom", "custom_function", expression="x * 2.0"),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=chain("in", "custom", "gap", "out"),
    )

    assert "(lambda x: x * 2.0)" in render_torch(graph, 2)


# --- both frameworks, really built ------------------------------------------

TORCH_BUILDABLE = [
    "mlp",
    "small_cnn",
    "residual_block",
    "inception_module",
    "text_cnn",
    "transformer_text_classifier",
    "mini_gpt",
    "transformer_primitives",
    "llm_tiny",
]
# `llm_moe_small` is deliberately absent: at 193M parameters it allocates
# ~780MB, and building it in the same process as the TensorFlow suites pushes
# the run out of memory. `test_a_mixture_of_experts_holds_every_expert_...`
# covers the MoE path on a small graph instead.


@pytest.mark.slow
@pytest.mark.parametrize("template_id", TORCH_BUILDABLE)
def test_torch_modules_build_and_agree_with_the_analytic_estimate(template_id, tmp_path):
    """The check that keeps the second emitter honest.

    Keras and torch count a few things differently — Keras includes
    BatchNormalization's moving statistics, and torch's LSTM carries a second
    bias vector — so only templates without those layers are compared exactly.
    Everything else must still build.
    """

    import importlib.util
    import sys

    resolved, shapes, estimate, issues = resolve(TEMPLATES[template_id].graph, 7)
    assert not errors(issues)
    code = emit_torch_module(
        resolved, shapes, architecture_name=TEMPLATES[template_id].name, default_num_classes=7
    )

    path = tmp_path / f"{template_id}_torch.py"
    path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"{template_id}_torch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        model = module.build_model(num_classes=7)
    finally:
        sys.modules.pop(spec.name, None)

    total = sum(parameter.numel() for parameter in model.parameters())
    counts_differ = any(
        resolved.nodes[node_id].type in {"batch_norm", "lstm", "gru"}
        for node_id in resolved.order
    )
    if estimate is not None and not counts_differ:
        assert total == estimate
    else:
        assert total > 0


@pytest.mark.slow
def test_a_mixture_of_experts_holds_every_expert_and_routes_a_forward_pass(tmp_path):
    import importlib.util
    import sys

    import torch

    resolved, shapes, estimate, issues = resolve(moe_graph(num_experts=4), 4)
    assert not errors(issues)
    code = emit_torch_module(resolved, shapes, architecture_name="MoE", default_num_classes=4)
    path = tmp_path / "moe_torch.py"
    path.write_text(code, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("moe_torch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["moe_torch"] = module
    try:
        spec.loader.exec_module(module)
        model = module.build_model(num_classes=4)
    finally:
        sys.modules.pop("moe_torch", None)

    assert sum(parameter.numel() for parameter in model.parameters()) == estimate
    output = model(torch.randint(0, 100, (2, 32)))
    assert tuple(output.shape) == (2, 4)


@pytest.mark.slow
def test_keras_and_torch_agree_on_a_transformer_parameter_count(tmp_path, build_generated):
    """One graph, two emitters, one number."""

    import importlib.util
    import sys

    graph = TEMPLATES["mini_gpt"].graph
    resolved, shapes, estimate, issues = resolve(graph, 7)
    assert not errors(issues)

    keras_model = build_generated(
        emit_module(resolved, shapes, architecture_name="Mini GPT", default_num_classes=7),
        tmp_path,
        "mini_gpt_keras",
        7,
    )

    path = tmp_path / "mini_gpt_torch.py"
    path.write_text(
        emit_torch_module(resolved, shapes, architecture_name="Mini GPT", default_num_classes=7),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("mini_gpt_torch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mini_gpt_torch"] = module
    try:
        spec.loader.exec_module(module)
        torch_model = module.build_model(num_classes=7)
    finally:
        sys.modules.pop("mini_gpt_torch", None)

    torch_total = sum(parameter.numel() for parameter in torch_model.parameters())
    assert keras_model.count_params() == estimate == torch_total
