"""Named vision and LLM blocks: structure, shapes, and both emitters.

The `slow` tests here are the ones that matter. Every block is built under real
TensorFlow *and* real PyTorch, and both are compared against the analytic
estimate. A block whose two emitters disagree is describing two different
models, and a block whose estimate is wrong is lying about what a config costs
— neither is catchable by reading the generated source.
"""

import importlib.util
import pathlib
import tempfile

import pytest

from app.ml.architecture.blocks import (
    LLM_FAMILIES,
    VISION_BLOCKS,
    llm_block_params,
)
from app.ml.architecture.catalog import NODE_SPECS
from app.ml.architecture.emit_keras import emit_module
from app.ml.architecture.emit_torch import emit_torch_module
from app.ml.architecture.graph import build_graph
from app.ml.architecture.shapes import infer_shapes
from app.ml.architecture.templates import TEMPLATES, chain, edge, node
from app.schemas import ArchitectureGraph


def resolve(graph: ArchitectureGraph, num_classes: int | None = 4):
    resolved = build_graph(graph)
    shapes, params, issues = infer_shapes(resolved, num_classes)
    return resolved, shapes, params, [*resolved.issues, *issues]


def errors(issues):
    return [issue.message for issue in issues if issue.severity == "error"]


def load(source: str, filename: str):
    path = pathlib.Path(tempfile.mkdtemp()) / filename
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem + str(abs(hash(source))), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_graph(block_type: str, **params) -> ArchitectureGraph:
    return ArchitectureGraph(
        nodes=[
            node("input", "input", shape="32,32,3"),
            node("block", block_type, **params),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units_from_dataset=True, activation="softmax"),
            node("output", "output"),
        ],
        edges=chain("input", "block", "gap", "head", "output"),
    )


def sequence_graph(block_type: str, width: int = 64, vocab: int = 500, **params):
    return ArchitectureGraph(
        nodes=[
            node("input", "input", shape="64"),
            node("embed", "embedding", input_dim=vocab, output_dim=width),
            node("block", block_type, **params),
            node("final_norm", "rms_norm"),
            node("head", "lm_head", vocab_size=vocab, vocab_from_dataset=False),
            node("output", "output"),
        ],
        edges=chain("input", "embed", "block", "final_norm", "head", "output"),
    )


# Small stand-ins for the published configs — same structure, buildable in a
# test. The families' real sizes are asserted in
# `test_architecture_torch_and_presets.py` against their published counts.
SMALL_LLM_OVERRIDES: dict[str, dict] = {
    "llama_block": dict(layers=2, num_heads=4, num_kv_heads=2, head_dim=16, ffn_dim=128),
    "qwen3_block": dict(layers=2, num_heads=4, num_kv_heads=2, head_dim=16, ffn_dim=128),
    "mistral_block": dict(
        layers=2, num_heads=4, num_kv_heads=2, head_dim=16, ffn_dim=128, sliding_window=16
    ),
    "mixtral_block": dict(
        layers=2, num_heads=4, num_kv_heads=2, head_dim=16, ffn_dim=64,
        num_experts=4, experts_per_token=2,
    ),
    "gemma3_block": dict(
        layers=3, num_heads=4, num_kv_heads=2, head_dim=16, ffn_dim=128,
        sliding_window=16, global_every=2,
    ),
    "deepseek_block": dict(
        layers=3, num_heads=4, ffn_dim=32, dense_ffn_dim=64, num_experts=4,
        experts_per_token=2, shared_experts=1, dense_layers=1, kv_lora_rank=32,
        q_lora_rank=48, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
    ),
    "kimi_block": dict(
        layers=2, num_heads=4, ffn_dim=32, dense_ffn_dim=64, num_experts=6,
        experts_per_token=2, shared_experts=1, dense_layers=1, kv_lora_rank=32,
        q_lora_rank=48, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
    ),
    "gpt2_block": dict(layers=2, num_heads=4, num_kv_heads=4, head_dim=16, ffn_dim=128),
    "bert_block": dict(layers=2, num_heads=4, num_kv_heads=4, head_dim=16, ffn_dim=128),
}

SMALL_VISION_OVERRIDES: dict[str, dict] = {
    "resnet_block": dict(variant="basic", filters=16, blocks=2, stride=2),
    "inverted_residual_block": dict(filters=24, expand_ratio=6, stride=2, blocks=2),
    "dense_block": dict(growth_rate=8, layers=4),
    "inception_block": {},
    "convnext_block": dict(filters=32, blocks=2),
    "vit_block": dict(layers=2, num_heads=4, head_dim=16, mlp_dim=128),
}

LLM_TYPES = sorted(SMALL_LLM_OVERRIDES)
VISION_TYPES = sorted(SMALL_VISION_OVERRIDES)


# --- catalog ----------------------------------------------------------------


def test_every_declared_family_has_a_catalog_entry_marked_as_a_block():
    for family in [*LLM_FAMILIES, *VISION_BLOCKS]:
        spec = NODE_SPECS[family.type]
        assert spec.kind == "block"
        # A block's numbers are only meaningful if you can check where they
        # came from, so every one records its published source.
        assert spec.source, family.type


def test_primitives_are_not_marked_as_blocks():
    for node_type in ("conv2d", "dense", "rms_norm", "mla_attention", "geglu"):
        assert NODE_SPECS[node_type].kind == "layer"


def test_family_defaults_cover_every_llm_block_parameter():
    """A missing default would emit as a KeyError at generation time."""

    for family in LLM_FAMILIES:
        params = llm_block_params(family.type)
        for spec in NODE_SPECS[family.type].params:
            assert spec.key in params, f"{family.type}.{spec.key}"


# --- shapes and validation --------------------------------------------------


@pytest.mark.parametrize("block_type", VISION_TYPES)
def test_vision_blocks_resolve_a_shape(block_type):
    graph = (
        image_graph(block_type, **SMALL_VISION_OVERRIDES[block_type])
        if block_type != "vit_block"
        else ArchitectureGraph(
            nodes=[
                node("input", "input", shape="32,32,3"),
                node("patches", "patch_embedding", patch_size=8, embed_dim=64),
                node("block", "vit_block", **SMALL_VISION_OVERRIDES[block_type]),
                node("pool", "global_avg_pool1d"),
                node("head", "dense", units_from_dataset=True, activation="softmax"),
                node("output", "output"),
            ],
            edges=chain("input", "patches", "block", "pool", "head", "output"),
        )
    )
    _resolved, shapes, params, issues = resolve(graph)

    assert not errors(issues), errors(issues)
    assert shapes["block"] is not None
    assert params is not None and params > 0


@pytest.mark.parametrize("block_type", LLM_TYPES)
def test_llm_blocks_preserve_the_sequence_shape(block_type):
    _resolved, shapes, params, issues = resolve(
        sequence_graph(block_type, **SMALL_LLM_OVERRIDES[block_type])
    )

    assert not errors(issues), errors(issues)
    assert shapes["block"] == [64, 64]
    assert params is not None and params > 0


def test_a_patch_size_that_does_not_divide_the_image_is_rejected():
    graph = ArchitectureGraph(
        nodes=[
            node("input", "input", shape="30,30,3"),
            node("patches", "patch_embedding", patch_size=16, embed_dim=64),
            node("pool", "global_avg_pool1d"),
            node("head", "dense", units=4),
            node("output", "output"),
        ],
        edges=chain("input", "patches", "pool", "head", "output"),
    )

    assert any("does not divide" in message for message in errors(resolve(graph)[3]))


def test_a_vit_block_whose_heads_do_not_fill_the_width_is_rejected():
    """Heads × head dim must equal the token width, or the two emitters diverge."""

    graph = ArchitectureGraph(
        nodes=[
            node("input", "input", shape="32,32,3"),
            node("patches", "patch_embedding", patch_size=8, embed_dim=64),
            node("block", "vit_block", layers=1, num_heads=3, head_dim=16, mlp_dim=64),
            node("pool", "global_avg_pool1d"),
            node("head", "dense", units=4),
            node("output", "output"),
        ],
        edges=chain("input", "patches", "block", "pool", "head", "output"),
    )

    assert any("must equal the token width" in message for message in errors(resolve(graph)[3]))


def test_routing_a_token_to_more_experts_than_exist_is_rejected():
    graph = sequence_graph(
        "mixtral_block", layers=1, num_heads=4, num_kv_heads=2, head_dim=16,
        ffn_dim=64, num_experts=2, experts_per_token=8,
    )

    assert any("of only 2 experts" in message for message in errors(resolve(graph)[3]))


def test_more_leading_dense_layers_than_layers_is_rejected():
    graph = sequence_graph(
        "deepseek_block", layers=2, num_heads=4, ffn_dim=32, num_experts=4,
        experts_per_token=2, dense_layers=5, kv_lora_rank=32, qk_nope_head_dim=16,
        qk_rope_head_dim=8, v_head_dim=16,
    )

    assert any("leading dense layers" in message for message in errors(resolve(graph)[3]))


# --- generated code ---------------------------------------------------------


@pytest.mark.parametrize("block_type", LLM_TYPES)
def test_the_generated_code_records_the_familys_structure(block_type):
    """Every structural parameter appears in the call, not just in a comment.

    The point of a named block is that its generated file says what the
    architecture *is*. Reading `norm_placement="sandwich"` in a Gemma module is
    the difference between code that documents itself and code that needs the
    helper opened to be understood.
    """

    resolved, shapes, _params, issues = resolve(
        sequence_graph(block_type, **SMALL_LLM_OVERRIDES[block_type])
    )
    assert not errors(issues)
    code = emit_module(resolved, shapes, architecture_name=block_type)

    defaults = llm_block_params(block_type)
    assert "def llm_block(" in code
    for key in ("attention", "ffn", "norm_placement", "router"):
        assert f'{key}="{defaults[key]}"' in code


def test_deepseek_emits_latent_attention_and_a_sigmoid_router():
    resolved, shapes, _params, _issues = resolve(
        sequence_graph("deepseek_block", **SMALL_LLM_OVERRIDES["deepseek_block"])
    )
    code = emit_module(resolved, shapes, architecture_name="deepseek")

    assert "class LatentAttention" in code
    assert "kv_a_proj_with_mqa" in code
    assert "e_score_correction_bias" in code
    assert 'attention="mla"' in code


def test_a_plain_cnn_carries_no_transformer_machinery():
    """Helpers are emitted only where used, or every file becomes unreadable."""

    resolved, shapes, _params, _issues = resolve(TEMPLATES["small_cnn"].graph)
    code = emit_module(resolved, shapes, architecture_name="small cnn")

    for absent in ("LatentAttention", "SparseMoE", "llm_block", "ResNetStage"):
        assert absent not in code


@pytest.mark.parametrize(
    ("template_id", "present", "absent"),
    [
        # Gemma is dense grouped-query attention: no latent attention, no router.
        ("gemma3_4b", ["FamilyAttention", "GatedFeedForward"], ["LatentAttention", "SparseMoE"]),
        # GPT-2 predates rotary, normalizes with LayerNorm, and has a plain MLP.
        # The rotary helpers still ship — `FamilyAttention` branches on
        # `rope_theta` — but the call passes 0, which is the honest statement.
        ("gpt2_124m", ["FamilyAttention", "rope_theta=0.0"], ["class RMSNorm", "GatedFeedForward"]),
        # DeepSeek is the opposite: latent attention and a router, no plain GQA.
        ("deepseek_v3", ["LatentAttention", "SparseMoE"], ["FamilyAttention"]),
        ("mixtral_8x7b", ["FamilyAttention", "SparseMoE"], ["LatentAttention"]),
    ],
)
def test_helpers_follow_the_blocks_parameters_not_its_name(template_id, present, absent):
    """A generated file should describe the model it builds and nothing else.

    Emitting every family's machinery into every family's module would put
    DeepSeek's latent attention in a Gemma export and rotary helpers in a GPT-2
    one — code that reads as though the architecture uses them.
    """

    resolved, shapes, _params, issues = resolve(TEMPLATES[template_id].graph, 10)
    assert not errors(issues)
    code = emit_module(resolved, shapes, architecture_name=template_id)

    for expected in present:
        assert expected in code, (template_id, expected)
    for unexpected in absent:
        assert unexpected not in code, (template_id, unexpected)


def test_a_resnet_carries_no_llm_helpers():
    resolved, shapes, _params, _issues = resolve(TEMPLATES["resnet18"].graph)
    code = emit_module(resolved, shapes, architecture_name="resnet18")

    assert "def resnet_stage(" in code
    assert "FamilyAttention" not in code


# --- both emitters agree ----------------------------------------------------


def torch_parameter_count(module) -> int:
    model = module.build_model(4)
    # Buffers too: BatchNorm's running statistics are non-trainable weights that
    # Keras `count_params()` includes. `num_batches_tracked` is a torch-only
    # counter with no Keras counterpart, so it is excluded.
    return sum(parameter.numel() for parameter in model.parameters()) + sum(
        buffer.numel()
        for name, buffer in model.named_buffers()
        if not name.endswith("num_batches_tracked")
    )


def build_both(graph: ArchitectureGraph, label: str) -> tuple[int, int, int]:
    resolved, shapes, estimate, issues = resolve(graph)
    assert not errors(issues), errors(issues)

    keras_module = load(
        emit_module(resolved, shapes, architecture_name=label, default_num_classes=4),
        "generated_model.py",
    )
    torch_module = load(
        emit_torch_module(resolved, shapes, architecture_name=label, default_num_classes=4),
        "generated_model_torch.py",
    )
    return estimate, keras_module.build_model(4).count_params(), torch_parameter_count(torch_module)


@pytest.mark.slow
@pytest.mark.parametrize("block_type", VISION_TYPES)
def test_vision_blocks_build_identically_in_both_frameworks(block_type):
    if block_type == "vit_block":
        graph = ArchitectureGraph(
            nodes=[
                node("input", "input", shape="32,32,3"),
                node("patches", "patch_embedding", patch_size=8, embed_dim=64),
                node("block", "vit_block", **SMALL_VISION_OVERRIDES[block_type]),
                node("pool", "global_avg_pool1d"),
                node("head", "dense", units_from_dataset=True, activation="softmax"),
                node("output", "output"),
            ],
            edges=chain("input", "patches", "block", "pool", "head", "output"),
        )
    else:
        graph = image_graph(block_type, **SMALL_VISION_OVERRIDES[block_type])

    estimate, keras_count, torch_count = build_both(graph, block_type)

    assert estimate == keras_count == torch_count


@pytest.mark.slow
@pytest.mark.parametrize("block_type", LLM_TYPES)
def test_llm_blocks_build_identically_in_both_frameworks(block_type):
    estimate, keras_count, torch_count = build_both(
        sequence_graph(block_type, **SMALL_LLM_OVERRIDES[block_type]), block_type
    )

    assert estimate == keras_count == torch_count


@pytest.mark.slow
def test_the_new_primitives_build_in_both_frameworks():
    graphs = {
        "depthwise_se": ArchitectureGraph(
            nodes=[
                node("input", "input", shape="32,32,3"),
                node("dw", "depthwise_conv2d", kernel_size=3, strides=2, activation="relu"),
                node("se", "squeeze_excite", ratio=2),
                node("gap", "global_avg_pool2d"),
                node("head", "dense", units_from_dataset=True, activation="softmax"),
                node("output", "output"),
            ],
            edges=chain("input", "dw", "se", "gap", "head", "output"),
        ),
        "geglu": sequence_graph("geglu", hidden_dim=128),
        "mla": ArchitectureGraph(
            nodes=[
                node("input", "input", shape="64"),
                node("embed", "embedding", input_dim=500, output_dim=64),
                node("norm", "rms_norm"),
                node("attn", "mla_attention", num_heads=4, kv_lora_rank=32, q_lora_rank=48,
                     qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16),
                node("residual", "add"),
                node("head", "lm_head", vocab_size=500, vocab_from_dataset=False),
                node("output", "output"),
            ],
            edges=[
                *chain("input", "embed", "norm", "attn", "residual"),
                edge("embed", "residual"),
                *chain("residual", "head", "output"),
            ],
        ),
    }
    for label, graph in graphs.items():
        estimate, keras_count, torch_count = build_both(graph, label)
        assert estimate == keras_count == torch_count, label


@pytest.mark.slow
@pytest.mark.parametrize(
    ("template_id", "published"),
    [
        # Every published figure below is for a 1000-class ImageNet head; these
        # graphs carry a 10-class one, so the difference is exactly the head.
        ("resnet18", 11_689_512),
        ("resnet50", 25_557_032),
        ("vgg16", 138_357_544),
        ("mobilenet_v2", 3_504_872),
        ("vit_b16", 86_567_656),
        ("convnext_tiny", 28_589_128),
    ],
)
def test_vision_templates_reproduce_their_published_size(template_id, published):
    """The head is the only thing between these graphs and the real networks."""

    graph = TEMPLATES[template_id].graph
    _resolved, shapes, estimate, issues = resolve(graph, 10)
    assert not errors(issues), errors(issues)

    head = next(graph_node for graph_node in graph.nodes if graph_node.params.get("units_from_dataset"))
    penultimate = shapes[next(iter(edge.source for edge in graph.edges if edge.target == head.id))]
    width = penultimate[-1]
    # Swap this graph's 10-class head for the 1000-class one the paper reports.
    scaled = estimate - (width * 10 + 10) + (width * 1000 + 1000)

    assert abs(scaled - published) / published < 0.03, (template_id, scaled, published)
