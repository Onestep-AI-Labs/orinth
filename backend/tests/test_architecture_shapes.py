"""Analytic shape inference and parameter estimation (phase 17).

These assertions encode the layer arithmetic the studio shows on the canvas.
`test_architecture_emit.py` cross-checks a sample of them against a real
TensorFlow build, which is what keeps this file honest.
"""

import pytest

from app.ml.architecture.graph import build_graph
from app.ml.architecture.shapes import infer_shapes
from app.ml.architecture.templates import edge, node
from app.schemas import ArchitectureGraph


def shapes_for(nodes, edges, num_classes=None):
    resolved = build_graph(ArchitectureGraph(nodes=nodes, edges=edges))
    shapes, params, issues = infer_shapes(resolved, num_classes)
    return shapes, params, [*resolved.issues, *issues]


def errors(issues):
    return [issue.message for issue in issues if issue.severity == "error"]


def chain_from_input(shape: str, *layers):
    """Wire Input → each layer in turn → Output, returning the shape map."""

    nodes = [node("in", "input", shape=shape)]
    ids = ["in"]
    for index, (node_type, params) in enumerate(layers):
        node_id = f"n{index}"
        nodes.append(node(node_id, node_type, **params))
        ids.append(node_id)
    nodes.append(node("out", "output"))
    ids.append("out")
    edges = [edge(source, target) for source, target in zip(ids, ids[1:], strict=False)]
    return shapes_for(nodes, edges)


@pytest.mark.parametrize(
    ("padding", "strides", "expected"),
    [
        ("same", 1, [32, 32, 8]),
        ("same", 2, [16, 16, 8]),
        ("valid", 1, [30, 30, 8]),
        ("valid", 2, [15, 15, 8]),
    ],
)
def test_conv2d_shape_arithmetic(padding, strides, expected):
    shapes, _params, issues = chain_from_input(
        "32,32,3",
        ("conv2d", {"filters": 8, "kernel_size": 3, "padding": padding, "strides": strides}),
    )

    assert not errors(issues)
    assert shapes["n0"] == expected


def test_pooling_defaults_its_stride_to_the_pool_size():
    shapes, _params, issues = chain_from_input(
        "32,32,3", ("max_pool2d", {"pool_size": 2, "strides": 0, "padding": "valid"})
    )

    assert not errors(issues)
    assert shapes["n0"] == [16, 16, 3]


def test_kernel_larger_than_the_input_is_reported():
    _shapes, _params, issues = chain_from_input(
        "4,4,3", ("conv2d", {"filters": 8, "kernel_size": 7, "padding": "valid"})
    )

    assert any("does not fit" in message for message in errors(issues))


def test_global_pooling_reduces_a_feature_map_to_a_vector():
    shapes, _params, issues = chain_from_input("16,16,64", ("global_avg_pool2d", {}))

    assert not errors(issues)
    assert shapes["n0"] == [64]


def test_flatten_multiplies_the_dimensions():
    shapes, _params, _issues = chain_from_input("7,7,64", ("flatten", {}))

    assert shapes["n0"] == [7 * 7 * 64]


def test_reshape_rejects_an_element_count_mismatch():
    _shapes, _params, issues = chain_from_input(
        "7,7,64", ("reshape", {"target_shape": "7,7,32"})
    )

    assert any("Cannot reshape" in message for message in errors(issues))


def test_reshape_accepts_a_matching_element_count():
    shapes, _params, issues = chain_from_input(
        "7,7,64", ("reshape", {"target_shape": "49,64"})
    )

    assert not errors(issues)
    assert shapes["n0"] == [49, 64]


def test_conv2d_transpose_upsamples():
    shapes, _params, _issues = chain_from_input(
        "8,8,16", ("conv2d_transpose", {"filters": 8, "kernel_size": 3, "strides": 2})
    )

    assert shapes["n0"] == [16, 16, 8]


def test_upsampling_and_padding_adjust_the_spatial_dimensions():
    shapes, _params, _issues = chain_from_input(
        "8,8,3", ("up_sampling2d", {"size": 2}), ("zero_padding2d", {"padding": 2})
    )

    assert shapes["n0"] == [16, 16, 3]
    assert shapes["n1"] == [20, 20, 3]


def test_embedding_adds_a_feature_axis_to_a_token_sequence():
    shapes, _params, _issues = chain_from_input(
        "200", ("embedding", {"input_dim": 5000, "output_dim": 64})
    )

    assert shapes["n0"] == [200, 64]


@pytest.mark.parametrize(
    ("return_sequences", "bidirectional", "expected"),
    [
        (False, False, [32]),
        (True, False, [200, 32]),
        (False, True, [64]),
        (True, True, [200, 64]),
    ],
)
def test_recurrent_shape_depends_on_sequences_and_direction(
    return_sequences, bidirectional, expected
):
    shapes, _params, issues = chain_from_input(
        "200",
        ("embedding", {"input_dim": 5000, "output_dim": 16}),
        (
            "lstm",
            {
                "units": 32,
                "return_sequences": return_sequences,
                "bidirectional": bidirectional,
            },
        ),
    )

    assert not errors(issues)
    assert shapes["n1"] == expected


def test_conv2d_on_a_rank_2_input_is_reported():
    _shapes, _params, issues = chain_from_input("200", ("conv2d", {"filters": 8}))

    assert any("rank-3 input" in message for message in errors(issues))


def test_group_norm_requires_channels_divisible_by_groups():
    _shapes, _params, issues = chain_from_input(
        "8,8,30", ("group_norm", {"groups": 32})
    )

    assert any("divide evenly" in message for message in errors(issues))


# --- merge nodes -----------------------------------------------------------


def merge_graph(merge_type: str, left_filters: int, right_filters: int, **params):
    nodes = [
        node("in", "input", shape="8,8,3"),
        node("a", "conv2d", filters=left_filters, kernel_size=1),
        node("b", "conv2d", filters=right_filters, kernel_size=1),
        node("m", merge_type, **params),
        node("gap", "global_avg_pool2d"),
        node("out", "output"),
    ]
    edges = [
        edge("in", "a"),
        edge("in", "b"),
        edge("a", "m"),
        edge("b", "m"),
        edge("m", "gap"),
        edge("gap", "out"),
    ]
    return shapes_for(nodes, edges)


def test_add_requires_matching_shapes():
    _shapes, _params, issues = merge_graph("add", 8, 16)

    assert any("matching input shapes" in message for message in errors(issues))


def test_add_passes_matching_shapes_through():
    shapes, _params, issues = merge_graph("add", 8, 8)

    assert not errors(issues)
    assert shapes["m"] == [8, 8, 8]


def test_concatenate_sums_the_axis_it_joins_on():
    shapes, _params, issues = merge_graph("concatenate", 8, 16, axis=-1)

    assert not errors(issues)
    assert shapes["m"] == [8, 8, 24]


def test_concatenate_rejects_a_mismatch_on_another_axis():
    nodes = [
        node("in", "input", shape="8,8,3"),
        node("a", "conv2d", filters=4, kernel_size=1, strides=1),
        node("b", "conv2d", filters=4, kernel_size=1, strides=2),
        node("m", "concatenate", axis=-1),
        node("gap", "global_avg_pool2d"),
        node("out", "output"),
    ]
    edges = [
        edge("in", "a"),
        edge("in", "b"),
        edge("a", "m"),
        edge("b", "m"),
        edge("m", "gap"),
        edge("gap", "out"),
    ]

    _shapes, _params, issues = shapes_for(nodes, edges)

    assert any("every other dimension to match" in message for message in errors(issues))


# --- unknown shapes and error containment ----------------------------------


def test_a_shape_error_does_not_cascade_into_its_successors():
    """One broken wire should produce one message, not one per downstream node."""

    _shapes, _params, issues = chain_from_input(
        "200",
        ("conv2d", {"filters": 8}),
        ("batch_norm", {}),
        ("dropout", {}),
        ("flatten", {}),
    )

    assert len(errors(issues)) == 1


def test_downstream_of_an_unknown_shape_stays_unknown_rather_than_erroring():
    shapes, _params, issues = chain_from_input(
        "200", ("conv2d", {"filters": 8}), ("dropout", {})
    )

    assert shapes["n1"] is None
    assert len(errors(issues)) == 1


def test_unparseable_input_shape_is_reported():
    _shapes, _params, issues = chain_from_input("not,a,shape", ("flatten", {}))

    assert any("comma-separated whole numbers" in message for message in errors(issues))


# --- parameter estimation --------------------------------------------------


def test_dense_parameter_count_includes_the_bias():
    _shapes, params, _issues = chain_from_input(
        "16,16,8", ("global_avg_pool2d", {}), ("dense", {"units": 4, "use_bias": True})
    )

    assert params == 8 * 4 + 4


def test_dense_head_sized_from_the_dataset_uses_the_supplied_class_count():
    resolved = build_graph(
        ArchitectureGraph(
            nodes=[
                node("in", "input", shape="8"),
                node("head", "dense", units_from_dataset=True, activation="softmax"),
                node("out", "output"),
            ],
            edges=[edge("in", "head"), edge("head", "out")],
        )
    )

    shapes, params, _issues = infer_shapes(resolved, num_classes=7)

    assert shapes["head"] == [7]
    assert params == 8 * 7 + 7


def test_estimate_is_unknown_when_the_class_count_is_not_supplied():
    resolved = build_graph(
        ArchitectureGraph(
            nodes=[
                node("in", "input", shape="8"),
                node("head", "dense", units_from_dataset=True),
                node("out", "output"),
            ],
            edges=[edge("in", "head"), edge("head", "out")],
        )
    )

    shapes, params, _issues = infer_shapes(resolved, num_classes=None)

    assert shapes["head"] == [None]
    assert params is None


def test_estimate_is_unknown_when_a_backbone_is_present():
    """Guessing a backbone's parameter count would be worse than saying so."""

    shapes, params, issues = chain_from_input(
        "224,224,3",
        ("pretrained_backbone", {"application": "EfficientNetB0", "pooling": "avg"}),
    )

    assert not errors(issues)
    assert shapes["n0"] == [1280]
    assert params is None
