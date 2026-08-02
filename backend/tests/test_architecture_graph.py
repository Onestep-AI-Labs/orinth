"""Structural validation of the architecture IR (phase 17).

Covers what `build_graph` must catch before any code is generated: cycles,
arity, dangling edges, unreachable nodes, and param coercion.
"""

import pytest

from app.ml.architecture.catalog import CATEGORY_ORDER, NODE_SPECS, node_catalog
from app.ml.architecture.emit_keras import EMITTERS
from app.ml.architecture.graph import build_graph, parse_shape
from app.ml.architecture.shapes import SHAPE_RULES

# `node` and `edge` are the same builders the starter templates are declared
# with, so a graph in a test is constructed exactly like a real one.
from app.ml.architecture.templates import edge, node
from app.schemas import ArchitectureEdge, ArchitectureGraph, ArchitectureNode


def linear_graph(*, shape: str = "8,8,3") -> ArchitectureGraph:
    return ArchitectureGraph(
        nodes=[
            node("in", "input", shape=shape),
            node("gap", "global_avg_pool2d"),
            node("head", "dense", units=2, activation="softmax"),
            node("out", "output"),
        ],
        edges=[edge("in", "gap"), edge("gap", "head"), edge("head", "out")],
    )


def messages(resolved) -> list[str]:
    return [issue.message for issue in resolved.issues]


def test_registries_cover_exactly_the_same_node_types():
    """The catalog, shape rules, and emitters must never drift apart.

    A type in the catalog but missing from either registry is a node the user
    can drop on the canvas and then cannot build — the failure mode this
    three-way check exists to make impossible.
    """

    assert set(NODE_SPECS) == set(SHAPE_RULES) == set(EMITTERS)


def test_every_catalog_category_is_in_the_display_order():
    for spec in NODE_SPECS.values():
        assert spec.category in CATEGORY_ORDER


def test_node_catalog_filters_by_task_type_but_keeps_universal_nodes():
    text_nodes = {spec.type for spec in node_catalog("text_classification")}
    image_nodes = {spec.type for spec in node_catalog("classification")}

    # Embedding declares text task types; Conv2D declares none, so it is offered everywhere.
    assert "embedding" in text_nodes
    assert "embedding" not in image_nodes
    assert {"conv2d", "dense", "input", "output"} <= text_nodes & image_nodes


def test_linear_graph_resolves_in_topological_order():
    resolved = build_graph(linear_graph())

    assert resolved.ok
    assert resolved.order == ["in", "gap", "head", "out"]
    assert resolved.input_ids == ["in"]
    assert resolved.output_ids == ["out"]


def test_variable_names_are_assigned_in_topological_order():
    resolved = build_graph(linear_graph())

    assert [resolved.nodes[node_id].var_name for node_id in resolved.order] == [
        "x_input_1",
        "x_global_avg_pool2d_1",
        "x_dense_1",
        "x_output_1",
    ]


def test_declaration_order_does_not_change_the_resolved_order():
    """Codegen must be reproducible, so ordering cannot depend on node order."""

    graph = linear_graph()
    shuffled = ArchitectureGraph(
        nodes=list(reversed(graph.nodes)), edges=list(reversed(graph.edges))
    )

    assert build_graph(shuffled).order == build_graph(graph).order


def test_cycle_is_reported_against_its_nodes():
    graph = linear_graph()
    graph.edges.append(edge("head", "gap"))

    resolved = build_graph(graph)

    assert not resolved.ok
    assert any("part of a loop" in message for message in messages(resolved))


def test_self_loop_is_rejected():
    graph = linear_graph()
    graph.edges.append(edge("gap", "gap"))

    resolved = build_graph(graph)

    assert not resolved.ok
    assert any("cannot connect to itself" in message for message in messages(resolved))


def test_edge_to_a_missing_node_is_reported_and_dropped():
    graph = linear_graph()
    graph.edges.append(ArchitectureEdge(id="e_ghost", source="ghost", target="gap"))

    resolved = build_graph(graph)

    assert not resolved.ok
    assert any("not on the canvas" in message for message in messages(resolved))
    assert resolved.nodes["gap"].inputs == ["in"]


def test_duplicate_node_id_is_reported_once():
    graph = linear_graph()
    graph.nodes.append(node("gap", "global_avg_pool2d"))

    resolved = build_graph(graph)

    assert not resolved.ok
    assert sum("Duplicate node id" in message for message in messages(resolved)) == 1


def test_unknown_node_type_is_reported_without_dropping_the_rest():
    graph = linear_graph()
    graph.nodes.append(ArchitectureNode(id="mystery", type="quantum_layer"))

    resolved = build_graph(graph)

    assert not resolved.ok
    assert any("Unknown node type" in message for message in messages(resolved))
    assert "gap" in resolved.nodes


def test_merge_node_below_its_minimum_arity_is_reported():
    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("sum", "add"),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=[edge("in", "sum"), edge("sum", "gap"), edge("gap", "out")],
    )

    resolved = build_graph(graph)

    assert not resolved.ok
    assert any("at least 2 inputs" in message for message in messages(resolved))


def test_single_input_node_rejects_a_second_connection():
    graph = ArchitectureGraph(
        nodes=[
            node("in_a", "input", shape="8,8,3"),
            node("in_b", "input", shape="8,8,3"),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=[edge("in_a", "gap"), edge("in_b", "gap"), edge("gap", "out")],
    )

    resolved = build_graph(graph)

    assert not resolved.ok
    assert any("at most 1 input" in message for message in messages(resolved))


def test_node_not_reaching_the_output_is_a_warning_not_an_error():
    graph = linear_graph()
    graph.nodes.append(node("stray", "dropout"))

    resolved = build_graph(graph)

    assert resolved.ok
    assert any("will not be part of the model" in message for message in messages(resolved))
    assert "stray" not in resolved.order


def test_missing_input_and_output_nodes_are_both_reported():
    graph = ArchitectureGraph(nodes=[node("solo", "dropout")], edges=[])

    resolved = build_graph(graph)

    assert not resolved.ok
    assert any("Add an Input node" in message for message in messages(resolved))
    assert any("Add an Output node" in message for message in messages(resolved))


def test_multiple_output_nodes_are_rejected():
    graph = linear_graph()
    graph.nodes.append(node("out_2", "output"))
    graph.edges.append(edge("gap", "out_2"))

    resolved = build_graph(graph)

    assert not resolved.ok
    assert any("more than one Output node" in message for message in messages(resolved))


def test_empty_canvas_is_reported_plainly():
    resolved = build_graph(ArchitectureGraph())

    assert not resolved.ok
    assert "The canvas is empty." in messages(resolved)


def test_duplicate_edge_between_the_same_pair_is_ignored():
    graph = linear_graph()
    graph.edges.append(ArchitectureEdge(id="e_dup", source="in", target="gap"))

    resolved = build_graph(graph)

    assert resolved.ok
    assert resolved.nodes["gap"].inputs == ["in"]


def test_multi_input_order_follows_edge_declaration_order():
    """Concatenate and Subtract are not commutative, so input order must be stable."""

    graph = ArchitectureGraph(
        nodes=[
            node("in", "input", shape="8,8,3"),
            node("a", "conv2d", filters=4),
            node("b", "conv2d", filters=4),
            node("join", "concatenate", axis=-1),
            node("gap", "global_avg_pool2d"),
            node("out", "output"),
        ],
        edges=[
            edge("in", "a"),
            edge("in", "b"),
            edge("b", "join"),
            edge("a", "join"),
            edge("join", "gap"),
            edge("gap", "out"),
        ],
    )

    resolved = build_graph(graph)

    assert resolved.nodes["join"].inputs == ["b", "a"]


# --- param resolution ------------------------------------------------------


def test_params_fall_back_to_catalog_defaults_when_absent():
    graph = ArchitectureGraph(
        nodes=[ArchitectureNode(id="d", type="dropout"), node("out", "output")],
        edges=[edge("d", "out")],
    )

    resolved = build_graph(graph)

    assert resolved.nodes["d"].params["rate"] == 0.2


def test_out_of_range_number_is_clamped_with_a_warning():
    graph = linear_graph()
    graph.nodes.append(node("drop", "dropout", rate=5.0))
    graph.edges.append(edge("gap", "drop"))

    resolved = build_graph(graph)

    assert resolved.nodes["drop"].params["rate"] == 0.9
    assert any("lowered to the maximum" in message for message in messages(resolved))


def test_invalid_select_value_falls_back_to_the_default():
    graph = linear_graph()
    graph.nodes.append(node("conv", "conv2d", padding="diagonal"))
    graph.edges.append(edge("in", "conv"))

    resolved = build_graph(graph)

    assert resolved.nodes["conv"].params["padding"] == "same"


def test_unknown_params_are_ignored_so_older_graphs_still_open():
    graph = ArchitectureGraph(
        nodes=[
            ArchitectureNode(id="d", type="dropout", params={"rate": 0.5, "retired_knob": 7}),
            node("out", "output"),
        ],
        edges=[edge("d", "out")],
    )

    resolved = build_graph(graph)

    assert resolved.nodes["d"].params == {"rate": 0.5}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("224,224,3", [224, 224, 3]),
        (" 200 ", [200]),
        ("none,128", [None, 128]),
        ([32, 32, 1], [32, 32, 1]),
        ("", None),
        ("64,abc", None),
        (None, None),
    ],
)
def test_parse_shape(raw, expected):
    assert parse_shape(raw) == expected
