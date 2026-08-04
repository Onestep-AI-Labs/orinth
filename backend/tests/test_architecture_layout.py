"""Auto-layout for graphs that arrive without canvas coordinates (phase 17)."""

from app.ml.architecture.layout import COLUMN_PITCH, ORIGIN_X, auto_layout
from app.ml.architecture.templates import chain, edge, node
from app.schemas import ArchitectureGraph


def positions(graph: ArchitectureGraph) -> dict[str, tuple[float, float]]:
    return {item.id: (item.position["x"], item.position["y"]) for item in graph.nodes}


def test_a_chain_lays_out_left_to_right_in_one_row():
    graph = auto_layout(
        ArchitectureGraph(
            nodes=[node("a", "input"), node("b", "flatten"), node("c", "output")],
            edges=chain("a", "b", "c"),
        )
    )

    placed = positions(graph)
    assert placed["a"] == (ORIGIN_X, placed["a"][1])
    assert placed["b"][0] == ORIGIN_X + COLUMN_PITCH
    assert placed["c"][0] == ORIGIN_X + 2 * COLUMN_PITCH
    assert len({y for _x, y in placed.values()}) == 1


def test_a_branch_puts_its_siblings_in_different_rows():
    graph = auto_layout(
        ArchitectureGraph(
            nodes=[
                node("in", "input"),
                node("a", "conv2d"),
                node("b", "conv2d"),
                node("join", "add"),
            ],
            edges=[edge("in", "a"), edge("in", "b"), edge("a", "join"), edge("b", "join")],
        )
    )

    placed = positions(graph)
    assert placed["a"][0] == placed["b"][0]
    assert placed["a"][1] != placed["b"][1]
    # The merge sits past both branches.
    assert placed["join"][0] > placed["a"][0]


def test_layout_is_deterministic_so_an_import_round_trip_is_stable():
    graph = ArchitectureGraph(
        nodes=[node("in", "input"), node("a", "conv2d"), node("b", "conv2d"), node("j", "add")],
        edges=[edge("in", "a"), edge("in", "b"), edge("a", "j"), edge("b", "j")],
    )

    assert positions(auto_layout(graph)) == positions(auto_layout(graph))


def test_layout_does_not_hang_on_a_cyclic_graph():
    """Layout must terminate on a graph the validator will reject anyway."""

    graph = auto_layout(
        ArchitectureGraph(
            nodes=[node("a", "conv2d"), node("b", "conv2d"), node("c", "conv2d")],
            edges=[edge("a", "b"), edge("b", "c"), edge("c", "a")],
        )
    )

    assert len(graph.nodes) == 3


def test_layout_preserves_edges_params_and_training_defaults():
    original = ArchitectureGraph(
        nodes=[node("a", "input", shape="8,8,3"), node("b", "output")],
        edges=chain("a", "b"),
        training_defaults={"epochs": 7},
    )

    laid_out = auto_layout(original)

    assert laid_out.nodes[0].params["shape"] == "8,8,3"
    assert [item.id for item in laid_out.edges] == [item.id for item in original.edges]
    assert laid_out.training_defaults == {"epochs": 7}


def test_an_empty_graph_lays_out_to_nothing():
    assert auto_layout(ArchitectureGraph()).nodes == []
