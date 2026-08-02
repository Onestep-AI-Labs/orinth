"""Layered auto-layout for graphs that arrive without positions.

Imported architectures and starter templates have structure but no canvas
coordinates. This assigns them: longest-path layering for the column, median
of predecessor rows for the row. Pure and deterministic — the same graph
always lays out identically, so an import/export round trip is stable.

No dependency: a general graph-drawing library would be far more than this
needs, and the layouts it produces for a mostly-linear model graph are not
better than a layered pass.
"""

from __future__ import annotations

from app.schemas import ArchitectureGraph, ArchitectureNode

COLUMN_PITCH = 240.0
ROW_PITCH = 120.0
ORIGIN_X = 80.0
ORIGIN_Y = 80.0


def auto_layout(graph: ArchitectureGraph) -> ArchitectureGraph:
    """Return a copy of `graph` with every node positioned."""

    nodes = {node.id: node for node in graph.nodes}
    incoming: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for edge in graph.edges:
        if edge.source not in nodes or edge.target not in nodes or edge.source == edge.target:
            continue
        incoming[edge.target].append(edge.source)
        outgoing[edge.source].append(edge.target)

    layers = _layer_of(nodes, incoming, outgoing)
    rows = _rows(nodes, incoming, layers)
    positioned = [
        ArchitectureNode(
            id=node.id,
            type=node.type,
            label=node.label,
            position={
                "x": ORIGIN_X + layers[node.id] * COLUMN_PITCH,
                "y": ORIGIN_Y + rows[node.id] * ROW_PITCH,
            },
            params=node.params,
        )
        for node in graph.nodes
    ]
    return ArchitectureGraph(
        schema_version=graph.schema_version,
        nodes=positioned,
        edges=list(graph.edges),
        training_defaults=dict(graph.training_defaults),
    )


def _layer_of(
    nodes: dict[str, ArchitectureNode],
    incoming: dict[str, list[str]],
    outgoing: dict[str, list[str]],
) -> dict[str, int]:
    """Longest path from any source, computed iteratively.

    A cycle would never settle, so the relaxation is capped at one pass per
    node — layout must not hang on a graph the validator will reject anyway.
    """

    layers = {node_id: 0 for node_id in nodes}
    for _ in range(len(nodes)):
        changed = False
        for node_id in sorted(nodes):
            for source in incoming[node_id]:
                if layers[source] + 1 > layers[node_id]:
                    layers[node_id] = layers[source] + 1
                    changed = True
        if not changed:
            break
    return layers


def _rows(
    nodes: dict[str, ArchitectureNode],
    incoming: dict[str, list[str]],
    layers: dict[str, int],
) -> dict[str, float]:
    """Place each node near the median row of its predecessors, avoiding overlap."""

    by_layer: dict[int, list[str]] = {}
    for node_id in sorted(nodes, key=lambda key: (layers[key], key)):
        by_layer.setdefault(layers[node_id], []).append(node_id)

    rows: dict[str, float] = {}
    for layer in sorted(by_layer):
        taken: set[float] = set()
        for node_id in by_layer[layer]:
            sources = [rows[source] for source in incoming[node_id] if source in rows]
            preferred = _median(sources) if sources else 0.0
            row = round(preferred)
            while float(row) in taken:
                row += 1
            taken.add(float(row))
            rows[node_id] = float(row)
    return rows


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
