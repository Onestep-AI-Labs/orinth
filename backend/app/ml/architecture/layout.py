"""Layered auto-layout for graphs that arrive without positions.

Imported architectures and starter templates have structure but no canvas
coordinates. This assigns them: longest-path layering for the column, and a row
that makes the graph's shape visible rather than merely legal.

Three rules, in the order they are applied:

1. **Longest path** sets the column.
2. **Siblings fan out symmetrically** around their shared parent, so an
   Inception module's four branches straddle the stem rather than hanging off
   the bottom of it.
3. **A bypass pushes the path it bypasses off the main row.** An edge that skips
   more than one column — a residual shortcut, a U-Net skip — is drawn straight
   while the layers it goes around bow away from it. Without this a residual
   block lays out as a single line with the skip edge hidden underneath the
   nodes it is supposed to be routing around, which is a picture of the wrong
   architecture.

Pure and deterministic, so an import/export round trip is stable, and
mirrored by `frontend/features/models/architectures/auto-layout.ts` so the
studio's Tidy produces the same arrangement the template shipped with.
"""

from __future__ import annotations

import math

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
    rows = _rows(nodes, incoming, outgoing, layers)
    lift = min(rows.values()) if rows else 0.0
    positioned = [
        ArchitectureNode(
            id=node.id,
            type=node.type,
            label=node.label,
            position={
                "x": ORIGIN_X + layers[node.id] * COLUMN_PITCH,
                "y": ORIGIN_Y + (rows[node.id] - lift) * ROW_PITCH,
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


def _bypass_offsets(
    nodes: dict[str, ArchitectureNode],
    incoming: dict[str, list[str]],
    outgoing: dict[str, list[str]],
    layers: dict[str, int],
) -> dict[str, int]:
    """How far off the main row each node sits because something skips past it.

    For every edge spanning more than one column, the nodes it skips over are
    pushed away by one row so the shortcut can be drawn as a straight line
    between them. Overlapping spans stack, so two nested skips do not land on
    top of each other.
    """

    spans: list[tuple[int, int, str, str]] = []
    for source in sorted(nodes):
        for target in sorted(outgoing[source]):
            if layers[target] - layers[source] > 1:
                spans.append((layers[source], layers[target], source, target))
    if not spans:
        return {}

    offsets: dict[str, int] = {}
    # Widest span first, so an outer skip claims row 1 and a nested one row 2 —
    # the arrangement a reader expects from a diagram of nested blocks.
    claimed: list[tuple[int, int, int]] = []
    for start, end, source, target in sorted(
        spans, key=lambda span: (span[0] - span[1], span[0], span[2])
    ):
        depth = 1 + sum(
            1 for other_start, other_end, _ in claimed if other_start < end and start < other_end
        )
        claimed.append((start, end, depth))
        for node_id in _between(source, target, incoming, outgoing, layers):
            offsets[node_id] = max(offsets.get(node_id, 0), depth)
    return offsets


def _between(
    source: str,
    target: str,
    incoming: dict[str, list[str]],
    outgoing: dict[str, list[str]],
    layers: dict[str, int],
) -> set[str]:
    """Nodes strictly inside the span: reachable from `source` and reaching `target`."""

    forward: set[str] = set()
    stack = list(outgoing[source])
    while stack:
        node_id = stack.pop()
        if node_id in forward or layers[node_id] >= layers[target]:
            continue
        forward.add(node_id)
        stack.extend(outgoing[node_id])

    inside: set[str] = set()
    stack = list(incoming[target])
    while stack:
        node_id = stack.pop()
        if node_id in inside or node_id not in forward:
            continue
        inside.add(node_id)
        stack.extend(incoming[node_id])
    return inside


def _rows(
    nodes: dict[str, ArchitectureNode],
    incoming: dict[str, list[str]],
    outgoing: dict[str, list[str]],
    layers: dict[str, int],
) -> dict[str, float]:
    """Place each node near its predecessors, fanning siblings and skips apart."""

    offsets = _bypass_offsets(nodes, incoming, outgoing, layers)

    by_layer: dict[int, list[str]] = {}
    for node_id in sorted(nodes, key=lambda key: (layers[key], key)):
        by_layer.setdefault(layers[node_id], []).append(node_id)

    # `base` ignores the bypass offsets so they never compound down a chain: a
    # five-layer detour is one row off the main line, not five.
    base: dict[str, float] = {}
    rows: dict[str, float] = {}
    for layer in sorted(by_layer):
        taken: set[float] = set()
        # Group by shared parent set so siblings can be centred together.
        siblings: dict[str, list[str]] = {}
        for node_id in by_layer[layer]:
            key = "|".join(sorted(incoming[node_id]))
            siblings.setdefault(key, []).append(node_id)

        for group in siblings.values():
            sources = [base[source] for source in incoming[group[0]] if source in base]
            centre = _median(sources) if sources else 0.0
            # Odd counts sit one dead centre; even counts straddle it.
            spread = (len(group) - 1) / 2
            for index, node_id in enumerate(group):
                row = _round_half_up(centre + index - spread) + offsets.get(node_id, 0)
                while row in taken:
                    row += 1
                taken.add(row)
                rows[node_id] = row
                base[node_id] = row - offsets.get(node_id, 0)
    return rows


def _round_half_up(value: float) -> float:
    """Round .5 away from zero's neighbour, matching JavaScript's `Math.round`.

    Python's built-in `round` breaks ties to even, so four siblings straddling a
    parent land on rows -2, 0, 1, 2 instead of -1, 0, 1, 2. The frontend's Tidy
    is JavaScript; the two layouts have to agree or pressing Tidy on a freshly
    opened template moves every node.
    """

    return math.floor(value + 0.5)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
