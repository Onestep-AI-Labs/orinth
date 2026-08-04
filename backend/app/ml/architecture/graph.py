"""Architecture graph IR: parsing, structural validation, and ordering.

The client posts an `ArchitectureGraph` of loosely-typed nodes and edges. This
module resolves it into a `ResolvedGraph` — nodes matched to their catalog
spec, params coerced, edges indexed, and a deterministic topological order —
collecting every structural problem as an `ArchitectureIssue` rather than
raising on the first one, so the studio can show all of them at once.

Nothing here imports TensorFlow. Structural validation runs on the request
path; the authoritative build happens in a subprocess (see the compile runner).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from app.ml.architecture.catalog import NODE_SPECS
from app.schemas import (
    AdvancedParameterSpec,
    ArchitectureEdge,
    ArchitectureGraph,
    ArchitectureIssue,
    NodeSpec,
)

# Node types with special standing in the graph, referenced by the validator,
# the shape pass, and the emitter alike.
INPUT_TYPE = "input"
OUTPUT_TYPE = "output"


@dataclass
class ResolvedNode:
    """A canvas node matched to its catalog spec, with params coerced."""

    id: str
    type: str
    label: str
    spec: NodeSpec
    params: dict[str, Any]
    # Source node ids in edge-declaration order. Order matters for
    # non-commutative multi-input nodes (Concatenate, Subtract).
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    # Generated Python identifier / Keras layer name. Assigned in topological
    # order so regenerating an unchanged graph is byte-identical.
    var_name: str = ""


@dataclass
class ResolvedGraph:
    nodes: dict[str, ResolvedNode] = field(default_factory=dict)
    # Topological order over `nodes`, restricted to nodes that reach the output.
    order: list[str] = field(default_factory=list)
    input_ids: list[str] = field(default_factory=list)
    output_ids: list[str] = field(default_factory=list)
    issues: list[ArchitectureIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def errors(self) -> list[ArchitectureIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]


def build_graph(graph: ArchitectureGraph) -> ResolvedGraph:
    """Resolve, validate, and order a posted graph.

    Always returns a `ResolvedGraph`; check `.ok` before emitting code. A graph
    with errors still carries whatever structure was recoverable, so the studio
    can render shapes for the healthy part of a broken canvas.
    """

    resolved = ResolvedGraph()
    seen_ids: set[str] = set()

    for node in graph.nodes:
        if node.id in seen_ids:
            resolved.issues.append(
                ArchitectureIssue(
                    severity="error",
                    node_id=node.id,
                    message=f"Duplicate node id {node.id!r}.",
                )
            )
            continue
        seen_ids.add(node.id)
        spec = NODE_SPECS.get(node.type)
        if spec is None:
            resolved.issues.append(
                ArchitectureIssue(
                    severity="error",
                    node_id=node.id,
                    message=(
                        f"Unknown node type {node.type!r}. It was kept with its "
                        "settings intact but cannot be built."
                    ),
                )
            )
            continue
        params, param_issues = resolve_params(spec, node.params, node.id)
        resolved.issues.extend(param_issues)
        resolved.nodes[node.id] = ResolvedNode(
            id=node.id,
            type=node.type,
            label=node.label or spec.name,
            spec=spec,
            params=params,
        )

    _index_edges(graph.edges, resolved)
    _classify_terminals(resolved)
    # Arity is only checked on nodes that reach the Output node. A node the
    # user has dropped but not wired up yet is incomplete, not broken — it is
    # excluded from codegen and reported as a warning, so a half-placed node
    # cannot block saving or training an otherwise-complete model.
    contributing = _nodes_reaching_outputs(resolved)
    _check_arity(resolved, contributing)
    _check_classifier_head(resolved, contributing)
    _check_batch_norm_momentum(resolved, contributing)
    resolved.order = _topological_order(resolved, contributing)
    _assign_var_names(resolved)
    return resolved


def _index_edges(edges: list[ArchitectureEdge], resolved: ResolvedGraph) -> None:
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        source = resolved.nodes.get(edge.source)
        target = resolved.nodes.get(edge.target)
        if source is None or target is None:
            resolved.issues.append(
                ArchitectureIssue(
                    severity="error",
                    edge_id=edge.id,
                    message="Edge references a node that is not on the canvas.",
                )
            )
            continue
        if edge.source == edge.target:
            resolved.issues.append(
                ArchitectureIssue(
                    severity="error",
                    node_id=edge.source,
                    edge_id=edge.id,
                    message="A node cannot connect to itself.",
                )
            )
            continue
        # A repeated source→target pair on the same port is a duplicate wire,
        # not a legitimate second input: it would emit the same tensor twice.
        key = (edge.source, edge.target, edge.target_port)
        if key in seen:
            resolved.issues.append(
                ArchitectureIssue(
                    severity="warning",
                    edge_id=edge.id,
                    message="Duplicate connection ignored.",
                )
            )
            continue
        seen.add(key)
        target.inputs.append(edge.source)
        source.outputs.append(edge.target)


def _check_arity(resolved: ResolvedGraph, contributing: set[str]) -> None:
    for node in resolved.nodes.values():
        if node.id not in contributing:
            continue
        count = len(node.inputs)
        spec = node.spec
        if count < spec.min_inputs:
            resolved.issues.append(
                ArchitectureIssue(
                    severity="error",
                    node_id=node.id,
                    message=(
                        f"{spec.name} needs at least {spec.min_inputs} "
                        f"input{'' if spec.min_inputs == 1 else 's'} but has {count}."
                    ),
                )
            )
        if spec.max_inputs >= 0 and count > spec.max_inputs:
            resolved.issues.append(
                ArchitectureIssue(
                    severity="error",
                    node_id=node.id,
                    message=(
                        f"{spec.name} accepts at most {spec.max_inputs} "
                        f"input{'' if spec.max_inputs == 1 else 's'} but has {count}."
                    ),
                )
            )


def _check_classifier_head(resolved: ResolvedGraph, contributing: set[str]) -> None:
    """Warn about the two head mistakes that only surface once training starts.

    Both are invisible on the canvas — the graph compiles, the shapes resolve,
    `model.summary()` looks right — and both are fatal to a run:

    * A fixed `units` head is sized for one dataset. The image runner refuses
      the run outright when the count does not match the label set, so a graph
      built from a blank canvas fails at launch rather than at edit time.
    * The runner compiles categorical cross-entropy over probabilities. A head
      left on the catalog's default `linear` activation hands it raw logits,
      which Keras clips and renormalises instead of rejecting: the run starts,
      the loss parks at ln(num_classes), and accuracy never leaves chance.

    Only the layer feeding the Output node is checked, so an intermediate Dense
    inside the model is left alone.
    """

    if len(resolved.output_ids) != 1:
        return
    output = resolved.nodes[resolved.output_ids[0]]
    heads = [
        resolved.nodes[node_id]
        for node_id in output.inputs
        if node_id in contributing and resolved.nodes[node_id].type == "dense"
    ]
    for head in heads:
        if not head.params.get("units_from_dataset"):
            resolved.issues.append(
                ArchitectureIssue(
                    severity="warning",
                    node_id=head.id,
                    message=(
                        f"Output head is fixed at {head.params.get('units')} units. Turn on "
                        '"Units = dataset class count" so it matches whatever dataset you '
                        "train on — image training refuses a run whose head and label set "
                        "disagree."
                    ),
                )
            )
        if head.params.get("activation") != "softmax":
            resolved.issues.append(
                ArchitectureIssue(
                    severity="warning",
                    node_id=head.id,
                    message=(
                        f"Output head activation is {head.params.get('activation')!r}. Training "
                        "compiles categorical cross-entropy over probabilities, so set it to "
                        "softmax — logits here train at chance without erroring."
                    ),
                )
            )


# Above this, BatchNorm's running statistics need more steps than a run on a
# few hundred images will ever take. Keras defaults to 0.99, which assumes
# thousands of steps per epoch.
_SLOW_BATCH_NORM_MOMENTUM = 0.95


def _check_batch_norm_momentum(resolved: ResolvedGraph, contributing: set[str]) -> None:
    """Warn when BatchNorm's running statistics will not converge in time.

    The failure this catches is the most confusing one the studio can produce,
    because nothing about it looks like a defect: training loss falls, training
    accuracy climbs past 0.7, and validation sits flat at exactly 1/num_classes
    with a loss of ln(num_classes) forever.

    BatchNorm normalises with *batch* statistics while training and with running
    averages at inference. Those averages move by `1 - momentum` per step, so at
    0.99 they need thousands of steps. A 210-image dataset at batch 32 is seven
    steps an epoch, and after eleven epochs the averages are less than halfway
    from their initial mean 0 / variance 1 — so at inference the network is
    normalised by numbers that describe nothing, and its output collapses.

    A warning rather than an error: the graph is valid, and on a large dataset
    0.99 is the right choice. Saved graphs carry an explicit momentum, so
    lowering the catalog default cannot reach them — this can.
    """

    for node_id in sorted(contributing):
        node = resolved.nodes[node_id]
        if node.type != "batch_norm":
            continue
        momentum = node.params.get("momentum")
        if not isinstance(momentum, (int, float)) or momentum <= _SLOW_BATCH_NORM_MOMENTUM:
            continue
        resolved.issues.append(
            ArchitectureIssue(
                severity="warning",
                node_id=node.id,
                message=(
                    f"Momentum {momentum:g} needs thousands of training steps before this "
                    "layer's running statistics are usable. On a small dataset the run "
                    "trains normally and validates at chance. Use 0.9 unless the dataset "
                    "is large."
                ),
            )
        )


def _classify_terminals(resolved: ResolvedGraph) -> None:
    resolved.input_ids = sorted(
        node.id for node in resolved.nodes.values() if node.type == INPUT_TYPE
    )
    resolved.output_ids = sorted(
        node.id for node in resolved.nodes.values() if node.type == OUTPUT_TYPE
    )
    if not resolved.nodes:
        resolved.issues.append(
            ArchitectureIssue(severity="error", message="The canvas is empty.")
        )
        return
    if not resolved.input_ids:
        resolved.issues.append(
            ArchitectureIssue(
                severity="error", message="Add an Input node — every model needs one."
            )
        )
    if not resolved.output_ids:
        resolved.issues.append(
            ArchitectureIssue(
                severity="error",
                message="Add an Output node to mark where the model ends.",
            )
        )
    elif len(resolved.output_ids) > 1:
        resolved.issues.append(
            ArchitectureIssue(
                severity="error",
                message=(
                    "This graph has more than one Output node. "
                    "Multi-output models are not supported yet."
                ),
            )
        )


def _topological_order(resolved: ResolvedGraph, contributing: set[str]) -> list[str]:
    """Kahn's algorithm over the nodes that actually reach an Output node.

    Ties break on node id so the order — and therefore generated code — is
    reproducible for a given graph regardless of node declaration order.
    """

    indegree = {
        node_id: sum(1 for source in resolved.nodes[node_id].inputs if source in contributing)
        for node_id in contributing
    }
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        node_id = heapq.heappop(ready)
        order.append(node_id)
        for target in sorted(set(resolved.nodes[node_id].outputs)):
            if target not in indegree:
                continue
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)

    if len(order) < len(contributing):
        stuck = sorted(set(contributing) - set(order))
        for node_id in stuck:
            resolved.issues.append(
                ArchitectureIssue(
                    severity="error",
                    node_id=node_id,
                    message="This node is part of a loop. Model graphs must be acyclic.",
                )
            )
        return []

    unreachable = sorted(set(resolved.nodes) - contributing)
    for node_id in unreachable:
        resolved.issues.append(
            ArchitectureIssue(
                severity="warning",
                node_id=node_id,
                message="Not connected to the Output node — it will not be part of the model.",
            )
        )
    return order


def _nodes_reaching_outputs(resolved: ResolvedGraph) -> set[str]:
    """Nodes with a path to an Output node, found by walking edges backwards.

    Cycles upstream of the output are included so `_topological_order` can
    report them; a cycle with no path to the output is merely unreachable and
    reported as such instead.
    """

    contributing: set[str] = set()
    stack = list(resolved.output_ids)
    while stack:
        node_id = stack.pop()
        if node_id in contributing:
            continue
        contributing.add(node_id)
        stack.extend(resolved.nodes[node_id].inputs)
    return contributing


def _assign_var_names(resolved: ResolvedGraph) -> None:
    """Name each node `x_<type>_<n>` in topological order.

    The same name is used for the Python variable and the Keras layer name, so
    a layer in `model.summary()` maps back to a node on the canvas by eye.
    """

    counters: dict[str, int] = {}
    for node_id in resolved.order:
        node = resolved.nodes[node_id]
        counters[node.type] = counters.get(node.type, 0) + 1
        node.var_name = f"x_{node.type}_{counters[node.type]}"


def resolve_params(
    spec: NodeSpec, raw: dict[str, Any], node_id: str
) -> tuple[dict[str, Any], list[ArchitectureIssue]]:
    """Coerce a node's raw params against its spec, filling in defaults.

    Unknown keys are ignored rather than rejected — a graph saved against an
    older catalog must still open. Out-of-range numbers are clamped with a
    warning instead of failing the whole graph.
    """

    values: dict[str, Any] = {}
    issues: list[ArchitectureIssue] = []
    for param in spec.params:
        if param.key not in raw or raw[param.key] is None:
            values[param.key] = param.default
            continue
        value, issue = _coerce(param, raw[param.key], node_id)
        values[param.key] = value
        if issue is not None:
            issues.append(issue)
    return values, issues


def _coerce(
    param: AdvancedParameterSpec, value: Any, node_id: str
) -> tuple[Any, ArchitectureIssue | None]:
    def invalid(reason: str) -> tuple[Any, ArchitectureIssue]:
        return param.default, ArchitectureIssue(
            severity="warning",
            node_id=node_id,
            message=f"{param.label}: {reason} Using the default ({param.default!r}).",
        )

    if param.type in {"int", "float"}:
        try:
            number = int(value) if param.type == "int" else float(value)
        except (TypeError, ValueError):
            return invalid(f"{value!r} is not a number.")
        if param.min is not None and number < param.min:
            return (
                type(number)(param.min),
                ArchitectureIssue(
                    severity="warning",
                    node_id=node_id,
                    message=f"{param.label}: raised to the minimum of {param.min:g}.",
                ),
            )
        if param.max is not None and number > param.max:
            return (
                type(number)(param.max),
                ArchitectureIssue(
                    severity="warning",
                    node_id=node_id,
                    message=f"{param.label}: lowered to the maximum of {param.max:g}.",
                ),
            )
        return number, None
    if param.type == "bool":
        return bool(value), None
    if param.type == "select":
        if param.options and value not in param.options:
            return invalid(f"{value!r} is not one of {param.options}.")
        return value, None
    if param.type == "multiselect":
        if not isinstance(value, list):
            return invalid("expected a list.")
        return [item for item in value if not param.options or item in param.options], None
    return str(value), None


def parse_shape(raw: Any) -> list[int | None] | None:
    """Parse a shape param such as ``"224,224,3"`` into ``[224, 224, 3]``.

    Shape params ride as text because `AdvancedParameterSpec` has no tuple
    type and the inspector renders it generically. `None` means unparseable,
    which the shape pass propagates as an unknown shape rather than an error.
    """

    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        parts: list[Any] = list(raw)
    else:
        parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        return None
    dims: list[int | None] = []
    for part in parts:
        text = str(part).strip().lower()
        if text in {"none", "null", "?", "-1"}:
            dims.append(None)
            continue
        try:
            dims.append(int(text))
        except (TypeError, ValueError):
            return None
    return dims
