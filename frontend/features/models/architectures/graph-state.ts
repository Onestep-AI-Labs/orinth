import type { Edge, Node } from "@xyflow/react";
import type {
  ArchitectureEdge,
  ArchitectureGraph,
  ArchitectureIssue,
  ArchitectureNode,
  NodeSpec
} from "@/types/api";

/** What a canvas node carries beyond its position. */
export type ArchNodeData = {
  spec: NodeSpec;
  params: Record<string, unknown>;
  label: string;
  shape: (number | null)[] | null;
  issues: ArchitectureIssue[];
};

export type ArchFlowNode = Node<ArchNodeData, "arch">;

/**
 * Group frames ride in `training_defaults.groups` rather than in `nodes`.
 *
 * They are annotations — no tensors, no code, no edges — so putting them in
 * the node list would mean every backend validator, shape rule, and emitter
 * had to learn to skip them. Keeping them out of the graph proper means the
 * backend never has to know they exist.
 */
export type StoredGroup = {
  id: string;
  title: string;
  tint: number;
  x: number;
  y: number;
  width: number;
  height: number;
  /**
   * The title's own placement and styling.
   *
   * Optional because a graph saved before the title became a movable text box
   * has none of them, and an architecture must always reopen. `titleTint` of
   * `-1` means neutral ink; anything else indexes `GROUP_TINTS`, so a title is
   * a token colour like everything else rather than a stored literal.
   */
  titleX?: number;
  titleY?: number;
  titleSize?: number;
  titleTint?: number;
};

/**
 * Convert the API graph into React Flow's node/edge arrays.
 *
 * A node whose type is not in the catalog still renders — it becomes a
 * placeholder carrying its stored params, so opening a graph saved against a
 * newer build shows what is missing rather than silently dropping it.
 */
export function toFlow(
  graph: ArchitectureGraph,
  specs: Map<string, NodeSpec>
): { nodes: ArchFlowNode[]; edges: Edge[] } {
  const nodes = graph.nodes.map<ArchFlowNode>((node) => ({
    id: node.id,
    type: "arch",
    position: { x: node.position?.x ?? 0, y: node.position?.y ?? 0 },
    data: {
      spec: specs.get(node.type) ?? placeholderSpec(node.type),
      params: { ...node.params },
      label: node.label ?? specs.get(node.type)?.name ?? node.type,
      shape: null,
      issues: []
    }
  }));
  const edges = graph.edges.map<Edge>((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target
  }));
  return { nodes, edges };
}

/** Convert React Flow state back into the API graph shape for save/validate. */
export function toGraph(
  nodes: ArchFlowNode[],
  edges: Edge[],
  trainingDefaults: Record<string, unknown> = {}
): ArchitectureGraph {
  return {
    schema_version: 1,
    nodes: nodes.map<ArchitectureNode>((node) => ({
      id: node.id,
      type: node.data.spec.type,
      label: node.data.label,
      position: { x: Math.round(node.position.x), y: Math.round(node.position.y) },
      params: node.data.params as Record<string, never>
    })),
    edges: edges.map<ArchitectureEdge>((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      source_port: "out",
      target_port: "in"
    })),
    training_defaults: trainingDefaults as Record<string, never>
  };
}

function placeholderSpec(type: string): NodeSpec {
  return {
    type,
    name: type,
    category: "Unavailable",
    description: "This node type is not in the current catalog.",
    params: [],
    inputs: [{ key: "in", label: "In" }],
    outputs: [{ key: "out", label: "Out" }],
    min_inputs: 0,
    max_inputs: -1,
    task_types: [],
    kind: "layer",
    source: ""
  };
}

/** Catalog defaults for a freshly dropped node. */
export function defaultParams(spec: NodeSpec): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  for (const param of spec.params) {
    params[param.key] = param.type === "multiselect" ? (param.default ?? []) : param.default;
  }
  return params;
}

/**
 * A readable, collision-free node id.
 *
 * Ids appear in generated layer names, so `conv2d_3` beats a uuid when a user
 * is reading `model.summary()` next to the canvas.
 */
export function nextNodeId(type: string, existing: Iterable<string>): string {
  const taken = new Set(existing);
  for (let index = 1; ; index += 1) {
    const candidate = `${type}_${index}`;
    if (!taken.has(candidate)) return candidate;
  }
}

export function edgeId(source: string, target: string): string {
  return `e_${source}_${target}`;
}

/**
 * How many layers a block node stands for, or 0 if it is not a stack.
 *
 * Blocks name their repeat count differently — a transformer stack calls it
 * `layers`, a ResNet stage `blocks` — because each matches the vocabulary of the
 * paper it comes from. The canvas needs one number regardless.
 */
export function blockRepeat(data: ArchNodeData): number {
  if (data.spec.kind !== "block") return 0;
  for (const key of ["layers", "blocks"]) {
    const value = data.params[key];
    if (typeof value === "number" && value > 0) return value;
  }
  return 0;
}

/** `(224, 224, 3)`, or a dash when the analytic pass could not resolve it. */
export function formatShape(shape: (number | null)[] | null | undefined): string {
  if (!shape) return "—";
  return `(${shape.map((dim) => (dim === null ? "?" : dim)).join(", ")})`;
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value.toLocaleString();
}

/** Group issues by the node they belong to; graph-level issues key on "". */
export function issuesByNode(issues: ArchitectureIssue[]): Map<string, ArchitectureIssue[]> {
  const grouped = new Map<string, ArchitectureIssue[]>();
  for (const issue of issues) {
    const key = issue.node_id ?? "";
    grouped.set(key, [...(grouped.get(key) ?? []), issue]);
  }
  return grouped;
}
