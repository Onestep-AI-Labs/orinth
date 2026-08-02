import type { Edge } from "@xyflow/react";
import type { ArchFlowNode } from "./graph-state";

/**
 * Layered left-to-right layout, mirroring `backend/app/ml/architecture/layout.py`.
 *
 * Column is the longest path from any source; row is the median row of a
 * node's predecessors, nudged down to avoid overlap. Deterministic, so
 * pressing Tidy twice never moves anything the second time.
 *
 * Reimplemented here rather than round-tripping to the backend because Tidy
 * should feel instant and operates purely on canvas state.
 */

const COLUMN_PITCH = 240;
const ROW_PITCH = 120;
const ORIGIN_X = 80;
const ORIGIN_Y = 80;

export function autoLayout(nodes: ArchFlowNode[], edges: Edge[]): ArchFlowNode[] {
  if (nodes.length === 0) return nodes;

  const ids = nodes.map((node) => node.id);
  const known = new Set(ids);
  const incoming = new Map<string, string[]>(ids.map((id) => [id, []]));
  for (const edge of edges) {
    if (!known.has(edge.source) || !known.has(edge.target) || edge.source === edge.target) continue;
    incoming.get(edge.target)?.push(edge.source);
  }

  // Longest-path layering, relaxed iteratively. Capped at one pass per node so
  // a cycle — which the validator rejects anyway — cannot spin forever.
  const layer = new Map<string, number>(ids.map((id) => [id, 0]));
  const ordered = [...ids].sort();
  for (let pass = 0; pass < nodes.length; pass += 1) {
    let changed = false;
    for (const id of ordered) {
      for (const source of incoming.get(id) ?? []) {
        const candidate = (layer.get(source) ?? 0) + 1;
        if (candidate > (layer.get(id) ?? 0)) {
          layer.set(id, candidate);
          changed = true;
        }
      }
    }
    if (!changed) break;
  }

  const byLayer = new Map<number, string[]>();
  for (const id of [...ids].sort((a, b) => (layer.get(a) ?? 0) - (layer.get(b) ?? 0) || a.localeCompare(b))) {
    const column = layer.get(id) ?? 0;
    byLayer.set(column, [...(byLayer.get(column) ?? []), id]);
  }

  // Rows: place each node near the median of its predecessors. Siblings of one
  // parent fan out symmetrically around it, so a branch looks like a branch
  // rather than a column of nodes hanging off the bottom of the parent.
  const row = new Map<string, number>();
  for (const column of [...byLayer.keys()].sort((a, b) => a - b)) {
    const taken = new Set<number>();
    const inColumn = byLayer.get(column) ?? [];

    // Group this column by parent so siblings can be centred together.
    const siblings = new Map<string, string[]>();
    for (const id of inColumn) {
      const key = (incoming.get(id) ?? []).slice().sort().join("|");
      siblings.set(key, [...(siblings.get(key) ?? []), id]);
    }

    for (const [, group] of siblings) {
      const parents = (incoming.get(group[0]) ?? [])
        .map((source) => row.get(source))
        .filter((value): value is number => value !== undefined);
      const centre = parents.length > 0 ? median(parents) : 0;
      // Odd counts sit one dead centre; even counts straddle it.
      const offset = (group.length - 1) / 2;
      group.forEach((id, index) => {
        let slot = Math.round(centre + index - offset);
        while (taken.has(slot)) slot += 1;
        taken.add(slot);
        row.set(id, slot);
      });
    }
  }

  // Shift everything so the topmost node sits on the first row rather than
  // wherever the fan-out arithmetic happened to land.
  const rows = [...row.values()];
  const lift = rows.length > 0 ? Math.min(...rows) : 0;

  return nodes.map((node) => ({
    ...node,
    position: {
      x: ORIGIN_X + (layer.get(node.id) ?? 0) * COLUMN_PITCH,
      y: ORIGIN_Y + ((row.get(node.id) ?? 0) - lift) * ROW_PITCH
    }
  }));
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}
