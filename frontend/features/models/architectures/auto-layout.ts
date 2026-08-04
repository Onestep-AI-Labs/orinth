import type { Edge } from "@xyflow/react";
import type { ArchFlowNode } from "./graph-state";

/**
 * Layered left-to-right layout, mirroring `backend/app/ml/architecture/layout.py`.
 *
 * Three rules, applied in order:
 *
 * 1. Column is the longest path from any source.
 * 2. Siblings fan out symmetrically around their shared parent, so an Inception
 *    module's branches straddle the stem rather than hanging off its bottom.
 * 3. An edge that skips more than one column — a residual shortcut, a U-Net skip
 *    — pushes the nodes it goes around off the main row, so the shortcut can be
 *    drawn straight. Without this a residual block lays out as one line with the
 *    skip hidden underneath the layers it is supposed to route around.
 *
 * Reimplemented here rather than round-tripping to the backend because Tidy
 * should feel instant. The two must stay in step: if they disagree, pressing
 * Tidy on a freshly opened template moves every node.
 */

/**
 * Horizontal spacing between columns.
 *
 * A node is 176px wide, so this is the wire length as much as the pitch: at
 * 240 the gap was 64px and a step edge had barely room to show its corner
 * before arriving, which read as nodes touching rather than as a graph with
 * connections. 300 leaves 124px of visible wire — enough for the hover pill to
 * sit on without covering either endpoint.
 *
 * `backend/app/ml/architecture/layout.py` carries the same value and must
 * change with it, or Tidy moves every node on a freshly opened template.
 */
export const COLUMN_PITCH = 300;
export const ROW_PITCH = 120;
const ORIGIN_X = 80;
const ORIGIN_Y = 80;

/** A node's drawn size, for hit-testing candidate positions against. */
export const NODE_WIDTH = 176;
export const NODE_HEIGHT = 78;

/**
 * The first spot at or below `start` that no existing node is sitting on.
 *
 * Dropping a new node on the centre of the view is right until the centre of
 * the view already has a node on it, which — on a graph you are in the middle
 * of editing — it usually does. Walking down a row at a time keeps the node in
 * the same column as whatever it will probably connect to, and lands it in
 * clear space rather than hidden behind what is already there.
 */
export function freeSpot(
  start: { x: number; y: number },
  nodes: readonly { position: { x: number; y: number } }[]
): { x: number; y: number } {
  const clearX = NODE_WIDTH + 24;
  const clearY = NODE_HEIGHT + 24;
  const occupied = (spot: { x: number; y: number }) =>
    nodes.some(
      (node) =>
        Math.abs(node.position.x - spot.x) < clearX &&
        Math.abs(node.position.y - spot.y) < clearY
    );
  let spot = start;
  // Bounded so a pathological graph cannot walk forever; 40 rows is far below
  // anything a viewport shows.
  for (let step = 0; step < 40 && occupied(spot); step += 1) {
    spot = { x: spot.x, y: spot.y + NODE_HEIGHT + 42 };
  }
  return spot;
}

export function autoLayout(nodes: ArchFlowNode[], edges: Edge[]): ArchFlowNode[] {
  if (nodes.length === 0) return nodes;

  const ids = nodes.map((node) => node.id);
  const known = new Set(ids);
  const incoming = new Map<string, string[]>(ids.map((id) => [id, []]));
  const outgoing = new Map<string, string[]>(ids.map((id) => [id, []]));
  for (const edge of edges) {
    if (!known.has(edge.source) || !known.has(edge.target) || edge.source === edge.target) continue;
    incoming.get(edge.target)?.push(edge.source);
    outgoing.get(edge.source)?.push(edge.target);
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

  const offsets = bypassOffsets(ordered, incoming, outgoing, layer);

  const byLayer = new Map<number, string[]>();
  for (const id of [...ids].sort(
    (a, b) => (layer.get(a) ?? 0) - (layer.get(b) ?? 0) || a.localeCompare(b)
  )) {
    const column = layer.get(id) ?? 0;
    byLayer.set(column, [...(byLayer.get(column) ?? []), id]);
  }

  // `base` ignores the bypass offsets so they never compound down a chain: a
  // five-layer detour is one row off the main line, not five.
  const base = new Map<string, number>();
  const row = new Map<string, number>();
  for (const column of [...byLayer.keys()].sort((a, b) => a - b)) {
    const taken = new Set<number>();
    const siblings = new Map<string, string[]>();
    for (const id of byLayer.get(column) ?? []) {
      const key = (incoming.get(id) ?? []).slice().sort().join("|");
      siblings.set(key, [...(siblings.get(key) ?? []), id]);
    }

    for (const [, group] of siblings) {
      const parents = (incoming.get(group[0]) ?? [])
        .map((source) => base.get(source))
        .filter((value): value is number => value !== undefined);
      const centre = parents.length > 0 ? median(parents) : 0;
      // Odd counts sit one dead centre; even counts straddle it.
      const spread = (group.length - 1) / 2;
      group.forEach((id, index) => {
        const offset = offsets.get(id) ?? 0;
        let slot = Math.round(centre + index - spread) + offset;
        while (taken.has(slot)) slot += 1;
        taken.add(slot);
        row.set(id, slot);
        base.set(id, slot - offset);
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

/**
 * How far off the main row each node sits because something skips past it.
 *
 * Overlapping spans stack, so two nested skips do not land on top of each
 * other; the widest is placed first so an outer shortcut claims row 1.
 */
function bypassOffsets(
  ids: string[],
  incoming: Map<string, string[]>,
  outgoing: Map<string, string[]>,
  layer: Map<string, number>
): Map<string, number> {
  const spans: { start: number; end: number; source: string; target: string }[] = [];
  for (const source of ids) {
    for (const target of [...(outgoing.get(source) ?? [])].sort()) {
      const start = layer.get(source) ?? 0;
      const end = layer.get(target) ?? 0;
      if (end - start > 1) spans.push({ start, end, source, target });
    }
  }
  const offsets = new Map<string, number>();
  if (spans.length === 0) return offsets;

  spans.sort((a, b) => a.start - a.end - (b.start - b.end) || a.start - b.start || a.source.localeCompare(b.source));
  const claimed: { start: number; end: number; depth: number }[] = [];
  for (const span of spans) {
    const depth =
      1 + claimed.filter((other) => other.start < span.end && span.start < other.end).length;
    claimed.push({ start: span.start, end: span.end, depth });
    for (const id of between(span.source, span.target, incoming, outgoing, layer)) {
      offsets.set(id, Math.max(offsets.get(id) ?? 0, depth));
    }
  }
  return offsets;
}

/** Nodes strictly inside a span: reachable from `source` and reaching `target`. */
function between(
  source: string,
  target: string,
  incoming: Map<string, string[]>,
  outgoing: Map<string, string[]>,
  layer: Map<string, number>
): Set<string> {
  const limit = layer.get(target) ?? 0;
  const forward = new Set<string>();
  const forwardStack = [...(outgoing.get(source) ?? [])];
  while (forwardStack.length > 0) {
    const id = forwardStack.pop() as string;
    if (forward.has(id) || (layer.get(id) ?? 0) >= limit) continue;
    forward.add(id);
    forwardStack.push(...(outgoing.get(id) ?? []));
  }

  const inside = new Set<string>();
  const backStack = [...(incoming.get(target) ?? [])];
  while (backStack.length > 0) {
    const id = backStack.pop() as string;
    if (inside.has(id) || !forward.has(id)) continue;
    inside.add(id);
    backStack.push(...(incoming.get(id) ?? []));
  }
  return inside;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}
