import type { Edge } from "@xyflow/react";
import type { ArchFlowNode, StoredGroup } from "./graph-state";
import { edgeId, nextNodeId } from "./graph-state";

/**
 * Marks a clipboard payload as ours.
 *
 * The cut/copy/paste handlers write JSON into `text/plain` rather than a custom
 * MIME type: custom types are unevenly supported across browsers, while
 * `text/plain` is universal and survives a round trip through another tab. The
 * marker is what keeps a paste of ordinary prose from being read as a graph.
 */
const CLIPBOARD_MARKER = "onestep.architecture.clipboard/v1";

/** Where a pasted copy lands relative to the original, when pasting in place. */
const PASTE_OFFSET = 32;

export type ClipboardPayload = {
  marker: typeof CLIPBOARD_MARKER;
  nodes: ArchFlowNode[];
  edges: Edge[];
  groups: StoredGroup[];
};

/**
 * The selected nodes, the edges wholly inside that selection, and the selected
 * group frames.
 *
 * An edge with one endpoint outside the selection is dropped rather than
 * carried: pasting it would either dangle or silently rewire the original
 * graph, and neither is what "copy these three layers" means.
 */
export function selectionPayload(
  nodes: ArchFlowNode[],
  edges: Edge[],
  groups: StoredGroup[],
  selectedGroupIds: Set<string>
): ClipboardPayload | null {
  const chosen = nodes.filter((node) => node.selected);
  const chosenGroups = groups.filter((group) => selectedGroupIds.has(group.id));
  if (chosen.length === 0 && chosenGroups.length === 0) return null;
  const ids = new Set(chosen.map((node) => node.id));
  return {
    marker: CLIPBOARD_MARKER,
    nodes: chosen,
    edges: edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target)),
    groups: chosenGroups
  };
}

export function serializePayload(payload: ClipboardPayload): string {
  return JSON.stringify(payload);
}

/** Parse a clipboard string, or null if it did not come from this editor. */
export function parsePayload(raw: string): ClipboardPayload | null {
  if (!raw.includes(CLIPBOARD_MARKER)) return null;
  try {
    const parsed = JSON.parse(raw) as ClipboardPayload;
    if (parsed?.marker !== CLIPBOARD_MARKER || !Array.isArray(parsed.nodes)) return null;
    return {
      marker: CLIPBOARD_MARKER,
      nodes: parsed.nodes ?? [],
      edges: Array.isArray(parsed.edges) ? parsed.edges : [],
      groups: Array.isArray(parsed.groups) ? parsed.groups : []
    };
  } catch {
    return null;
  }
}

/**
 * Re-id a payload so it can coexist with what is already on the canvas, and
 * move it to `at` — the pointer, when the paste came from the context menu, or
 * a fixed offset from the original when it came from the keyboard.
 *
 * Internal edges are rewritten onto the new ids, so pasting a wired-up block
 * gives you a second wired-up block rather than a pile of loose nodes.
 */
export function instantiate(
  payload: ClipboardPayload,
  existingNodeIds: string[],
  existingGroupIds: string[],
  at: { x: number; y: number } | null
): { nodes: ArchFlowNode[]; edges: Edge[]; groups: StoredGroup[] } {
  const anchorX = Math.min(
    ...payload.nodes.map((node) => node.position.x),
    ...payload.groups.map((group) => group.x)
  );
  const anchorY = Math.min(
    ...payload.nodes.map((node) => node.position.y),
    ...payload.groups.map((group) => group.y)
  );
  const shiftX = at ? at.x - anchorX : PASTE_OFFSET;
  const shiftY = at ? at.y - anchorY : PASTE_OFFSET;

  const taken = [...existingNodeIds];
  const rename = new Map<string, string>();
  const nodes = payload.nodes.map((node) => {
    const id = nextNodeId(node.data.spec.type, taken);
    taken.push(id);
    rename.set(node.id, id);
    return {
      ...node,
      id,
      selected: true,
      dragging: false,
      position: { x: node.position.x + shiftX, y: node.position.y + shiftY },
      data: { ...node.data, params: { ...node.data.params } }
    };
  });

  const edges = payload.edges.flatMap((edge) => {
    const source = rename.get(edge.source);
    const target = rename.get(edge.target);
    if (!source || !target) return [];
    return [{ ...edge, id: edgeId(source, target), source, target, selected: false }];
  });

  const groupIds = new Set(existingGroupIds);
  const groups = payload.groups.map((group) => {
    let id = `group_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
    while (groupIds.has(id)) id = `${id}x`;
    groupIds.add(id);
    return { ...group, id, x: group.x + shiftX, y: group.y + shiftY };
  });

  return { nodes, edges, groups };
}
