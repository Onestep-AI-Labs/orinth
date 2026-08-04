"use client";

import { useCallback, useRef, useState } from "react";
import type { Edge } from "@xyflow/react";
import type { ArchFlowNode, StoredGroup } from "./graph-state";

/** Everything an undo step has to restore. */
export type GraphSnapshot = {
  nodes: ArchFlowNode[];
  edges: Edge[];
  groups: StoredGroup[];
};

/** How many steps back the editor remembers. */
const HISTORY_LIMIT = 100;

/**
 * Snapshot-based undo/redo for the canvas.
 *
 * The stack is written *before* a change, not after — `take()` is called at the
 * top of each mutating action and at the start of a drag. That is what makes
 * the granularity match intent: a drag of forty position events is one undo
 * step because only the mousedown took a snapshot, and one Ctrl+Z puts the node
 * back where it was picked up rather than one pixel to the left.
 *
 * Snapshots live in refs rather than state so taking one never re-renders the
 * canvas; only the two booleans the toolbar reads are state.
 */
export function useHistory(read: () => GraphSnapshot, apply: (snapshot: GraphSnapshot) => void) {
  const past = useRef<GraphSnapshot[]>([]);
  const future = useRef<GraphSnapshot[]>([]);
  const [{ canUndo, canRedo }, setAvailability] = useState({ canUndo: false, canRedo: false });

  const sync = useCallback(() => {
    setAvailability({ canUndo: past.current.length > 0, canRedo: future.current.length > 0 });
  }, []);

  /** Record the state as it stands, discarding any redo branch. */
  const take = useCallback(() => {
    past.current = [...past.current, read()].slice(-HISTORY_LIMIT);
    future.current = [];
    sync();
  }, [read, sync]);

  const undo = useCallback(() => {
    const previous = past.current.at(-1);
    if (!previous) return;
    past.current = past.current.slice(0, -1);
    future.current = [...future.current, read()];
    apply(previous);
    sync();
  }, [read, apply, sync]);

  const redo = useCallback(() => {
    const next = future.current.at(-1);
    if (!next) return;
    future.current = future.current.slice(0, -1);
    past.current = [...past.current, read()];
    apply(next);
    sync();
  }, [read, apply, sync]);

  /** Drop the stack — used when a different graph is loaded into the canvas. */
  const reset = useCallback(() => {
    past.current = [];
    future.current = [];
    sync();
  }, [sync]);

  return { take, undo, redo, reset, canUndo, canRedo };
}
