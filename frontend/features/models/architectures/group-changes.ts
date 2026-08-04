import type { NodeChange } from "@xyflow/react";
import type { ArchFlowNode, StoredGroup } from "./graph-state";

/**
 * Fold React Flow's change stream into the stored group frames.
 *
 * Group frames are drawn as React Flow nodes but live in `training_defaults`,
 * not in the graph, so they cannot go through `applyNodeChanges` — this is the
 * equivalent for them, kept pure so the axis rules below are testable.
 *
 * `changes` must already be narrowed to group frames; the caller has the id set
 * because it also has to route the remainder to the model nodes.
 */
export function applyGroupChanges(
  groups: StoredGroup[],
  changes: NodeChange<ArchFlowNode>[]
): StoredGroup[] {
  if (changes.length === 0) return groups;

  const removed = new Set(
    changes.filter((change) => change.type === "remove").map((change) => change.id)
  );

  return groups
    .filter((group) => !removed.has(group.id))
    .map((group) => {
      let current = group;
      for (const change of changes) {
        if (!("id" in change) || change.id !== group.id) continue;
        if (change.type === "position" && change.position) {
          current = { ...current, x: change.position.x, y: change.position.y };
        }
        if (change.type === "dimensions" && change.dimensions) {
          current = { ...current, ...resizedTo(current, change) };
        }
      }
      return current;
    });
}

/**
 * Whether these changes should mark the architecture unsaved.
 *
 * Not simply "there was a change": a `dimensions` change that came from
 * measurement rather than from a drag is React Flow telling us what it saw, and
 * treating that as an edit would mark a freshly opened architecture dirty
 * before it was touched. Selection is likewise not an edit.
 *
 * Kept separate from `applyGroupChanges` so the caller can use a functional
 * state updater — during a resize drag several change batches can arrive before
 * React re-renders, and a captured `groups` would be stale.
 */
export function isGroupEdit(changes: NodeChange<ArchFlowNode>[]): boolean {
  return changes.some((change) =>
    change.type === "dimensions" ? isResize(change) : change.type !== "select"
  );
}

type DimensionChange = Extract<NodeChange<ArchFlowNode>, { type: "dimensions" }>;

/**
 * Whether this batch opens or closes a NodeResizer drag.
 *
 * Used to push exactly one undo step per drag rather than one per pointer
 * event. Both edges are recognised by `resizing` being a boolean, which is the
 * only field XYResizer sets on *both* its during-drag and its end change — the
 * end change carries no `setAttributes`, so testing that instead meant "end"
 * never fired and the drag never closed.
 */
export function resizePhase(changes: NodeChange<ArchFlowNode>[]): "start" | "end" | null {
  const fromResizer = changes.filter(
    (change): change is DimensionChange => change.type === "dimensions" && isResize(change)
  );
  // End wins over start: a batch carrying both is the tail of one drag.
  if (fromResizer.some((change) => change.resizing === false)) return "end";
  if (fromResizer.some((change) => change.resizing === true)) return "start";
  return null;
}

/**
 * A dimensions change is a real resize only when it carries `resizing`.
 *
 * XYResizer sets it — `true` while dragging, `false` on release — and nothing
 * else does, so it separates a deliberate drag from a measurement. A
 * measurement must not overwrite the stored size, or every reopen would nudge
 * frames toward whatever the DOM happened to lay out.
 */
function isResize(change: DimensionChange): boolean {
  return typeof change.resizing === "boolean";
}

/**
 * The size a resize change asks for, respecting its axis.
 *
 * `setAttributes` names the axis when the resizer is direction-locked and is
 * `true` (or, on the end change, absent) otherwise. XYResizer already reports
 * the untouched axis at its previous value, so writing both is correct in the
 * unlocked case; the axis check only guards a direction-locked resizer.
 */
function resizedTo(
  group: StoredGroup,
  change: DimensionChange
): { width: number; height: number } {
  if (!isResize(change) || !change.dimensions) {
    return { width: group.width, height: group.height };
  }
  const axis = change.setAttributes;
  return {
    width: axis === "height" ? group.width : change.dimensions.width,
    height: axis === "width" ? group.height : change.dimensions.height
  };
}
