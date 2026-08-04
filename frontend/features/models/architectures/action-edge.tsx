"use client";

import { memo } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  getSmoothStepPath,
  type EdgeProps
} from "@xyflow/react";
import { Plus, X } from "lucide-react";

/** Insert a node in the middle of this connection. Detail: `{ id }`. */
export const EDGE_INSERT_EVENT = "arch-edge-insert";
/** Remove this connection. Detail: `{ id }`. */
export const EDGE_DELETE_EVENT = "arch-edge-delete";
/** Keep or drop the hover affordance. Detail: `{ id: string | null }`. */
export const EDGE_HOVER_EVENT = "arch-edge-hover";

export type ActionEdgeData = {
  /** A fan-out or a merge, drawn as a curve rather than a step. */
  branching?: boolean;
  /** Whether the pointer is on this wire — or on its own buttons. */
  hovered?: boolean;
};

function announce(name: string, detail: unknown) {
  window.dispatchEvent(new CustomEvent(name, { detail }));
}

/**
 * A connection that carries its own edit controls.
 *
 * Hovering a wire reveals a two-button pill at its midpoint: **+** drops a node
 * into the middle of the connection, **×** removes it. Both were previously
 * reachable only from the right-click menu, which is the slowest path to the
 * two things you do to a wire most often — and neither was discoverable at all
 * without trying a right-click on a 2px line.
 *
 * The pill lives in `EdgeLabelRenderer`, a portal outside this edge's own DOM
 * subtree, so `.react-flow__edge:hover .pill` cannot reach it in CSS. Hover is
 * therefore state: the studio tracks which wire is under the pointer and hands
 * it back through `data.hovered`, and the pill re-asserts that hover on its own
 * pointer enter so moving from the line onto a button does not dismiss it.
 */
export const ActionEdge = memo(function ActionEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  style,
  data
}: EdgeProps) {
  const { branching, hovered } = (data ?? {}) as ActionEdgeData;
  const geometry = {
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition
  };
  const [path, labelX, labelY] = branching
    ? getBezierPath(geometry)
    : getSmoothStepPath({ ...geometry, borderRadius: 8 });

  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />
      <EdgeLabelRenderer>
        <div
          // `nodrag`/`nopan` so pressing a button never starts a marquee or a
          // pan behind it.
          className={`arch-edge-actions nodrag nopan${hovered ? " arch-edge-actions-open" : ""}`}
          style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          onPointerEnter={() => announce(EDGE_HOVER_EVENT, { id })}
          onPointerLeave={() => announce(EDGE_HOVER_EVENT, { id: null })}
        >
          {/* `data-tip` rather than `title`: the pill is already transient, and
              a native tooltip's ~1s delay outlives the hover that summoned it. */}
          <button
            type="button"
            className="arch-edge-action arch-tip arch-tip-up"
            aria-label="Insert a node on this connection"
            data-tip="Insert node"
            onClick={(event) => {
              event.stopPropagation();
              announce(EDGE_INSERT_EVENT, { id });
            }}
          >
            <Plus size={13} aria-hidden />
          </button>
          <button
            type="button"
            className="arch-edge-action arch-edge-action-danger arch-tip arch-tip-up"
            aria-label="Remove this connection"
            data-tip="Remove"
            onClick={(event) => {
              event.stopPropagation();
              announce(EDGE_DELETE_EVENT, { id });
            }}
          >
            <X size={13} aria-hidden />
          </button>
        </div>
      </EdgeLabelRenderer>
    </>
  );
});
