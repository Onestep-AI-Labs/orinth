"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { AlertTriangle } from "lucide-react";
import { CategoryIcon, categoryColorVar } from "./node-visuals";
import type { ArchFlowNode } from "./graph-state";
import { formatShape } from "./graph-state";

/**
 * One node on the canvas.
 *
 * A resting node is border-first per `frontend/DESIGN.md` §5 — hairline border,
 * no shadow. Selection darkens the border to `--accent` and adds the sanctioned
 * `--ring-accent`; an error swaps the border to `--danger`. The output shape is
 * the node's most useful fact, so it gets the mono line under the name.
 */
export const CanvasNode = memo(function CanvasNode({ data, selected }: NodeProps<ArchFlowNode>) {
  const hasError = data.issues.some((issue) => issue.severity === "error");
  const hasWarning = !hasError && data.issues.length > 0;
  const takesInput = data.spec.max_inputs !== 0;
  const emitsOutput = data.spec.outputs.length > 0;

  return (
    <div
      className={[
        "arch-node",
        selected ? "arch-node-selected" : "",
        hasError ? "arch-node-error" : "",
        hasWarning ? "arch-node-warning" : ""
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {takesInput && <Handle type="target" position={Position.Left} className="arch-handle" />}
      <span
        className="arch-node-stripe"
        style={{ background: categoryColorVar(data.spec.category) }}
        aria-hidden
      />
      <p className="arch-node-category">
        <CategoryIcon category={data.spec.category} />
        {data.spec.category}
      </p>
      <p className="arch-node-name">
        {data.label}
        {(hasError || hasWarning) && (
          <AlertTriangle
            className={hasError ? "arch-node-flag-error" : "arch-node-flag-warn"}
            size={13}
            aria-hidden
          />
        )}
      </p>
      <p className="arch-node-shape">{formatShape(data.shape)}</p>
      {emitsOutput && <Handle type="source" position={Position.Right} className="arch-handle" />}
    </div>
  );
});
