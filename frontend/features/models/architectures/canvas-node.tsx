"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { AlertTriangle, Layers } from "lucide-react";
import { CategoryIcon, categoryColorVar, categoryTint } from "./node-visuals";
import type { ArchFlowNode } from "./graph-state";
import { blockRepeat, formatShape } from "./graph-state";

/**
 * One node on the canvas.
 *
 * A resting node is border-first per `frontend/DESIGN.md` §5 — hairline border,
 * no shadow. What carries meaning is the header: a pastel wash in the category's
 * colour from the `--label` ramp, so a glance across a forty-node graph groups
 * convolutions from norms from attention without reading a single name.
 *
 * A **block** — a ResNet stage, a Qwen3 decoder block — is drawn with a stacked
 * edge behind it and a `×N` badge, because it is not one layer. That distinction
 * is the difference between a canvas that says "12 nodes" and one that says
 * "this is 34 layers of Gemma".
 */
export const CanvasNode = memo(function CanvasNode({ data, selected }: NodeProps<ArchFlowNode>) {
  const hasError = data.issues.some((issue) => issue.severity === "error");
  const hasWarning = !hasError && data.issues.length > 0;
  const takesInput = data.spec.max_inputs !== 0;
  const emitsOutput = data.spec.outputs.length > 0;
  const isBlock = data.spec.kind === "block";
  const repeat = isBlock ? blockRepeat(data) : 0;
  const accent = categoryColorVar(data.spec.category);

  return (
    <div
      className={[
        "arch-node",
        isBlock ? "arch-node-block" : "",
        selected ? "arch-node-selected" : "",
        hasError ? "arch-node-error" : "",
        hasWarning ? "arch-node-warning" : ""
      ]
        .filter(Boolean)
        .join(" ")}
      style={{ "--arch-node-accent": accent } as React.CSSProperties}
    >
      {/* The stacked edge behind a block: it stands for N layers, not one. */}
      {isBlock && (
        <span
          className="arch-node-stack"
          style={{ background: categoryTint(data.spec.category, 22) }}
          aria-hidden
        />
      )}
      {takesInput && <Handle type="target" position={Position.Left} className="arch-handle" />}
      <div
        className="arch-node-head"
        style={{ background: categoryTint(data.spec.category, isBlock ? 18 : 11) }}
      >
        <p className="arch-node-category" style={{ color: accent }}>
          <CategoryIcon category={data.spec.category} />
          {data.spec.category}
        </p>
        {repeat > 1 && (
          <span className="arch-node-repeat" style={{ color: accent }} title={`${repeat} layers`}>
            <Layers size={11} aria-hidden />×{repeat}
          </span>
        )}
      </div>
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
