"use client";

import { memo, useState } from "react";
import { NodeResizer, type NodeProps } from "@xyflow/react";
import { Check, Palette } from "lucide-react";
import type { Node } from "@xyflow/react";
import { labelColor, labelFill } from "@/features/platform/utils";

export type GroupData = {
  title: string;
  /** Index into `GROUP_TINTS`, not a colour literal. */
  tint: number;
};

export type GroupFlowNode = Node<GroupData, "group">;

/**
 * The named pastel backgrounds a group frame can take.
 *
 * Pastels rather than the `--label` ramp at full strength: a frame sits
 * *behind* nodes across a large area, and anything saturated at that size
 * would fight the nodes for attention and break the accent budget. These are
 * the same eight hues at low alpha, which reads as a wash rather than a fill.
 */
export const GROUP_TINTS = [
  "Slate",
  "Amber",
  "Violet",
  "Green",
  "Rose",
  "Cyan",
  "Ochre",
  "Magenta"
].map((name, index) => ({
  name,
  // `labelFill` is the sanctioned way to get a translucent label colour;
  // concatenating an alpha suffix onto `labelColor` yields an invalid value
  // (DESIGN.md §8), which here would paint an opaque frame over the canvas.
  fill: labelFill(index, 9),
  edge: labelColor(index)
}));

/**
 * A titled, resizable frame drawn behind nodes.
 *
 * Purely an annotation — it holds no tensors, emits no code, and never joins
 * the graph. Its whole job is letting someone say "this region is the encoder"
 * on a canvas that would otherwise be forty anonymous boxes.
 */
export const GroupFrame = memo(function GroupFrame({
  id,
  data,
  selected
}: NodeProps<GroupFlowNode>) {
  const [editing, setEditing] = useState(false);
  const [picking, setPicking] = useState(false);
  const tint = GROUP_TINTS[data.tint % GROUP_TINTS.length];

  return (
    <div
      className={`arch-group${selected ? " arch-group-selected" : ""}`}
      style={{ background: tint.fill, borderColor: tint.edge }}
    >
      <NodeResizer minWidth={160} minHeight={120} isVisible={selected} lineClassName="arch-group-line" handleClassName="arch-group-handle" />
      <div className="arch-group-head">
        {editing ? (
          <input
            className="arch-group-title-input"
            autoFocus
            value={data.title}
            aria-label="Group title"
            onChange={(event) => {
              // React Flow node data is replaced, not mutated, so the studio
              // owns the update through this custom event.
              window.dispatchEvent(
                new CustomEvent("arch-group-title", {
                  detail: { id, title: event.target.value }
                })
              );
            }}
            onBlur={() => setEditing(false)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === "Escape") setEditing(false);
            }}
          />
        ) : (
          <button
            type="button"
            className="arch-group-title"
            style={{ color: tint.edge }}
            onClick={() => setEditing(true)}
            title="Rename this group"
          >
            {data.title || "Untitled group"}
          </button>
        )}
        <button
          type="button"
          className="arch-group-swatch"
          aria-label="Change the group colour"
          style={{ background: tint.edge }}
          onClick={() => setPicking((value) => !value)}
        >
          <Palette size={11} aria-hidden />
        </button>
      </div>
      {picking && (
        <div className="arch-group-palette" role="listbox" aria-label="Group colour">
          {GROUP_TINTS.map((option, index) => (
            <button
              key={option.name}
              type="button"
              role="option"
              aria-selected={index === data.tint}
              aria-label={option.name}
              title={option.name}
              className="arch-group-tint"
              style={{ background: option.edge }}
              onClick={() => {
                window.dispatchEvent(
                  new CustomEvent("arch-group-tint", { detail: { id, tint: index } })
                );
                setPicking(false);
              }}
            >
              {index === data.tint && <Check size={11} aria-hidden />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
});
