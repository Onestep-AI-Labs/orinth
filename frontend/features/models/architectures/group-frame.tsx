"use client";

import { memo, useEffect, useRef, useState } from "react";
import { NodeResizer, useStore, type NodeProps } from "@xyflow/react";
import type { Node } from "@xyflow/react";
import { labelColor, labelFill } from "@/features/platform/utils";

export type GroupData = {
  title: string;
  /** Index into `GROUP_TINTS`, not a colour literal. */
  tint: number;
  titleX: number;
  titleY: number;
  titleSize: number;
  /** Index into `GROUP_TINTS`, or -1 for neutral ink. */
  titleTint: number;
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

/** Where a title sits and how big it is before anyone moves or resizes it. */
export const TITLE_DEFAULTS = { x: 12, y: 10, size: 12 };
export const TITLE_SIZE_RANGE = { min: 8, max: 144 };
/**
 * Menu presets for the title size. Any value in range can still be typed.
 *
 * The word-processor ramp rather than a handful of round numbers: a frame's
 * title is set against the canvas zoom, not against body copy, so the useful
 * range runs from a caption on a small frame to a banner over a whole stage.
 * Fine steps at the small end where a point matters, coarse at the top where it
 * does not; the menu scrolls, so length costs nothing.
 */
export const TITLE_SIZE_PRESETS = [
  8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 96, 128, 144
];
/** `titleTint` sentinel for "use the body ink colour rather than a hue". */
export const TITLE_TINT_INK = -1;

/** The resolved colour for a title, given its own tint and the frame's. */
export function titleColorFor(titleTint: number, frameTint: number): string {
  if (titleTint === TITLE_TINT_INK) return "var(--color-ink)";
  const index = titleTint >= 0 ? titleTint : frameTint;
  return GROUP_TINTS[index % GROUP_TINTS.length].edge;
}

/**
 * A titled, resizable frame drawn behind nodes.
 *
 * Purely an annotation — it holds no tensors, emits no code, and never joins
 * the graph. Its whole job is letting someone say "this region is the encoder"
 * on a canvas that would otherwise be forty anonymous boxes.
 *
 * The title is its own text box: click to select it, drag to place it anywhere
 * in the frame, double-click to retype it. Size and colour live in the right
 * rail. A label pinned to the top-left corner is the wrong default for a frame
 * that might be wrapping a column of nodes, and floating a colour palette over
 * the canvas covered the very nodes the frame was describing.
 */
export const GroupFrame = memo(function GroupFrame({
  id,
  data,
  selected
}: NodeProps<GroupFlowNode>) {
  const [editing, setEditing] = useState(false);
  const [titleActive, setTitleActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  // Screen pixels are flow units multiplied by the zoom, so a drag has to be
  // divided by it or the title races the pointer at any zoom but 100%.
  const zoom = useStore((state) => state.transform[2]);
  const zoomRef = useRef(zoom);
  zoomRef.current = zoom;

  const tint = GROUP_TINTS[data.tint % GROUP_TINTS.length];

  useEffect(() => {
    if (!editing) return;
    inputRef.current?.focus();
    inputRef.current?.select();
  }, [editing]);

  // Deselecting the frame drops the title's selection with it, so a stray
  // outline is never left behind on an unselected frame.
  useEffect(() => {
    if (!selected) {
      setTitleActive(false);
      setEditing(false);
    }
  }, [selected]);

  function startTitleDrag(event: React.PointerEvent<HTMLElement>) {
    if (editing || event.button !== 0) return;
    event.stopPropagation();
    const startX = event.clientX;
    const startY = event.clientY;
    const originX = data.titleX;
    const originY = data.titleY;
    const target = event.currentTarget;
    target.setPointerCapture(event.pointerId);

    const move = (moveEvent: PointerEvent) => {
      window.dispatchEvent(
        new CustomEvent("arch-group-title-move", {
          detail: {
            id,
            titleX: originX + (moveEvent.clientX - startX) / zoomRef.current,
            titleY: originY + (moveEvent.clientY - startY) / zoomRef.current
          }
        })
      );
    };
    const stop = () => {
      target.releasePointerCapture?.(event.pointerId);
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", stop);
      target.removeEventListener("pointercancel", stop);
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", stop);
    target.addEventListener("pointercancel", stop);
  }

  const titleStyle: React.CSSProperties = {
    left: data.titleX,
    top: data.titleY,
    fontSize: data.titleSize,
    color: titleColorFor(data.titleTint, data.tint)
  };

  return (
    <div
      className={`arch-group${selected ? " arch-group-selected" : ""}`}
      style={{ background: tint.fill, borderColor: tint.edge }}
    >
      <NodeResizer
        minWidth={160}
        minHeight={120}
        isVisible={selected}
        lineClassName="arch-group-line"
        handleClassName="arch-group-handle"
      />
      {editing ? (
        <input
          ref={inputRef}
          className="arch-group-title-input nodrag nopan"
          style={titleStyle}
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
            // Backspace inside the field must edit text, not delete the frame.
            event.stopPropagation();
          }}
        />
      ) : (
        <button
          type="button"
          className={`arch-group-title nodrag nopan${titleActive ? " arch-group-title-active" : ""}`}
          style={titleStyle}
          title="Drag to move · double-click to rename"
          onPointerDown={startTitleDrag}
          onClick={(event) => {
            event.stopPropagation();
            setTitleActive(true);
          }}
          onDoubleClick={(event) => {
            event.stopPropagation();
            setEditing(true);
          }}
        >
          {data.title || "Untitled group"}
        </button>
      )}
    </div>
  );
});
