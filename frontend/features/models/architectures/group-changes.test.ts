import { describe, expect, it } from "vitest";
import type { NodeChange } from "@xyflow/react";
import { applyGroupChanges, isGroupEdit, resizePhase } from "./group-changes";
import type { ArchFlowNode, StoredGroup } from "./graph-state";

type Change = NodeChange<ArchFlowNode>;

const frame: StoredGroup = {
  id: "group_1",
  title: "Encoder",
  tint: 2,
  x: 100,
  y: 80,
  width: 400,
  height: 300
};

/**
 * A during-drag change from XYResizer.
 *
 * It always reports both axes — the one that was not dragged comes back at its
 * previous value — and `setAttributes` is `true` for every handle, because
 * `NodeResizer` never passes a `resizeDirection`.
 */
function resize(width: number, height: number): Change {
  return {
    id: "group_1",
    type: "dimensions",
    dimensions: { width, height },
    resizing: true,
    setAttributes: true
  };
}

/** The single change XYResizer emits on pointer release. Note: no `setAttributes`. */
function resizeEnd(width: number, height: number): Change {
  return {
    id: "group_1",
    type: "dimensions",
    dimensions: { width, height },
    resizing: false
  };
}

describe("applyGroupChanges — resizing", () => {
  it("dragging the right edge changes width only", () => {
    // The resizer reports the untouched height at its previous value.
    const [next] = applyGroupChanges([frame], [resize(560, 300)]);

    expect(next.width).toBe(560);
    expect(next.height).toBe(300);
  });

  it("dragging a corner changes both", () => {
    const [next] = applyGroupChanges([frame], [resize(560, 420)]);

    expect(next.width).toBe(560);
    expect(next.height).toBe(420);
  });

  it("commits the size on the end change, which carries no setAttributes", () => {
    const [next] = applyGroupChanges([frame], [resizeEnd(560, 420)]);

    expect(next.width).toBe(560);
    expect(next.height).toBe(420);
  });

  it("ignores a measurement, which carries no resizing flag", () => {
    const measured: Change = {
      id: "group_1",
      type: "dimensions",
      dimensions: { width: 383, height: 291 }
    };

    const [next] = applyGroupChanges([frame], [measured]);

    expect(next.width).toBe(400);
    expect(next.height).toBe(300);
  });

  it("respects the axis when the resizer is direction-locked", () => {
    const locked: Change = {
      id: "group_1",
      type: "dimensions",
      dimensions: { width: 560, height: 999 },
      resizing: true,
      setAttributes: "width"
    };

    const [next] = applyGroupChanges([frame], [locked]);

    expect(next).toMatchObject({ width: 560, height: 300 });
  });

  it("applies the position change that comes with a top-left resize", () => {
    const changes: Change[] = [
      { id: "group_1", type: "position", position: { x: 60, y: 40 } },
      resize(440, 340)
    ];

    const [next] = applyGroupChanges([frame], changes);

    expect(next).toMatchObject({ x: 60, y: 40, width: 440, height: 340 });
  });
});

describe("applyGroupChanges — other changes", () => {
  it("drops removed frames and leaves the rest alone", () => {
    const other: StoredGroup = { ...frame, id: "group_2", title: "Head" };

    const next = applyGroupChanges(
      [frame, other],
      [{ id: "group_1", type: "remove" }]
    );

    expect(next.map((group) => group.id)).toEqual(["group_2"]);
  });

  it("returns the same array when there is nothing to apply", () => {
    const groups = [frame];
    expect(applyGroupChanges(groups, [])).toBe(groups);
  });

  it("preserves title styling through a resize", () => {
    const styled: StoredGroup = { ...frame, titleX: 30, titleY: 40, titleSize: 20, titleTint: 5 };

    const [next] = applyGroupChanges([styled], [resize(560, 420)]);

    expect(next).toMatchObject({ titleX: 30, titleY: 40, titleSize: 20, titleTint: 5 });
  });
});

describe("resizePhase", () => {
  it("opens on the resizer's first change and closes on its last", () => {
    expect(resizePhase([resize(560, 420)])).toBe("start");
    // The end change carries no `setAttributes`; missing it left the drag open
    // forever, and every later change to the frame was swallowed by it.
    expect(resizePhase([resizeEnd(560, 420)])).toBe("end");
  });

  it("closes on a batch that both resizes and ends, so a drag cannot latch", () => {
    expect(resizePhase([resize(560, 420), resizeEnd(560, 420)])).toBe("end");
  });

  it("ignores measurement changes, so one drag stays one undo step", () => {
    // No `resizing` flag: this is React Flow reporting what it measured.
    expect(
      resizePhase([{ id: "group_1", type: "dimensions", dimensions: { width: 383, height: 291 } }])
    ).toBeNull();
  });

  it("ignores changes that are not dimensions at all", () => {
    expect(resizePhase([{ id: "group_1", type: "position", position: { x: 1, y: 2 } }])).toBeNull();
    expect(resizePhase([{ id: "group_1", type: "select", selected: true }])).toBeNull();
  });
});

describe("isGroupEdit", () => {
  it("counts a resize as an edit", () => {
    expect(isGroupEdit([resize(560, 420)])).toBe(true);
    expect(isGroupEdit([resizeEnd(560, 420)])).toBe(true);
  });

  it("does not count measurement or selection", () => {
    expect(
      isGroupEdit([{ id: "group_1", type: "dimensions", dimensions: { width: 1, height: 2 } }])
    ).toBe(false);
    expect(isGroupEdit([{ id: "group_1", type: "select", selected: true }])).toBe(false);
  });

  it("counts a move and a removal", () => {
    expect(isGroupEdit([{ id: "group_1", type: "position", position: { x: 1, y: 2 } }])).toBe(true);
    expect(isGroupEdit([{ id: "group_1", type: "remove" }])).toBe(true);
  });
});
