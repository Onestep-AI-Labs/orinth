import { describe, expect, it } from "vitest";
import type { Edge } from "@xyflow/react";
import { autoLayout } from "./auto-layout";
import { blockRepeat, type ArchFlowNode, type ArchNodeData } from "./graph-state";
import type { NodeSpec } from "@/types/api";

function spec(overrides: Partial<NodeSpec> = {}): NodeSpec {
  return {
    type: "conv2d",
    name: "Conv2D",
    category: "Convolution",
    description: "",
    params: [],
    inputs: [{ key: "in", label: "In" }],
    outputs: [{ key: "out", label: "Out" }],
    min_inputs: 1,
    max_inputs: 1,
    task_types: [],
    kind: "layer",
    source: "",
    ...overrides
  };
}

function node(id: string, data: Partial<ArchNodeData> = {}): ArchFlowNode {
  return {
    id,
    type: "arch",
    position: { x: 0, y: 0 },
    data: { spec: spec(), params: {}, label: id, shape: null, issues: [], ...data }
  };
}

function edge(source: string, target: string): Edge {
  return { id: `e_${source}_${target}`, source, target };
}

const COLUMN = 240;
const ROW = 120;

function column(nodes: ArchFlowNode[], id: string): number {
  return (nodes.find((item) => item.id === id)!.position.x - 80) / COLUMN;
}

function row(nodes: ArchFlowNode[], id: string): number {
  return (nodes.find((item) => item.id === id)!.position.y - 80) / ROW;
}

describe("autoLayout", () => {
  it("puts a straight chain on one row", () => {
    const laid = autoLayout(
      ["a", "b", "c"].map((id) => node(id)),
      [edge("a", "b"), edge("b", "c")]
    );

    expect(laid.map((item) => column(laid, item.id))).toEqual([0, 1, 2]);
    expect(laid.every((item) => row(laid, item.id) === 0)).toBe(true);
  });

  it("pushes a bypassed path off the row the shortcut runs along", () => {
    // A residual block: `stem` feeds both the convolutions and the merge, so
    // the merge is five columns downstream of a direct edge. Drawn on one row
    // the skip hides underneath the layers it is supposed to route around.
    const ids = ["stem", "conv_1", "bn_1", "conv_2", "add"];
    const laid = autoLayout(ids.map((id) => node(id)), [
      edge("stem", "conv_1"),
      edge("conv_1", "bn_1"),
      edge("bn_1", "conv_2"),
      edge("conv_2", "add"),
      edge("stem", "add")
    ]);

    // The shortcut's endpoints stay on the main row...
    expect(row(laid, "stem")).toBe(0);
    expect(row(laid, "add")).toBe(0);
    // ...and everything it skips bows away from it, all by the same amount so
    // a five-layer detour is one row off the line rather than five.
    expect(row(laid, "conv_1")).toBe(1);
    expect(row(laid, "bn_1")).toBe(1);
    expect(row(laid, "conv_2")).toBe(1);
  });

  it("fans siblings symmetrically around their shared parent", () => {
    const ids = ["stem", "b1", "b2", "b3", "b4", "merge"];
    const laid = autoLayout(ids.map((id) => node(id)), [
      edge("stem", "b1"),
      edge("stem", "b2"),
      edge("stem", "b3"),
      edge("stem", "b4"),
      edge("b1", "merge"),
      edge("b2", "merge"),
      edge("b3", "merge"),
      edge("b4", "merge")
    ]);

    const branches = ["b1", "b2", "b3", "b4"].map((id) => row(laid, id));
    expect(new Set(branches).size).toBe(4);
    expect(Math.max(...branches) - Math.min(...branches)).toBe(3);
    // All four share one column, which is what makes it read as a fan.
    expect(new Set(["b1", "b2", "b3", "b4"].map((id) => column(laid, id))).size).toBe(1);
  });

  it("nests overlapping shortcuts on separate rows", () => {
    // Two residuals in a row, as in a transformer layer wired by hand.
    const ids = ["x", "n1", "a1", "r1", "n2", "f1", "r2"];
    const laid = autoLayout(ids.map((id) => node(id)), [
      edge("x", "n1"),
      edge("n1", "a1"),
      edge("a1", "r1"),
      edge("x", "r1"),
      edge("r1", "n2"),
      edge("n2", "f1"),
      edge("f1", "r2"),
      edge("r1", "r2")
    ]);

    expect(row(laid, "x")).toBe(0);
    expect(row(laid, "r1")).toBe(0);
    expect(row(laid, "r2")).toBe(0);
    expect(row(laid, "a1")).toBeGreaterThan(0);
    expect(row(laid, "f1")).toBeGreaterThan(0);
  });

  it("is idempotent, so pressing Tidy twice moves nothing the second time", () => {
    const ids = ["a", "b", "c", "d"];
    const edges = [edge("a", "b"), edge("b", "c"), edge("c", "d"), edge("a", "d")];
    const once = autoLayout(ids.map((id) => node(id)), edges);
    const twice = autoLayout(once, edges);

    expect(twice.map((item) => item.position)).toEqual(once.map((item) => item.position));
  });

  it("leaves a cyclic graph alone rather than spinning", () => {
    const laid = autoLayout(
      ["a", "b"].map((id) => node(id)),
      [edge("a", "b"), edge("b", "a")]
    );

    expect(laid).toHaveLength(2);
  });
});

describe("blockRepeat", () => {
  it("is zero for a plain layer", () => {
    expect(blockRepeat(node("conv").data)).toBe(0);
  });

  it("reads `layers` from a transformer-style block", () => {
    const data = node("blocks", {
      spec: spec({ type: "qwen3_block", kind: "block" }),
      params: { layers: 36 }
    }).data;

    expect(blockRepeat(data)).toBe(36);
  });

  it("reads `blocks` from a vision stage, which names it differently", () => {
    const data = node("stage", {
      spec: spec({ type: "resnet_block", kind: "block" }),
      params: { blocks: 3, filters: 64 }
    }).data;

    expect(blockRepeat(data)).toBe(3);
  });
});
