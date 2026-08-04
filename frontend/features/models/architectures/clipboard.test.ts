import { describe, expect, it } from "vitest";
import type { Edge } from "@xyflow/react";
import { instantiate, parsePayload, selectionPayload, serializePayload } from "./clipboard";
import type { ArchFlowNode, StoredGroup } from "./graph-state";
import type { NodeSpec } from "@/types/api";

function spec(type: string): NodeSpec {
  return {
    type,
    name: type,
    category: "Core",
    description: "",
    params: [],
    inputs: [{ key: "in", label: "In" }],
    outputs: [{ key: "out", label: "Out" }],
    min_inputs: 1,
    max_inputs: 1,
    task_types: [],
    kind: "layer",
    source: ""
  };
}

function node(id: string, type: string, selected: boolean, x = 0, y = 0): ArchFlowNode {
  return {
    id,
    type: "arch",
    position: { x, y },
    selected,
    data: { spec: spec(type), params: { units: 64 }, label: id, shape: null, issues: [] }
  };
}

const group: StoredGroup = {
  id: "group_1",
  title: "Encoder",
  tint: 2,
  x: 10,
  y: 20,
  width: 400,
  height: 200
};

describe("selectionPayload", () => {
  it("returns null when nothing is selected", () => {
    expect(selectionPayload([node("a", "dense", false)], [], [], new Set())).toBeNull();
  });

  it("keeps only edges whose endpoints are both selected", () => {
    const nodes = [node("a", "dense", true), node("b", "relu", true), node("c", "dense", false)];
    const edges: Edge[] = [
      { id: "e_a_b", source: "a", target: "b" },
      { id: "e_b_c", source: "b", target: "c" }
    ];
    const payload = selectionPayload(nodes, edges, [], new Set());
    expect(payload?.nodes.map((item) => item.id)).toEqual(["a", "b"]);
    expect(payload?.edges.map((item) => item.id)).toEqual(["e_a_b"]);
  });

  it("carries selected group frames", () => {
    const payload = selectionPayload([], [], [group], new Set(["group_1"]));
    expect(payload?.groups).toEqual([group]);
  });
});

describe("parsePayload", () => {
  it("round-trips a serialized selection", () => {
    const payload = selectionPayload([node("a", "dense", true)], [], [], new Set());
    expect(parsePayload(serializePayload(payload!))?.nodes).toHaveLength(1);
  });

  it("ignores clipboard text that did not come from the canvas", () => {
    expect(parsePayload("just some copied prose")).toBeNull();
    expect(parsePayload('{"nodes": []}')).toBeNull();
  });
});

describe("instantiate", () => {
  it("re-ids nodes around what is already on the canvas and rewires internal edges", () => {
    const payload = selectionPayload(
      [node("dense_1", "dense", true, 100, 100), node("dense_2", "dense", true, 300, 100)],
      [{ id: "e_dense_1_dense_2", source: "dense_1", target: "dense_2" }],
      [],
      new Set()
    )!;
    const added = instantiate(payload, ["dense_1", "dense_2"], [], null);

    expect(added.nodes.map((item) => item.id)).toEqual(["dense_3", "dense_4"]);
    expect(added.edges).toEqual([
      expect.objectContaining({ source: "dense_3", target: "dense_4" })
    ]);
    // Offset from the original so the copy is visibly a second node.
    expect(added.nodes[0].position).toEqual({ x: 132, y: 132 });
  });

  it("anchors the paste at a supplied position", () => {
    const payload = selectionPayload(
      [node("dense_1", "dense", true, 100, 100), node("dense_2", "dense", true, 300, 180)],
      [],
      [],
      new Set()
    )!;
    const added = instantiate(payload, ["dense_1", "dense_2"], [], { x: 0, y: 0 });

    // The top-left of the selection lands on the pointer; relative layout holds.
    expect(added.nodes[0].position).toEqual({ x: 0, y: 0 });
    expect(added.nodes[1].position).toEqual({ x: 200, y: 80 });
  });

  it("gives pasted groups fresh ids", () => {
    const payload = selectionPayload([], [], [group], new Set(["group_1"]))!;
    const added = instantiate(payload, [], ["group_1"], null);
    expect(added.groups).toHaveLength(1);
    expect(added.groups[0].id).not.toBe("group_1");
    expect(added.groups[0].title).toBe("Encoder");
  });

  it("copies params rather than sharing the object", () => {
    const payload = selectionPayload([node("dense_1", "dense", true)], [], [], new Set())!;
    const added = instantiate(payload, ["dense_1"], [], null);
    added.nodes[0].data.params.units = 128;
    expect(payload.nodes[0].data.params.units).toBe(64);
  });
});
