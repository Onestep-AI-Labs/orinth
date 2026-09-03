import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { CodeEditor, lintCustomFunction, lintCustomLayer } from "./code-editor";

/**
 * The custom-layer field used to be a textarea with a highlighted `<pre>`
 * behind it. These pin the two things that changed when it became the same
 * editor the notebook uses: it is a real editor, and a finding marks its line
 * rather than only appearing in the list underneath.
 */

const LAYER = `class ScaledAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def call(self, x):
        return tf.nn.gelu(x) * self.units
`;

describe("CodeEditor", () => {
  it("mounts a real editor, not a textarea", () => {
    const { container } = render(
      <CodeEditor value={LAYER} onChange={vi.fn()} findings={[]} ariaLabel="Layer source" />
    );

    expect(container.querySelector(".cm-editor")).toBeTruthy();
    expect(container.querySelector("textarea.code-editor-input")).toBeNull();
    // Line numbers and the diagnostic margin are what make it readable at
    // twenty lines; both come from the shared editor.
    expect(container.querySelector(".cm-lineNumbers")).toBeTruthy();
    expect(container.querySelector(".cm-diagnostic-gutter")).toBeTruthy();
    expect(container.textContent).toContain("ScaledAttention");
  });

  it("marks a flagged line in the margin and lists the finding", () => {
    const { container } = render(
      <CodeEditor
        value={LAYER}
        onChange={vi.fn()}
        findings={[{ line: 2, severity: "warning", message: "Missing super().__init__" }]}
        ariaLabel="Layer source"
      />
    );

    expect(container.querySelector(".cm-diagnostic-warning")).toBeTruthy();
    expect(screen.getByText("Missing super().__init__")).toBeTruthy();
  });
});

describe("lintCustomLayer", () => {
  it("passes a layer that is actually a layer", () => {
    expect(lintCustomLayer(LAYER, "ScaledAttention")).toEqual([]);
  });

  it("catches the name in the field disagreeing with the source", () => {
    const findings = lintCustomLayer(LAYER, "SomethingElse");
    expect(findings.some((finding) => finding.severity === "error")).toBe(true);
  });

  it("catches a layer with no call()", () => {
    const findings = lintCustomLayer(
      "class Thing(tf.keras.layers.Layer):\n    pass\n",
      "Thing"
    );
    expect(findings.map((finding) => finding.message).join(" ")).toContain("call()");
  });
});

describe("lintCustomFunction", () => {
  it("accepts one expression over x", () => {
    expect(lintCustomFunction("tf.nn.gelu(x)")).toEqual([]);
  });

  it("refuses a statement", () => {
    expect(lintCustomFunction("y = tf.nn.gelu(x)").length).toBeGreaterThan(0);
  });
});
