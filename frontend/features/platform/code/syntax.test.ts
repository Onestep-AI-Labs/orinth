import { describe, expect, it } from "vitest";
import { EditorState } from "@codemirror/state";
import { python } from "@codemirror/lang-python";
import { highlightTree } from "@lezer/highlight";
import { syntaxTree } from "@codemirror/language";
import { orinthHighlight } from "@/features/platform/code/syntax";

/**
 * The first cut shipped `python()` with no HighlightStyle, so the parser built
 * a tree nothing consumed and every cell rendered flat. These assert the tree
 * is actually being styled, which is the thing that regressed silently.
 */

function classesFor(code: string): Map<string, string> {
  const state = EditorState.create({ doc: code, extensions: [python()] });
  const found = new Map<string, string>();
  highlightTree(syntaxTree(state), orinthHighlight, (from, to, classes) => {
    found.set(state.doc.sliceString(from, to), classes);
  });
  return found;
}

describe("orinthHighlight", () => {
  it("styles keywords, strings, numbers, and comments distinctly", () => {
    const found = classesFor("# note\ndef run(x=1):\n    return 'ok'\n");

    expect(found.get("def")).toBeTruthy();
    expect(found.get("'ok'")).toBeTruthy();
    expect(found.get("1")).toBeTruthy();
    expect(found.get("# note")).toBeTruthy();

    // The point of colour here is that these are *different*. A style sheet
    // that assigned one class to everything would still pass a "is it styled"
    // check, so the distinctions are what get asserted.
    const distinct = new Set([found.get("def"), found.get("'ok'"), found.get("1"), found.get("# note")]);
    expect(distinct.size).toBe(4);
  });

  it("styles a class definition and a decorator", () => {
    const found = classesFor("@dataclass\nclass Thing:\n    pass\n");
    expect(found.get("class")).toBeTruthy();
    expect(found.get("dataclass") || found.get("@")).toBeTruthy();
  });

  it("assigns every colour from a token, never a literal", () => {
    // DESIGN.md §2: no hex, rgb, or hsl anywhere in styles.
    for (const spec of orinthHighlight.specs) {
      const color = String((spec as { color?: string }).color ?? "");
      if (!color) continue;
      expect(color).toMatch(/^(oklch\(var\(--|var\(--)/);
    }
  });
});
