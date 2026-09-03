import { describe, expect, it } from "vitest";
import { EditorState } from "@codemirror/state";
import { CompletionContext, type CompletionResult } from "@codemirror/autocomplete";
import { python } from "@codemirror/lang-python";
import { pythonApiCompletions } from "@/features/platform/code/python-api";

/**
 * The studio has no kernel to ask, so this table is the only thing standing
 * between a custom layer and typing `tf.keras.layers.Conv2D` from memory. These
 * assert the shape of the answer, not the size of the list — a list that has to
 * be complete would be stale within a release.
 */

function complete(code: string, explicit = false): CompletionResult | null {
  const state = EditorState.create({ doc: code, extensions: [python()] });
  const context = new CompletionContext(state, code.length, explicit);
  return pythonApiCompletions()(context);
}

function labels(result: CompletionResult | null): string[] {
  return (result?.options ?? []).map((option) => String(option.label));
}

describe("pythonApiCompletions", () => {
  it("completes a dotted namespace from its prefix", () => {
    const result = complete("x = tf.keras.layers.Con");
    expect(labels(result)).toContain("Conv2D");
    expect(labels(result)).toContain("Conv2DTranspose");
    // `from` points at the character after the last dot, so accepting a
    // completion replaces the member and not the namespace.
    expect(result?.from).toBe("x = tf.keras.layers.".length);
  });

  it("knows the members a Layer subclass actually has", () => {
    expect(labels(complete("self."))).toContain("add_weight");
    expect(labels(complete("self."))).toContain("compute_output_shape");
  });

  it("covers both frameworks the studio emits", () => {
    expect(labels(complete("nn."))).toContain("Conv2d");
    expect(labels(complete("torch."))).toContain("einsum");
    expect(labels(complete("tf.nn."))).toContain("gelu");
  });

  it("offers identifiers the author already wrote", () => {
    const result = complete("hidden_units = 32\nself.proj = hid");
    expect(labels(result)).toContain("hidden_units");
  });

  it("stays quiet on an unknown namespace rather than guessing", () => {
    expect(complete("mystery.")).toBeNull();
  });

  it("does not fire unasked on one character", () => {
    // A popup on every keystroke in a twenty-line layer body is noise; an
    // explicit Ctrl-Space still answers.
    expect(complete("t")).toBeNull();
    expect(labels(complete("t", true))).toContain("tf");
  });
});
