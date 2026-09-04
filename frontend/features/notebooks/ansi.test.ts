import { describe, expect, it } from "vitest";
import { parseAnsi, stripAnsi } from "@/features/notebooks/ansi";

const ESC = "";

describe("parseAnsi", () => {
  it("returns plain text as one unstyled span", () => {
    expect(parseAnsi("hello")).toEqual([{ text: "hello", className: "" }]);
  });

  it("colours the text an escape opened", () => {
    const spans = parseAnsi(`${ESC}[31mFAILED${ESC}[0m ok`);
    expect(spans).toEqual([
      { text: "FAILED", className: "nb-ansi-red" },
      { text: " ok", className: "" }
    ]);
  });

  it("carries bold alongside a colour", () => {
    const [span] = parseAnsi(`${ESC}[1;32mPASSED`);
    expect(span.className).toContain("nb-ansi-green");
    expect(span.className).toContain("nb-ansi-bold");
  });

  it("treats a bare reset as a full reset", () => {
    // `ESC[m` is the empty-parameter form and means the same as `ESC[0m`.
    const spans = parseAnsi(`${ESC}[31mred${ESC}[mplain`);
    expect(spans[1]).toEqual({ text: "plain", className: "" });
  });

  it("maps bright variants onto the same tokens as their base colours", () => {
    expect(parseAnsi(`${ESC}[91mx`)[0].className).toBe("nb-ansi-red");
  });

  it("drops a code it does not model rather than approximating it", () => {
    // Backgrounds and 256-colour selectors are not represented; a wrong colour
    // would be worse than none, and the text must still survive.
    expect(parseAnsi(`${ESC}[48;5;196mtext`)).toEqual([{ text: "text", className: "" }]);
  });

  it("keeps a traceback's frames distinguishable", () => {
    const traceback = `${ESC}[0;31mValueError${ESC}[0m: ${ESC}[0;32mboom${ESC}[0m`;
    const spans = parseAnsi(traceback);
    expect(spans.map((span) => span.text)).toEqual(["ValueError", ": ", "boom"]);
    expect(spans[0].className).toBe("nb-ansi-red");
    expect(spans[2].className).toBe("nb-ansi-green");
  });

  it("does not lose text that has no escapes at all", () => {
    const noisy = "line one\nline two\n";
    expect(parseAnsi(noisy).map((span) => span.text).join("")).toBe(noisy);
  });
});

describe("stripAnsi", () => {
  it("removes every escape and keeps the text", () => {
    expect(stripAnsi(`${ESC}[1;31mred${ESC}[0m text`)).toBe("red text");
  });
});
