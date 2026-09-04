/**
 * SGR escape sequences → token-coloured spans.
 *
 * Kernel output is full of these: pytest's red F, a traceback's highlighted
 * frame, tqdm's bars. Stripping them loses the only structure the output has;
 * rendering them raw shows `[0;31m` to the user. Neither is acceptable, and the
 * parser is forty lines, so it is ours rather than a dependency.
 *
 * Colours map onto the `--label-*` data-visualization ramp — the one sanctioned
 * exception to the single-accent rule (DESIGN.md §2), because these *are*
 * categorical values encoded by colour, and a traceback where red and green
 * mean the same thing is not a traceback.
 */

export type AnsiSpan = { text: string; className: string };

//: SGR code → the class carrying its token. Only the eight base colours, their
//: bright variants, and bold: kernels in practice emit nothing else, and a full
//: 256-colour table would be dead code with a maintenance cost.
const COLORS: Record<number, string> = {
  30: "nb-ansi-black",
  31: "nb-ansi-red",
  32: "nb-ansi-green",
  33: "nb-ansi-yellow",
  34: "nb-ansi-blue",
  35: "nb-ansi-magenta",
  36: "nb-ansi-cyan",
  37: "nb-ansi-white",
  90: "nb-ansi-black",
  91: "nb-ansi-red",
  92: "nb-ansi-green",
  93: "nb-ansi-yellow",
  94: "nb-ansi-blue",
  95: "nb-ansi-magenta",
  96: "nb-ansi-cyan",
  97: "nb-ansi-white"
};

// eslint-disable-next-line no-control-regex
const PATTERN = /\x1b\[([0-9;]*)m/g;

export function parseAnsi(input: string): AnsiSpan[] {
  const spans: AnsiSpan[] = [];
  let cursor = 0;
  let color = "";
  let bold = false;

  const push = (text: string) => {
    if (!text) return;
    const className = [color, bold ? "nb-ansi-bold" : ""].filter(Boolean).join(" ");
    spans.push({ text, className });
  };

  for (const match of input.matchAll(PATTERN)) {
    push(input.slice(cursor, match.index));
    cursor = (match.index ?? 0) + match[0].length;
    // An empty parameter list (`ESC[m`) means reset, same as `0`.
    for (const raw of (match[1] || "0").split(";")) {
      const code = Number(raw || "0");
      if (code === 0) {
        color = "";
        bold = false;
      } else if (code === 1) {
        bold = true;
      } else if (code === 22) {
        bold = false;
      } else if (code === 39) {
        color = "";
      } else if (COLORS[code]) {
        color = COLORS[code];
      }
      // Everything else — backgrounds, underline, 256-colour selectors — is
      // dropped rather than approximated. A wrong colour is worse than none.
    }
  }
  push(input.slice(cursor));
  return spans;
}

/** Strip every escape, for places that need the text and not the styling. */
export function stripAnsi(input: string): string {
  return input.replace(PATTERN, "");
}
