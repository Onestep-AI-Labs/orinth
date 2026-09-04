import type { CompletionContext, CompletionResult } from "@codemirror/autocomplete";
import type { NotebookKernel } from "@/features/notebooks/kernel-client";
import { stripAnsi } from "@/features/notebooks/ansi";

/**
 * Wiring the kernel's `complete_request` into CodeMirror.
 *
 * The kernel is the only thing that can answer usefully. It sees the live
 * namespace, so after `df = orinth.datasets.load(…)` it knows what `df.`
 * offers; a static keyword list never will, and buffer-scraped words offer
 * strings that are attributes of nothing.
 *
 * The cost is that every completion is a round trip. Two things keep that
 * tolerable: it only fires on an explicit request or after a `.`, and a
 * request that overlaps another is abandoned — `context.aborted` is checked
 * after the await, because by then the user has usually typed another
 * character and the answer is already stale.
 */

//: A trailing `.` is the one place completion is worth firing unasked: it is
//: unambiguous intent, and it is where a namespace is genuinely unguessable.
const IMPLICIT = /\.$/;

//: `_private` and `__dunder__` are real attributes and legitimately completable,
//: but putting them first buries the ones anyone wanted.
function rank(label: string): number {
  if (label.startsWith("__")) return 2;
  if (label.startsWith("_")) return 1;
  return 0;
}

export function kernelCompletions(getKernel: () => NotebookKernel | null) {
  return async (context: CompletionContext): Promise<CompletionResult | null> => {
    const kernel = getKernel();
    if (!kernel) return null;

    const before = context.state.doc.sliceString(0, context.pos);
    if (!context.explicit && !IMPLICIT.test(before)) return null;

    const reply = await kernel.complete(context.state.doc.toString(), context.pos);
    // The user has typed since this was sent; a popup built from a stale
    // namespace is worse than no popup.
    if (!reply || context.aborted) return null;

    const sorted = [...reply.matches].sort(
      (a, b) => rank(a) - rank(b) || a.localeCompare(b)
    );

    return {
      from: reply.start,
      to: reply.end,
      options: sorted.map((label) => ({
        label,
        // The kernel returns a flat list with no kinds, so claiming one would
        // be inventing information the icon then asserts.
        type: label.endsWith("(") ? "function" : undefined,
        boost: -rank(label)
      })),
      // The kernel already filtered against the prefix it was given; letting
      // CodeMirror filter again drops matches on `_`-prefixed and dotted
      // labels whose shape its matcher does not expect.
      filter: false
    };
  };
}

/** Shift-Tab: the short docstring for whatever is under the cursor. */
export async function inspectAt(
  kernel: NotebookKernel | null,
  code: string,
  cursor: number
): Promise<string | null> {
  if (!kernel) return null;
  const text = await kernel.inspect(code, cursor);
  // Kernel docstrings arrive ANSI-coloured for a terminal; the popup renders
  // as plain text, so the escapes have to come off or they show as `[0;31m`.
  return text ? stripAnsi(text) : null;
}
