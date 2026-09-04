"use client";

import { useMemo, useRef } from "react";
import type { EditorView } from "@codemirror/view";
import { PythonEditor } from "@/features/platform/code/python-editor";
import { kernelCompletions, inspectAt } from "@/features/notebooks/completion";
import type { NotebookKernel } from "@/features/notebooks/kernel-client";

/**
 * One code cell: an editor, not a textarea.
 *
 * Everything structural — highlighting, 4-space indent, bracket matching,
 * history, the theme from platform tokens — lives in `PythonEditor`, which the
 * architecture studio's custom-layer field also uses. What is left here is the
 * part that is genuinely about a *notebook*:
 *
 * - **Shift-Enter and Cmd-Enter run the cell**, at the highest precedence so
 *   they arrive before CodeMirror's newline binding and before the completion
 *   keymap claims Enter.
 * - **Completion from the kernel.** It introspects the live namespace, which is
 *   the only way `df.` can offer polars methods after
 *   `df = orinth.datasets.load(…)`. A static list cannot know that.
 * - **Shift-Tab inspects**, through the kernel's `inspect_request`.
 */
export function CellEditor({
  value,
  onChange,
  onRun,
  getKernel,
  readOnly = false
}: {
  value: string;
  onChange: (value: string) => void;
  onRun: () => void;
  getKernel?: () => NotebookKernel | null;
  readOnly?: boolean;
}) {
  // Refs so the bindings and the completion source stay stable for the editor's
  // life while still reaching the current handler.
  const runRef = useRef(onRun);
  const kernelRef = useRef(getKernel);
  runRef.current = onRun;
  kernelRef.current = getKernel;

  const keys = useMemo(
    () => [
      { key: "Shift-Enter", run: () => (runRef.current(), true) },
      { key: "Mod-Enter", run: () => (runRef.current(), true) },
      {
        key: "Shift-Tab",
        run: (target: EditorView) => {
          void showInspect(target, kernelRef.current?.() ?? null);
          return true;
        }
      }
    ],
    []
  );

  const hasKernel = Boolean(getKernel);
  const completions = useMemo(
    () => (hasKernel ? [kernelCompletions(() => kernelRef.current?.() ?? null)] : []),
    [hasKernel]
  );

  return (
    <PythonEditor
      className="nb-cell-editor code-surface"
      value={value}
      onChange={onChange}
      keys={keys}
      completions={completions}
      readOnly={readOnly}
      ariaLabel="Code cell"
    />
  );
}

/**
 * Shift-Tab: the kernel's docstring for whatever is under the cursor.
 *
 * Rendered as a plain element positioned by CodeMirror rather than as a
 * `Tooltip` extension, because the content arrives asynchronously and a
 * StateField that has to be updated from a promise is more machinery than one
 * absolutely-positioned node.
 */
async function showInspect(view: EditorView, kernel: NotebookKernel | null) {
  const existing = view.dom.querySelector(".nb-inspect");
  existing?.remove();
  if (!kernel) return;

  const text = await inspectAt(kernel, view.state.doc.toString(), view.state.selection.main.head);
  if (!text) return;

  const node = document.createElement("pre");
  node.className = "nb-inspect";
  node.textContent = text;
  node.addEventListener("click", () => node.remove());
  view.dom.appendChild(node);

  const dismiss = (event: KeyboardEvent) => {
    if (event.key !== "Escape") return;
    node.remove();
    document.removeEventListener("keydown", dismiss);
  };
  document.addEventListener("keydown", dismiss);
}
