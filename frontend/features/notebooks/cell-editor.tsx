"use client";

import { useEffect, useRef, useState } from "react";
import { EditorState, Prec } from "@codemirror/state";
import {
  EditorView,
  drawSelection,
  highlightActiveLine,
  highlightActiveLineGutter,
  highlightSpecialChars,
  keymap,
  lineNumbers,
  rectangularSelection
} from "@codemirror/view";
import {
  defaultKeymap,
  history,
  historyKeymap,
  indentWithTab
} from "@codemirror/commands";
import {
  acceptCompletion,
  autocompletion,
  closeBrackets,
  closeBracketsKeymap,
  completionKeymap,
  startCompletion
} from "@codemirror/autocomplete";
import { bracketMatching, indentOnInput, indentUnit } from "@codemirror/language";
import { python } from "@codemirror/lang-python";
import { kernelCompletions, inspectAt } from "@/features/notebooks/completion";
import { pythonHighlighting } from "@/features/notebooks/syntax";
import type { NotebookKernel } from "@/features/notebooks/kernel-client";

/**
 * One code cell: an editor, not a textarea.
 *
 * Python is whitespace-significant, so indent-aware Tab is not a nicety, and
 * the first cut shipped without two things that turned out to matter more than
 * either of us expected:
 *
 * - **Syntax colour.** `python()` was parsing and nothing consumed the tree, so
 *   every cell rendered flat. See `syntax.ts`.
 * - **Completion from the kernel.** The kernel introspects the live namespace,
 *   which is the only way `df.` can offer polars methods after
 *   `df = orinth.datasets.load(…)`. A static keyword list cannot know that.
 *
 * The theme is written from platform tokens rather than imported. A vendor
 * theme is the line phase 17 drew for React Flow, and for the same reason: two
 * styling systems on one screen read as two applications.
 */

const theme = EditorView.theme({
  "&": { fontSize: "13px", backgroundColor: "transparent", color: "var(--color-ink)" },
  ".cm-content": {
    fontFamily: "var(--font-mono, SFMono-Regular, Consolas, monospace)",
    padding: "10px 0"
  },
  "&.cm-focused": { outline: "none" },
  ".cm-gutters": {
    backgroundColor: "transparent",
    border: "none",
    color: "var(--color-ink-subtle)",
    fontSize: "11px"
  },
  ".cm-activeLine": { backgroundColor: "var(--color-wash)" },
  ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--color-ink-muted)" },
  ".cm-cursor": { borderLeftColor: "var(--color-ink)" },
  "&.cm-focused .cm-selectionBackground, ::selection, .cm-selectionBackground": {
    backgroundColor: "var(--color-accent-tint-strong)"
  },
  // A matched bracket is marked with a border rather than a fill: a filled
  // bracket at 13px reads as a selection, which it is not.
  ".cm-matchingBracket": {
    outline: "1px solid var(--color-accent)",
    backgroundColor: "transparent"
  },
  ".cm-nonmatchingBracket": { outline: "1px solid var(--color-danger)" },
  ".cm-tooltip": {
    border: "1px solid var(--color-line)",
    borderRadius: "var(--radius-sm)",
    backgroundColor: "var(--color-surface)",
    color: "var(--color-ink)",
    boxShadow: "var(--shadow-overlay)"
  },
  ".cm-tooltip-autocomplete ul li": {
    fontFamily: "var(--font-mono, monospace)",
    fontSize: "12px",
    padding: "3px 8px"
  },
  ".cm-tooltip-autocomplete ul li[aria-selected]": {
    backgroundColor: "var(--color-accent-tint)",
    color: "var(--color-accent-strong)"
  }
});

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
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const [doc, setDoc] = useState<string | null>(null);

  // Held in refs so the keymap always calls the current handler without
  // rebuilding the editor — recreating the view on every render would lose the
  // cursor on each keystroke.
  const runRef = useRef(onRun);
  const changeRef = useRef(onChange);
  const kernelRef = useRef(getKernel);
  runRef.current = onRun;
  changeRef.current = onChange;
  kernelRef.current = getKernel;

  useEffect(() => {
    if (!host.current || view.current) return;
    const editor = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: value,
        extensions: [
          lineNumbers(),
          highlightActiveLine(),
          highlightActiveLineGutter(),
          highlightSpecialChars(),
          drawSelection(),
          rectangularSelection(),
          history(),
          python(),
          pythonHighlighting,
          bracketMatching(),
          closeBrackets(),
          indentOnInput(),
          // PEP 8. CodeMirror defaults to two, which is wrong for Python and
          // produces files that look broken outside the browser.
          indentUnit.of("    "),
          autocompletion({
            override: kernelRef.current ? [kernelCompletions(() => kernelRef.current?.() ?? null)] : [],
            activateOnTyping: true,
            // The kernel already ordered them; re-sorting by CodeMirror's own
            // score would put `__class__` above the attribute someone wanted.
            defaultKeymap: false,
            icons: false
          }),
          theme,
          EditorState.readOnly.of(readOnly),
          Prec.highest(
            keymap.of([
              // Shift-Enter and Cmd/Ctrl-Enter run the cell. Highest precedence
              // so they reach here before CodeMirror's newline binding, and
              // before the completion keymap claims Enter.
              { key: "Shift-Enter", run: () => (runRef.current(), true) },
              { key: "Mod-Enter", run: () => (runRef.current(), true) },
              // Tab accepts a completion when one is open and indents
              // otherwise, which is the behaviour every notebook has.
              { key: "Tab", run: acceptCompletion },
              { key: "Mod-Space", run: startCompletion },
              {
                key: "Shift-Tab",
                run: (target) => {
                  void showInspect(target, kernelRef.current?.() ?? null);
                  return true;
                }
              }
            ])
          ),
          keymap.of([...closeBracketsKeymap, ...completionKeymap, ...defaultKeymap, ...historyKeymap, indentWithTab]),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) {
              const next = update.state.doc.toString();
              setDoc(next);
              changeRef.current(next);
            }
          })
        ]
      })
    });
    view.current = editor;
    return () => {
      editor.destroy();
      view.current = null;
    };
    // Mounted once: `value` is the initial document and the effect below
    // reconciles later external changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const editor = view.current;
    if (!editor) return;
    const current = editor.state.doc.toString();
    // Only reconcile a genuine divergence. Echoing every local keystroke back
    // through here would reset the selection mid-word.
    if (current === value || doc === value) return;
    editor.dispatch({ changes: { from: 0, to: current.length, insert: value } });
  }, [doc, value]);

  return <div className="nb-cell-editor" ref={host} />;
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
