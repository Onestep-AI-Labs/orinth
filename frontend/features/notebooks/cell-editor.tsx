"use client";

import { useEffect, useRef } from "react";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, lineNumbers } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { Prec } from "@codemirror/state";

/**
 * One code cell, editable.
 *
 * CodeMirror 6 rather than a `<textarea>`: Python is whitespace-significant, so
 * an editor without indent-aware Tab and bracket matching makes writing a loop
 * genuinely unpleasant, and that is the whole activity here.
 *
 * The theme is written from platform tokens rather than imported — a vendor
 * theme is exactly the "no vendor theme enters the app" line phase 17 drew for
 * React Flow, and for the same reason: two styling systems on one screen read
 * as two applications.
 */

const theme = EditorView.theme({
  "&": {
    fontSize: "13px",
    backgroundColor: "transparent",
    color: "var(--color-ink)"
  },
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
  ".cm-activeLineGutter": { backgroundColor: "transparent" },
  ".cm-cursor": { borderLeftColor: "var(--color-ink)" },
  "&.cm-focused .cm-selectionBackground, ::selection": {
    backgroundColor: "var(--color-accent-tint-strong)"
  }
});

export function CellEditor({
  value,
  onChange,
  onRun,
  readOnly = false
}: {
  value: string;
  onChange: (value: string) => void;
  onRun: () => void;
  readOnly?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  // Held in a ref so the run keybinding always calls the current handler
  // without rebuilding the editor — recreating the view on every render would
  // lose the cursor on each keystroke.
  const runRef = useRef(onRun);
  const changeRef = useRef(onChange);
  runRef.current = onRun;
  changeRef.current = onChange;

  useEffect(() => {
    if (!host.current || view.current) return;
    const editor = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: value,
        extensions: [
          lineNumbers(),
          history(),
          python(),
          theme,
          EditorState.readOnly.of(readOnly),
          // Highest precedence so Shift-Enter reaches this before CodeMirror's
          // own newline binding claims it.
          Prec.highest(
            keymap.of([
              {
                key: "Shift-Enter",
                run: () => {
                  runRef.current();
                  return true;
                }
              },
              {
                key: "Mod-Enter",
                run: () => {
                  runRef.current();
                  return true;
                }
              }
            ])
          ),
          keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) changeRef.current(update.state.doc.toString());
          })
        ]
      })
    });
    view.current = editor;
    return () => {
      editor.destroy();
      view.current = null;
    };
    // Deliberately mounted once: `value` is the initial document, and the
    // effect below reconciles later external changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const editor = view.current;
    if (!editor) return;
    const current = editor.state.doc.toString();
    // Only reconcile when the value genuinely diverged — echoing every local
    // keystroke back through here would reset the selection mid-word.
    if (current === value) return;
    editor.dispatch({ changes: { from: 0, to: current.length, insert: value } });
  }, [value]);

  return <div className="nb-cell-editor" ref={host} />;
}
