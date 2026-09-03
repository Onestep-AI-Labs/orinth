"use client";

import { useEffect, useRef, useState } from "react";
import { EditorState, Prec, type Extension } from "@codemirror/state";
import {
  EditorView,
  drawSelection,
  gutter,
  GutterMarker,
  highlightActiveLine,
  highlightActiveLineGutter,
  highlightSpecialChars,
  keymap,
  lineNumbers,
  placeholder as placeholderExtension,
  rectangularSelection,
  type KeyBinding
} from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import {
  acceptCompletion,
  autocompletion,
  closeBrackets,
  closeBracketsKeymap,
  completionKeymap,
  startCompletion,
  type CompletionSource
} from "@codemirror/autocomplete";
import { bracketMatching, indentOnInput, indentUnit } from "@codemirror/language";
import { python } from "@codemirror/lang-python";
import { pythonHighlighting } from "@/features/platform/code/syntax";

/**
 * The Python editor, once, for every surface that edits Python.
 *
 * There are two of those — a notebook cell and a custom layer in the
 * architecture studio — and until this existed they were different editors: the
 * notebook had CodeMirror with completion and indent-aware Tab, the studio had
 * a transparent `<textarea>` over a highlighted `<pre>`. Which meant Tab did
 * something different depending on which Python you were typing, and only one
 * of them could complete anything. Python is whitespace-significant; an editor
 * that guesses at indentation is not a small difference.
 *
 * What varies between the two is passed in: the completion source (a live
 * kernel in a notebook, a static API list in the studio), extra key bindings,
 * and diagnostics. The rest — highlighting, 4-space indent, bracket matching,
 * history, the theme built from platform tokens — is the same because there is
 * no reason for it to differ.
 *
 * The theme is written from tokens rather than imported. A vendor theme is the
 * line phase 17 drew for React Flow, and for the same reason: two styling
 * systems on one screen read as two applications.
 */

export type EditorDiagnostic = {
  /** 1-indexed, matching what a traceback prints. */
  line: number;
  severity: "error" | "warning";
  message: string;
};

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
  ".cm-placeholder": { color: "var(--color-ink-subtle)" },
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
  },
  ".cm-completionDetail": { color: "var(--color-ink-subtle)", fontStyle: "normal" }
});

/**
 * A dot in its own gutter beside a flagged line.
 *
 * In the margin rather than under the text: a squiggle at 13px in a 200px-wide
 * inspector is indistinguishable from an underline, and the message itself is
 * already listed below the editor where it can be read.
 */
class DiagnosticMarker extends GutterMarker {
  constructor(private readonly severity: "error" | "warning") {
    super();
  }
  toDOM() {
    const node = document.createElement("span");
    node.className = `cm-diagnostic-dot cm-diagnostic-${this.severity}`;
    return node;
  }
}

const MARKERS = {
  error: new DiagnosticMarker("error"),
  warning: new DiagnosticMarker("warning")
};

function diagnosticGutter(getDiagnostics: () => EditorDiagnostic[]): Extension {
  return gutter({
    class: "cm-diagnostic-gutter",
    lineMarker(view, block) {
      const line = view.state.doc.lineAt(block.from).number;
      const found = getDiagnostics().filter((entry) => entry.line === line);
      if (found.length === 0) return null;
      return found.some((entry) => entry.severity === "error")
        ? MARKERS.error
        : MARKERS.warning;
    },
    // Without this the gutter is only recomputed on a document change, so a
    // finding that appears from a *sibling* field (the class-name mismatch)
    // would not mark its line until the next keystroke.
    lineMarkerChange: () => true
  });
}

export function PythonEditor({
  value,
  onChange,
  keys = [],
  completions = [],
  diagnostics = [],
  readOnly = false,
  placeholder,
  ariaLabel,
  className = "code-surface",
  minHeight,
  maxHeight
}: {
  value: string;
  onChange: (value: string) => void;
  /** Extra bindings, at the highest precedence — run, inspect, submit. */
  keys?: KeyBinding[];
  completions?: CompletionSource[];
  diagnostics?: EditorDiagnostic[];
  readOnly?: boolean;
  placeholder?: string;
  ariaLabel?: string;
  className?: string;
  minHeight?: string;
  maxHeight?: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const [doc, setDoc] = useState<string | null>(null);

  // Held in refs so the editor is built once and still calls the current
  // handlers — rebuilding the view on every render would lose the cursor on
  // each keystroke.
  const changeRef = useRef(onChange);
  const keysRef = useRef(keys);
  const completionsRef = useRef(completions);
  const diagnosticsRef = useRef(diagnostics);
  changeRef.current = onChange;
  keysRef.current = keys;
  completionsRef.current = completions;
  diagnosticsRef.current = diagnostics;

  useEffect(() => {
    if (!host.current || view.current) return;
    const editor = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: value,
        extensions: [
          lineNumbers(),
          diagnosticGutter(() => diagnosticsRef.current),
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
            override: [
              (context) => {
                for (const source of completionsRef.current) {
                  const result = source(context);
                  if (result) return result;
                }
                return null;
              }
            ],
            activateOnTyping: true,
            // Sources order their own results; re-scoring here would put
            // `__class__` above the attribute someone actually wanted.
            defaultKeymap: false,
            icons: false
          }),
          theme,
          EditorView.lineWrapping,
          EditorState.readOnly.of(readOnly),
          placeholder ? placeholderExtension(placeholder) : [],
          Prec.highest(
            keymap.of([
              ...keysRef.current,
              // Tab accepts a completion when one is open and indents
              // otherwise, which is the behaviour every notebook has.
              { key: "Tab", run: acceptCompletion },
              { key: "Mod-Space", run: startCompletion }
            ])
          ),
          keymap.of([
            ...closeBracketsKeymap,
            ...completionKeymap,
            ...defaultKeymap,
            ...historyKeymap,
            indentWithTab
          ]),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) {
              const next = update.state.doc.toString();
              setDoc(next);
              changeRef.current(next);
            }
          }),
          EditorView.contentAttributes.of(ariaLabel ? { "aria-label": ariaLabel } : {})
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

  // A finding can appear without the document changing — the studio lints the
  // class-name field against the body — so the gutter is asked to repaint.
  useEffect(() => {
    view.current?.dispatch({});
  }, [diagnostics]);

  return (
    <div
      className={className}
      ref={host}
      style={{
        ...(minHeight ? { minHeight } : {}),
        ...(maxHeight ? { maxHeight, overflow: "auto" } : {})
      }}
    />
  );
}
