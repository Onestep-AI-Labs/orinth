"use client";

import { useMemo, useRef, useState } from "react";
import { PrismAsync as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/cjs/styles/prism";
import { AlertTriangle, Info } from "lucide-react";

export type LintFinding = { line: number; severity: "error" | "warning"; message: string };

/**
 * An editable Python field with real syntax highlighting.
 *
 * A transparent textarea sits exactly on top of a highlighted `<pre>`, both
 * sharing one font metric and scroll offset. The user types into the textarea
 * and reads the highlighted layer beneath — the standard way to get syntax
 * colour without shipping a full editor engine, which for a twenty-line layer
 * body would be far more weight than the job needs.
 */
export function CodeEditor({
  value,
  onChange,
  findings,
  minRows = 8,
  ariaLabel
}: {
  value: string;
  onChange: (value: string) => void;
  findings: LintFinding[];
  minRows?: number;
  ariaLabel: string;
}) {
  const [scroll, setScroll] = useState({ top: 0, left: 0 });
  const textarea = useRef<HTMLTextAreaElement>(null);
  const lines = value.split("\n");
  const rows = Math.min(Math.max(lines.length + 1, minRows), 40);

  // Tab should indent rather than leave the field — the single thing that most
  // makes a textarea feel unusable for code.
  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Tab") return;
    event.preventDefault();
    const element = event.currentTarget;
    const { selectionStart, selectionEnd } = element;
    const next = `${value.slice(0, selectionStart)}    ${value.slice(selectionEnd)}`;
    onChange(next);
    requestAnimationFrame(() => {
      element.selectionStart = element.selectionEnd = selectionStart + 4;
    });
  };

  return (
    <div className="code-editor">
      <div className="code-editor-frame" style={{ height: `${rows * 1.55}em` }}>
        <div className="code-editor-gutter" style={{ transform: `translateY(-${scroll.top}px)` }}>
          {lines.map((_line, index) => (
            <span
              key={index}
              className={
                findings.some((finding) => finding.line === index + 1)
                  ? "code-editor-line code-editor-line-flagged"
                  : "code-editor-line"
              }
            >
              {index + 1}
            </span>
          ))}
        </div>
        <div className="code-editor-surface">
          <div
            className="code-editor-highlight"
            aria-hidden
            style={{ transform: `translate(-${scroll.left}px, -${scroll.top}px)` }}
          >
            <SyntaxHighlighter
              language="python"
              style={oneLight}
              customStyle={{
                margin: 0,
                padding: 0,
                background: "transparent",
                fontSize: "0.75rem",
                lineHeight: 1.55,
                overflow: "visible"
              }}
            >
              {`${value}\n`}
            </SyntaxHighlighter>
          </div>
          <textarea
            ref={textarea}
            className="code-editor-input"
            aria-label={ariaLabel}
            value={value}
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
            onKeyDown={onKeyDown}
            onScroll={(event) =>
              setScroll({
                top: event.currentTarget.scrollTop,
                left: event.currentTarget.scrollLeft
              })
            }
            onChange={(event) => onChange(event.target.value)}
          />
        </div>
      </div>
      {findings.length > 0 && (
        <ul className="code-editor-findings">
          {findings.map((finding, index) => (
            <li key={index} className={`code-finding code-finding-${finding.severity}`}>
              {finding.severity === "error" ? (
                <AlertTriangle size={13} aria-hidden />
              ) : (
                <Info size={13} aria-hidden />
              )}
              <span className="code-finding-line">L{finding.line}</span>
              <span>{finding.message}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * Static checks for a custom-layer body, run as you type.
 *
 * Deliberately shallow: these are the mistakes that actually happen when
 * writing a Keras layer in a text field, and each one is cheap to detect
 * without parsing Python. Anything subtler surfaces as a real traceback when
 * the graph trains — that is the authoritative check, and this only exists to
 * catch the obvious things before you get there.
 */
export function lintCustomLayer(source: string, expectedClass: string): LintFinding[] {
  const findings: LintFinding[] = [];
  const lines = source.split("\n");
  if (!source.trim()) {
    return [{ line: 1, severity: "error", message: "The layer has no source." }];
  }

  const classMatches = [...source.matchAll(/^\s*class\s+([A-Za-z_]\w*)\s*\(([^)]*)\)/gm)];
  if (classMatches.length === 0) {
    findings.push({ line: 1, severity: "error", message: "No class definition found." });
  }
  const declared = expectedClass.trim();
  if (declared && classMatches.length > 0 && !classMatches.some((m) => m[1] === declared)) {
    findings.push({
      line: 1,
      severity: "error",
      message: `The class name field says “${declared}” but the source defines ${classMatches
        .map((m) => m[1])
        .join(", ")}.`
    });
  }
  for (const match of classMatches) {
    if (!/Layer|Module/.test(match[2])) {
      findings.push({
        line: lineOf(source, match.index ?? 0),
        severity: "warning",
        message: `${match[1]} does not subclass tf.keras.layers.Layer — Keras will not treat it as a layer.`
      });
    }
  }
  if (classMatches.length > 0 && !/def\s+call\s*\(/.test(source)) {
    findings.push({
      line: 1,
      severity: "error",
      message: "No call() method — a Keras layer needs one to do anything."
    });
  }
  if (/def\s+__init__/.test(source) && !/super\(\s*\)\.__init__/.test(source)) {
    findings.push({
      line: lineOf(source, source.indexOf("def __init__")),
      severity: "warning",
      message: "__init__ does not call super().__init__(**kwargs); Keras needs it to register the layer."
    });
  }

  lines.forEach((line, index) => {
    if (/^\t/.test(line)) {
      findings.push({
        line: index + 1,
        severity: "warning",
        message: "Tab indentation — mixing tabs and spaces is a syntax error in Python."
      });
    }
  });

  const unbalanced = unbalancedLine(lines);
  if (unbalanced) {
    findings.push({
      line: unbalanced,
      severity: "error",
      message: "Unbalanced bracket or parenthesis."
    });
  }
  return findings.slice(0, 6);
}

/** Static checks for a one-expression custom function. */
export function lintCustomFunction(expression: string): LintFinding[] {
  const findings: LintFinding[] = [];
  if (!expression.trim()) {
    return [{ line: 1, severity: "error", message: "The function has no expression." }];
  }
  if (expression.includes("\n")) {
    findings.push({
      line: 2,
      severity: "error",
      message: "Must be a single expression. Use a Custom layer for multi-line logic."
    });
  }
  if (/\breturn\b|\bimport\b|=(?!=)/.test(expression)) {
    findings.push({
      line: 1,
      severity: "error",
      message: "Statements are not allowed — write only the expression, e.g. tf.nn.gelu(x)."
    });
  }
  if (!/\bx\b/.test(expression)) {
    findings.push({
      line: 1,
      severity: "warning",
      message: "The expression never uses x, so the incoming tensor is discarded."
    });
  }
  if (unbalancedLine([expression])) {
    findings.push({ line: 1, severity: "error", message: "Unbalanced bracket or parenthesis." });
  }
  return findings;
}

function lineOf(source: string, index: number): number {
  return source.slice(0, index).split("\n").length;
}

function unbalancedLine(lines: string[]): number | null {
  const pairs: Record<string, string> = { ")": "(", "]": "[", "}": "{" };
  const stack: Array<{ char: string; line: number }> = [];
  lines.forEach((line, index) => {
    // Strings and comments are skipped crudely; a full tokenizer would be far
    // more than a bracket check warrants.
    const code = line.replace(/(['"]).*?\1/g, "").replace(/#.*$/, "");
    for (const char of code) {
      if ("([{".includes(char)) stack.push({ char, line: index + 1 });
      else if (char in pairs) {
        if (stack.length === 0 || stack[stack.length - 1].char !== pairs[char]) {
          return;
        }
        stack.pop();
      }
    }
  });
  return stack.length > 0 ? stack[0].line : null;
}
