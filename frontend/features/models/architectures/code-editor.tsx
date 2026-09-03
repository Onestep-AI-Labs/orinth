"use client";

import { useMemo } from "react";
import { AlertTriangle, Info } from "lucide-react";
import { PythonEditor } from "@/features/platform/code/python-editor";
import { pythonApiCompletions } from "@/features/platform/code/python-api";

export type LintFinding = { line: number; severity: "error" | "warning"; message: string };

/**
 * The Python field for a custom layer or a custom function.
 *
 * This was a transparent `<textarea>` over a highlighted `<pre>` — enough for
 * colour, and nothing else. Which meant the one place in the product where you
 * write Python *by hand, against an API you half-remember* had no completion,
 * no bracket matching, and a Tab key that inserted four spaces wherever the
 * caret happened to be, including at the start of a `return`. The notebook next
 * door had all of it. Now they are the same editor: `PythonEditor` supplies the
 * mechanics, and this supplies the two things specific to a layer body —
 * completion over the Keras and torch surface (there is no kernel here to ask,
 * so the table in `python-api.ts` is the only source available) and the lint
 * findings, marked in the gutter and listed underneath.
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
  const completions = useMemo(() => [pythonApiCompletions()], []);

  return (
    <div className="code-editor">
      <div className="code-editor-frame">
        <PythonEditor
          value={value}
          onChange={onChange}
          completions={completions}
          diagnostics={findings}
          ariaLabel={ariaLabel}
          placeholder={minRows > 4 ? "class MyLayer(tf.keras.layers.Layer): ..." : "tf.nn.gelu(x)"}
          // A one-expression function does not need ten lines of empty field,
          // and a layer body cannot be written in three.
          minHeight={`${Math.max(minRows, 3) * 1.55}em`}
          maxHeight="40em"
        />
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
