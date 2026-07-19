"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";
import { Button } from "@/features/platform/ui";

type TokenKind = "plain" | "comment" | "command";
type Token = { text: string; kind: TokenKind };
type CopyState = "idle" | "copied" | "failed";

/**
 * Shell-snippet highlighting, deliberately hand-rolled.
 *
 * `react-syntax-highlighter` is already a dependency, but every theme it ships
 * is a block of hardcoded hex that DESIGN.md forbids, and it would pull a
 * highlighter runtime onto a page that is otherwise static. These snippets are
 * a handful of commands, so three token roles cover them.
 */
function tokenizeLine(line: string): Token[] {
  if (!line.trim()) return [{ text: line, kind: "plain" }];

  const indent = line.match(/^\s*/)?.[0] ?? "";
  const body = line.slice(indent.length);
  if (body.startsWith("#")) return [{ text: line, kind: "comment" }];

  const tokens: Token[] = [];
  if (indent) tokens.push({ text: indent, kind: "plain" });

  // A trailing comment on a command line keeps its own role.
  const hash = body.indexOf(" #");
  const code = hash >= 0 ? body.slice(0, hash) : body;
  const comment = hash >= 0 ? body.slice(hash) : "";

  const space = code.indexOf(" ");
  if (space === -1) {
    tokens.push({ text: code, kind: "command" });
  } else {
    tokens.push({ text: code.slice(0, space), kind: "command" });
    tokens.push({ text: code.slice(space), kind: "plain" });
  }

  if (comment) tokens.push({ text: comment, kind: "comment" });
  return tokens;
}

export function CodeBlock({ code, label }: { code: string; label: string }) {
  const [state, setState] = useState<CopyState>("idle");
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Unmounting mid-countdown would otherwise set state on a dead component.
  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);

  async function copy() {
    if (timer.current) clearTimeout(timer.current);
    try {
      // Absent on http origins that aren't localhost, so this is a real branch,
      // not defensive padding — the failed state tells the user to copy by hand.
      if (!navigator.clipboard) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(code);
      setState("copied");
    } catch {
      setState("failed");
    }
    timer.current = setTimeout(() => setState("idle"), 2400);
  }

  return (
    <div className="landing-code-wrap">
      <div className="landing-code-copy">
        {/* The shared Button primitive, so hover/focus/active states and the
            token palette come from the system rather than a one-off. */}
        <Button
          variant="secondary"
          size="sm"
          onClick={copy}
          aria-label={state === "copied" ? `${label} copied` : `Copy ${label}`}
        >
          {state === "copied" ? <Check size={14} /> : <Copy size={14} />}
          {state === "copied" ? "Copied" : state === "failed" ? "Select to copy" : "Copy"}
        </Button>
      </div>
      <pre className="landing-code">
        <code>
          {code.split("\n").map((line, index) => (
            // Lines are static content, so the index is a stable key here.
            <span className="landing-code-line" key={index}>
              {tokenizeLine(line).map((token, tokenIndex) => (
                <span className={`landing-code-${token.kind}`} key={tokenIndex}>
                  {token.text}
                </span>
              ))}
              {"\n"}
            </span>
          ))}
        </code>
      </pre>
      <span aria-live="polite" className="sr-only">
        {state === "copied" ? "Copied to clipboard" : state === "failed" ? "Copy failed" : ""}
      </span>
    </div>
  );
}
