"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
// PrismAsync code-splits language grammars out of the initial chat bundle but
// still auto-loads them on demand — so code is actually highlighted (the
// *Light* variant needs manual registerLanguage and renders colorless without
// it, which is why highlighting looked dead before). `oneLight` is a light
// theme so tokens read on the light code surface; both the outer and inner
// theme backgrounds are stripped below so the block shows no dark highlight.
import { PrismAsync as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/cjs/styles/prism";
import { Brain, Check, ChevronRight, Copy } from "lucide-react";

/**
 * Rich assistant/user message rendering for the chat surface: GitHub-flavored
 * markdown, syntax-highlighted code with a copy button, and collapsible
 * "thinking" (reasoning) blocks parsed from `<think>…</think>` tags — the
 * chat-LLM conventions users expect (à la OpenWebUI), expressed in the design
 * tokens.
 */
export function ChatMessageContent({
  content,
  streaming,
  showThinking = true
}: {
  content: string;
  streaming?: boolean;
  showThinking?: boolean;
}) {
  const { thinking, thinkingOpen, answer } = splitThinking(content);
  if (!content) return null;
  return (
    <>
      {showThinking && thinking !== null && (
        <ThinkingBlock text={thinking} active={Boolean(thinkingOpen && streaming)} />
      )}
      {answer.trim().length > 0 && <Markdown text={answer} />}
    </>
  );
}

function Markdown({ text }: { text: string }) {
  return (
    <div className="chat-md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code: CodeBlock as never,
          // react-markdown wraps block code in <pre>; the CodeBlock renders its
          // own container, so collapse the default <pre> to a pass-through.
          pre: ({ children }) => <>{children}</>
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

function CodeBlock({ className, children }: { className?: string; children?: React.ReactNode }) {
  const raw = String(children ?? "").replace(/\n$/, "");
  const match = /language-(\w+)/.exec(className || "");
  const isBlock = Boolean(match) || raw.includes("\n");
  if (!isBlock) {
    return <code className="chat-inline-code">{children}</code>;
  }
  return (
    <div className="chat-code-block">
      <div className="chat-code-head">
        <span className="chat-code-lang">{match?.[1] ?? "text"}</span>
        <CopyButton value={raw} label="Copy code" />
      </div>
      <SyntaxHighlighter
        style={oneLight}
        language={match?.[1] ?? "text"}
        PreTag="div"
        // Strip the theme's own backgrounds (outer pre + inner code) so the
        // code renders as colored text on the block surface — no dark line
        // highlight.
        customStyle={{ margin: 0, borderRadius: 0, background: "transparent", fontSize: 13 }}
        codeTagProps={{ style: { background: "transparent" } }}
      >
        {raw}
      </SyntaxHighlighter>
    </div>
  );
}

function ThinkingBlock({ text, active }: { text: string; active: boolean }) {
  const [open, setOpen] = useState(false);
  const expanded = active || open;
  return (
    <div className={`chat-think${active ? " chat-think-active" : ""}`}>
      <button
        type="button"
        className="chat-think-toggle"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={expanded}
      >
        <ChevronRight size={14} className={`chat-think-chevron${expanded ? " chat-think-chevron-open" : ""}`} />
        <Brain size={14} />
        <span>{active ? "Thinking…" : "Thought process"}</span>
      </button>
      {expanded && (
        <div className="chat-think-body">
          <Markdown text={text.trim()} />
        </div>
      )}
    </div>
  );
}

export function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="chat-copy-btn"
      title={label}
      aria-label={label}
      onClick={() => {
        navigator.clipboard?.writeText(value);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

/**
 * Split a `<think>…</think>` (or `<thinking>`) reasoning block out of the raw
 * text. During streaming the closing tag may not have arrived yet, in which
 * case everything after the opening tag is the in-progress thought and there is
 * no answer yet.
 */
function splitThinking(content: string): {
  thinking: string | null;
  thinkingOpen: boolean;
  answer: string;
} {
  const open = content.match(/<think(?:ing)?>/i);
  if (!open || open.index === undefined) {
    return { thinking: null, thinkingOpen: false, answer: content };
  }
  const before = content.slice(0, open.index);
  const rest = content.slice(open.index + open[0].length);
  const close = rest.match(/<\/think(?:ing)?>/i);
  if (!close || close.index === undefined) {
    return { thinking: rest, thinkingOpen: true, answer: before };
  }
  const thinking = rest.slice(0, close.index);
  const after = rest.slice(close.index + close[0].length);
  return { thinking, thinkingOpen: false, answer: `${before}${after}` };
}
