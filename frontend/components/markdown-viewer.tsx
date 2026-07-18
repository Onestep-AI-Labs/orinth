"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dracula } from "react-syntax-highlighter/dist/cjs/styles/prism";
import { Check, Copy } from "lucide-react";
import { useState } from "react";

function CodeBlock({ node, inline, className, children, ...props }: any) {
  const match = /language-(\w+)/.exec(className || "");
  const [copied, setCopied] = useState(false);
  const codeText = String(children).replace(/\n$/, "");

  const handleCopy = () => {
    navigator.clipboard.writeText(codeText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!inline && match) {
    return (
      <div className="relative group not-prose my-6">
        <button
          onClick={handleCopy}
          className="absolute right-3 top-3 p-1.5 rounded-md bg-white/10 hover:bg-white/20 text-white/70 hover:text-white transition-all opacity-0 group-hover:opacity-100 z-10 border border-white/10 shadow-sm"
          title="Copy code"
        >
          {copied ? <Check size={16} className="text-teal" /> : <Copy size={16} />}
        </button>
        <SyntaxHighlighter
          {...props}
          style={dracula}
          language={match[1]}
          PreTag="div"
          className="rounded-lg overflow-hidden text-sm !m-0 !bg-navy-deep"
          showLineNumbers={true}
        >
          {codeText}
        </SyntaxHighlighter>
      </div>
    );
  }
  return (
    <code className={`${className} bg-black/5 text-pink-600 rounded-md px-1.5 py-0.5`} {...props}>
      {children}
    </code>
  );
}

export function MarkdownViewer({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code: CodeBlock as any,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
