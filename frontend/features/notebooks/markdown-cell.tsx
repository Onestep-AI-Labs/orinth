"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Pencil } from "lucide-react";

/**
 * A prose cell, rendered until someone double-clicks it.
 *
 * Rendered-by-default is the whole point: templates are the SDK's
 * documentation, and a template that opens as raw `#` headings teaches nothing.
 * Double-click to edit is the convention every notebook uses, so it needs no
 * affordance of its own — the pencil is there for discoverability and for
 * keyboard users, who reach it by tab.
 *
 * Reuses the phase-15 `.chat-md` GFM renderer rather than styling markdown
 * twice. The two surfaces show the same kind of content and there is no reason
 * a table in a notebook should look different from a table in the chat.
 */

export function MarkdownCell({
  source,
  editing,
  onEdit
}: {
  source: string;
  editing: boolean;
  onEdit: () => void;
}) {
  if (editing) return null;
  return (
    <div
      className="nb-markdown chat-md"
      onDoubleClick={onEdit}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter") onEdit();
      }}
      title="Double-click to edit"
    >
      {source.trim() ? (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{source}</ReactMarkdown>
      ) : (
        <p className="form-caption">Empty markdown cell — double-click to write something.</p>
      )}
      <span className="nb-markdown-edit" aria-hidden="true">
        <Pencil size={12} />
      </span>
    </div>
  );
}
