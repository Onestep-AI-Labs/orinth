"use client";

import { useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Bold,
  Code,
  Heading,
  Image as ImageIcon,
  Italic,
  Link2,
  List,
  ListOrdered,
  Minus,
  Pencil,
  Quote,
  Sigma,
  Table
} from "lucide-react";

/**
 * A prose cell: rendered until someone edits it, and edited with a toolbar.
 *
 * Rendered-by-default is the whole point: templates are the SDK's
 * documentation, and a template that opens as raw `#` headings teaches nothing.
 * Double-click to edit is the convention every notebook uses.
 *
 * Editing shows the source beside a live preview, with a formatting toolbar
 * above — the shape every notebook editor has converged on, and the reason is
 * that markdown is a *language* people half-know. Someone who cannot remember
 * whether a link is `[]()` or `()[]` should not have to leave to find out, and
 * seeing the render update as you type is what makes a table fixable.
 *
 * Reuses the phase-15 `.chat-md` GFM renderer rather than styling markdown
 * twice. There is no reason a table in a notebook should look different from a
 * table in the chat.
 */

type Action = {
  key: string;
  label: string;
  icon: React.ReactNode;
  /** Wraps the selection. */
  wrap?: [string, string];
  /** Prefixes each selected line. */
  prefix?: string;
  /** Inserted whole when there is no selection to act on. */
  snippet?: string;
};

//: Ordered by how often it is reached for, not by markdown's grammar.
const ACTIONS: Action[] = [
  { key: "heading", label: "Heading", icon: <Heading size={15} />, prefix: "## " },
  { key: "bold", label: "Bold", icon: <Bold size={15} />, wrap: ["**", "**"] },
  { key: "italic", label: "Italic", icon: <Italic size={15} />, wrap: ["_", "_"] },
  { key: "code", label: "Code", icon: <Code size={15} />, wrap: ["`", "`"] },
  { key: "link", label: "Link", icon: <Link2 size={15} />, wrap: ["[", "](https://)"] },
  { key: "image", label: "Image", icon: <ImageIcon size={15} />, snippet: "![alt](https://)" },
  { key: "quote", label: "Quote", icon: <Quote size={15} />, prefix: "> " },
  { key: "ordered", label: "Numbered list", icon: <ListOrdered size={15} />, prefix: "1. " },
  { key: "bullet", label: "Bulleted list", icon: <List size={15} />, prefix: "- " },
  { key: "rule", label: "Horizontal rule", icon: <Minus size={15} />, snippet: "\n---\n" },
  { key: "math", label: "LaTeX", icon: <Sigma size={15} />, wrap: ["$", "$"] },
  {
    key: "table",
    label: "Table",
    icon: <Table size={15} />,
    snippet: "\n| Column | Column |\n| --- | --- |\n|  |  |\n"
  }
];

function applyAction(action: Action, value: string, start: number, end: number) {
  const selected = value.slice(start, end);

  if (action.prefix) {
    // Line-oriented: a prefix applies to every line the selection touches, and
    // to the caret's own line when nothing is selected.
    const lineStart = value.lastIndexOf("\n", start - 1) + 1;
    const lineEnd = end + (value.slice(end).indexOf("\n") + 1 || value.length - end);
    const block = value.slice(lineStart, lineEnd);
    const prefixed = block
      .split("\n")
      .map((line, index) =>
        line.startsWith(action.prefix!)
          ? line.slice(action.prefix!.length)
          : `${action.prefix}${line}`
      )
      .join("\n");
    return {
      next: value.slice(0, lineStart) + prefixed + value.slice(lineEnd),
      caret: lineStart + prefixed.length
    };
  }

  if (action.wrap) {
    const [open, close] = action.wrap;
    const body = selected || action.key === "link" ? selected || "text" : "text";
    const inserted = `${open}${body}${close}`;
    return {
      next: value.slice(0, start) + inserted + value.slice(end),
      // Selects the body so typing replaces the placeholder rather than
      // appending to it.
      caret: start + open.length + body.length
    };
  }

  const snippet = action.snippet ?? "";
  return { next: value.slice(0, start) + snippet + value.slice(end), caret: start + snippet.length };
}

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
        <p className="form-caption">Empty text cell — double-click to write something.</p>
      )}
      <span className="nb-markdown-edit" aria-hidden="true">
        <Pencil size={12} />
      </span>
    </div>
  );
}

export function MarkdownEditor({
  value,
  onChange,
  onDone
}: {
  value: string;
  onChange: (value: string) => void;
  /** Rendering the cell — the same thing Shift-Enter does. */
  onDone: () => void;
}) {
  const field = useRef<HTMLTextAreaElement>(null);

  function run(action: Action) {
    const node = field.current;
    if (!node) return;
    const { next, caret } = applyAction(action, value, node.selectionStart, node.selectionEnd);
    onChange(next);
    // After React has written the new value, or the caret lands in the old one.
    requestAnimationFrame(() => {
      node.focus();
      node.selectionStart = node.selectionEnd = caret;
    });
  }

  return (
    <div className="nb-md-editor">
      <div className="nb-md-toolbar" role="toolbar" aria-label="Text formatting">
        {ACTIONS.map((action) => (
          <button
            key={action.key}
            type="button"
            className="icon-button"
            title={action.label}
            aria-label={action.label}
            // The toolbar must not steal focus, or the selection it is about to
            // act on is gone by the time the click lands.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => run(action)}
          >
            {action.icon}
          </button>
        ))}
        <button type="button" className="nb-md-close" onClick={onDone}>
          Close
        </button>
      </div>
      <div className="nb-md-split">
        <textarea
          ref={field}
          className="nb-md-source"
          value={value}
          aria-label="Text cell source"
          spellCheck
          placeholder="Write markdown…"
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.shiftKey || event.metaKey || event.ctrlKey)) {
              event.preventDefault();
              onDone();
            }
          }}
        />
        <div className="nb-md-preview chat-md">
          {value.trim() ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{value}</ReactMarkdown>
          ) : (
            <p className="form-caption">The render appears here as you type.</p>
          )}
        </div>
      </div>
    </div>
  );
}
