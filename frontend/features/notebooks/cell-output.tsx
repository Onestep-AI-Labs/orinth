"use client";

/* eslint-disable @next/next/no-img-element */

import { useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight, FileDown } from "lucide-react";
import { parseAnsi } from "@/features/notebooks/ansi";
import { Badge } from "@/features/platform/ui";
import type { ExecutionOutput } from "@/features/notebooks/kernel-client";

/**
 * Rendering what a cell produced.
 *
 * This is the work JupyterLab would have given away, and the honest cost of not
 * embedding it. It is bounded by an explicit allowlist, in dispatch order, and
 * anything outside that list renders a labelled row naming the MIME type with a
 * download action — never silently disappears. A user who cannot see their
 * output has no way to know whether the cell worked.
 *
 * `text/html` is deliberately **not** rendered as HTML. Sanitizing it properly
 * needs DOMPurify and a policy, and a half-sanitized `dangerouslySetInnerHTML`
 * fed by arbitrary kernel output is a script-injection hole in the studio. It
 * falls through to its `text/plain` alternative, which nbformat requires
 * producers to supply, and says so. Admitting HTML is a decision with a
 * threat model, not a rendering convenience.
 */

const MAX_TEXT = 200_000;

type Bundle = Record<string, unknown>;

function textOf(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.join("");
  return "";
}

function truncate(text: string): { text: string; truncated: boolean } {
  if (text.length <= MAX_TEXT) return { text, truncated: false };
  // A runaway loop printing megabytes must not take the tab down with it.
  return { text: text.slice(0, MAX_TEXT), truncated: true };
}

function AnsiText({ value, tone }: { value: string; tone: "stream" | "error" }) {
  const { text, truncated } = truncate(value);
  return (
    <pre className={`nb-output-term ${tone === "error" ? "nb-output-term-error" : ""}`}>
      {parseAnsi(text).map((span, index) => (
        <span key={index} className={span.className}>
          {span.text}
        </span>
      ))}
      {truncated && (
        <span className="nb-output-truncated">
          {"\n"}… output truncated at {MAX_TEXT.toLocaleString()} characters.
        </span>
      )}
    </pre>
  );
}

function JsonTree({ value }: { value: unknown }) {
  const [open, setOpen] = useState(false);
  const body = JSON.stringify(value, null, 2);
  const lines = body.split("\n");
  const collapsible = lines.length > 12;
  return (
    <div className="nb-output-json">
      {collapsible && (
        <button type="button" className="nb-output-toggle" onClick={() => setOpen((v) => !v)}>
          {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          {open ? "Collapse" : `Expand ${lines.length} lines`}
        </button>
      )}
      <pre className="nb-output-term">
        {collapsible && !open ? lines.slice(0, 12).join("\n") + "\n…" : body}
      </pre>
    </div>
  );
}

/** A run summary the SDK emits, rendered as metrics rather than as JSON. */
function RunCard({ value }: { value: Bundle }) {
  const metrics = (value.metrics as Record<string, number>) ?? {};
  return (
    <div className="nb-run-card">
      <div className="nb-run-card-head">
        <strong>{String(value.name ?? "run")}</strong>
        <Badge tone={value.status === "completed" ? "ok" : "info"}>
          {String(value.status ?? "running")}
        </Badge>
      </div>
      <dl className="nb-run-metrics">
        {Object.entries(metrics).map(([key, entry]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{typeof entry === "number" ? entry.toFixed(4) : String(entry)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function Unsupported({ mime, data }: { mime: string; data: unknown }) {
  const body = typeof data === "string" ? data : JSON.stringify(data);
  const href = `data:${mime};base64,${btoa(unescape(encodeURIComponent(body ?? "")))}`;
  return (
    <div className="nb-output-unsupported">
      <AlertTriangle size={14} aria-hidden="true" />
      <span>
        Orinth cannot display <code>{mime}</code> output.
      </span>
      <a className="nb-output-download" href={href} download={`output.${mime.split("/").pop()}`}>
        <FileDown size={13} /> Download
      </a>
    </div>
  );
}

/** The allowlist, in dispatch order. First match wins. */
function renderBundle(bundle: Bundle) {
  if (bundle["application/vnd.orinth.run+json"]) {
    return <RunCard value={bundle["application/vnd.orinth.run+json"] as Bundle} />;
  }
  for (const mime of ["image/png", "image/jpeg"] as const) {
    if (bundle[mime]) {
      return <img className="nb-output-image" src={`data:${mime};base64,${textOf(bundle[mime])}`} alt="" />;
    }
  }
  if (bundle["image/svg+xml"]) {
    // SVG can carry script, so it is shown as an image source rather than
    // inlined into the document — a data: URI in `src` cannot execute.
    const svg = encodeURIComponent(textOf(bundle["image/svg+xml"]));
    return <img className="nb-output-image" src={`data:image/svg+xml,${svg}`} alt="" />;
  }
  if (bundle["application/json"]) {
    return <JsonTree value={bundle["application/json"]} />;
  }
  if (bundle["text/plain"]) {
    return <AnsiText value={textOf(bundle["text/plain"])} tone="stream" />;
  }
  const [mime, data] = Object.entries(bundle)[0] ?? ["application/octet-stream", ""];
  return <Unsupported mime={mime} data={data} />;
}

export function CellOutput({ output }: { output: ExecutionOutput }) {
  const kind = output.output_type;

  if (kind === "stream") {
    return (
      <AnsiText
        value={textOf(output.text)}
        tone={output.name === "stderr" ? "error" : "stream"}
      />
    );
  }

  if (kind === "error") {
    const traceback = Array.isArray(output.traceback) ? output.traceback.join("\n") : "";
    return (
      <AnsiText
        value={traceback || `${output.ename}: ${output.evalue}`}
        tone="error"
      />
    );
  }

  if (kind === "execute_result" || kind === "display_data" || kind === "update_display_data") {
    const bundle = (output.data as Bundle) ?? {};
    const htmlOnly = bundle["text/html"] && !bundle["text/plain"] && !bundle["image/png"];
    return (
      <div className="nb-output-block">
        {renderBundle(bundle)}
        {htmlOnly ? (
          <p className="form-caption nb-output-note">
            This output is HTML, which Orinth does not render — see the notebook spec on why.
          </p>
        ) : null}
      </div>
    );
  }

  return null;
}
