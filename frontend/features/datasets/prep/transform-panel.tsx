"use client";

import { useState } from "react";
import { ChevronDown, Code2 } from "lucide-react";
import { Badge, Button, PanelTitle } from "@/features/platform/ui";
import type { PrepTransform } from "@/types/api";

/**
 * The Python that reshaped the upload, shown because it ran.
 *
 * A user who drops a spreadsheet in and is told it is now a text classifier is
 * owed the twenty lines that made that true. The rest of the Overview tab
 * attributes *decisions* — task, labels, split — and this attributes the one
 * step that rewrote the data itself, which is the only part of the agent that
 * cannot be checked by looking at the result.
 *
 * The code is collapsed by default and expanded in one click. Open by default
 * would put fifty lines of Python between the reader and the readiness checks
 * they came for; hidden behind a route would mean nobody ever reads it.
 */

const ENGINE_LABEL: Record<PrepTransform["engine"], string> = {
  builtin: "column statistics",
  llm: "model"
};

export function PrepTransformPanel({ transform }: { transform: PrepTransform }) {
  const [showCode, setShowCode] = useState(false);
  const kept = transform.input_rows
    ? Math.round((transform.output_rows / transform.input_rows) * 100)
    : 0;

  return (
    <section className="panel">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <PanelTitle icon={<Code2 size={18} />} title="How the rows were reshaped" />
        <Badge tone={transform.engine === "llm" ? "info" : "neutral"}>
          {transform.engine === "llm"
            ? `model · ${transform.model ?? "openrouter"}`
            : ENGINE_LABEL.builtin}
        </Badge>
      </div>

      <p className="form-caption prep-summary">
        Orinth could not map these columns to a task on its own, so it wrote a Python transform
        from {ENGINE_LABEL[transform.engine]} and ran it in a sandbox with no network and no
        filesystem. The upload itself was not modified.
      </p>

      {transform.notice && <p className="prep-notice">{transform.notice}</p>}

      <dl className="prep-transform-meta">
        <div>
          <dt>Rows</dt>
          <dd>
            {transform.input_rows.toLocaleString()} → {transform.output_rows.toLocaleString()}
            {kept > 0 && <span className="prep-transform-share"> ({kept}% kept)</span>}
          </dd>
        </div>
        {transform.target_column && (
          <div>
            <dt>Predicting</dt>
            <dd className="prep-transform-column">{transform.target_column}</dd>
          </div>
        )}
        {transform.feature_columns.length > 0 && (
          <div>
            <dt>Features</dt>
            <dd>{transform.feature_columns.length} columns</dd>
          </div>
        )}
        {transform.source_files.length > 0 && (
          <div>
            <dt>Read from</dt>
            <dd className="prep-transform-column">{transform.source_files.join(", ")}</dd>
          </div>
        )}
      </dl>

      {transform.rationale && <p className="form-caption prep-summary">{transform.rationale}</p>}

      <div className="accordion-panel advanced-panel">
        <div className="accordion-header">
          <button
            className="accordion-title"
            type="button"
            onClick={() => setShowCode((value) => !value)}
            aria-expanded={showCode}
          >
            <ChevronDown
              className={`accordion-icon ${showCode ? "" : "accordion-icon-collapsed"}`}
              size={17}
            />
            <span>The transform that ran</span>
          </button>
          {showCode && (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void navigator.clipboard?.writeText(transform.code)}
            >
              Copy
            </Button>
          )}
        </div>
        {showCode && (
          <div className="accordion-body">
            <pre className="prep-transform-code">{transform.code}</pre>
          </div>
        )}
      </div>
    </section>
  );
}
