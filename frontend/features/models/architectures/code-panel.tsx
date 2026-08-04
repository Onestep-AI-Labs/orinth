"use client";

import { useState } from "react";
import { PrismAsync as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/cjs/styles/prism";
import { Check, Copy, Download } from "lucide-react";
import { Button, ButtonLink, InlineSpinner } from "@/features/platform/ui";
import type { Framework } from "@/lib/api/architectures";

/**
 * The generated module, in either framework.
 *
 * Read-only on purpose: the graph is the source of truth and the code is
 * regenerated from it on every export and training run. Editing here would
 * create a second definition that silently diverges from what trains, so
 * Download is how you take ownership of the file.
 *
 * TensorFlow is what trains inside the platform; PyTorch is an export for
 * people whose stack is torch. Both render from the same resolved graph, so
 * they cannot describe different models.
 */
export function CodePanel({
  code,
  filename,
  downloadUrl,
  loading,
  error,
  framework,
  onFrameworkChange
}: {
  code: string;
  filename: string;
  downloadUrl: string;
  loading: boolean;
  error: string | null;
  framework: Framework;
  onFrameworkChange: (framework: Framework) => void;
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className="arch-code">
      <div className="arch-code-head">
        <div className="arch-code-identity">
          <div className="segmented-control arch-framework-toggle">
            {(["keras", "torch"] as const).map((option) => (
              <button
                key={option}
                type="button"
                className={framework === option ? "segmented-active" : ""}
                onClick={() => onFrameworkChange(option)}
              >
                {option === "keras" ? "TensorFlow" : "PyTorch"}
              </button>
            ))}
          </div>
          <p className="arch-code-name">{filename}</p>
        </div>
        <div className="arch-code-actions">
          {loading && <InlineSpinner label="Generating" />}
          <Button variant="ghost" size="sm" onClick={copy} disabled={!code}>
            {copied ? <Check size={15} aria-hidden /> : <Copy size={15} aria-hidden />}
            {copied ? "Copied" : "Copy"}
          </Button>
          <ButtonLink variant="secondary" size="sm" href={downloadUrl}>
            <Download size={15} aria-hidden /> Download
          </ButtonLink>
        </div>
      </div>
      {error ? (
        <p className="arch-code-error">{error}</p>
      ) : (
        <div className="arch-code-body">
          <SyntaxHighlighter
            language="python"
            style={oneLight}
            customStyle={{
              margin: 0,
              background: "transparent",
              fontSize: "0.8125rem",
              padding: "16px"
            }}
          >
            {code || "# Fix the issues below to generate code."}
          </SyntaxHighlighter>
        </div>
      )}
    </div>
  );
}
