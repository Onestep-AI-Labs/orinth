"use client";

import { useState } from "react";
import { ClipboardCopy, Database } from "lucide-react";
import { useDatasetCatalogQuery } from "@/features/datasets/hooks";
import { ReadinessBadge } from "@/features/datasets/readiness-badge";
import { toast } from "@/features/platform/toast";
import { EmptyState, InlineSpinner, PanelTitle } from "@/features/platform/ui";
import { formatDatasetTask } from "@/features/platform/utils";

/**
 * The datasets in this project, with the line of Python that loads one.
 *
 * The problem this solves is small and constant: a notebook needs a dataset id,
 * and the id is a slug with a hex suffix that nobody types correctly from
 * memory. Copying `orinth.datasets.load("…")` rather than the bare id is the
 * difference between one paste and one paste plus remembering the call.
 *
 * Readiness rides along because it is the thing worth knowing before loading:
 * a dataset that is not trainable will still `load()`, and finding that out
 * from a badge beats finding it out from a confusing frame.
 */
export function DatasetRail({ projectId }: { projectId: string }) {
  const catalogQuery = useDatasetCatalogQuery(projectId);
  const [copied, setCopied] = useState("");
  const datasets = catalogQuery.data ?? [];

  async function copy(datasetId: string) {
    const snippet = `orinth.datasets.load("${datasetId}", "train")`;
    try {
      await navigator.clipboard.writeText(snippet);
      setCopied(datasetId);
      // Reset so the check does not stick on a row the user has moved past.
      setTimeout(() => setCopied(""), 1500);
    } catch {
      // A denied clipboard permission is not worth an error banner; the id is
      // on screen and selectable either way.
      toast.error("Could not reach the clipboard. Select the id instead.");
    }
  }

  return (
    <section className="panel">
      <div className="mb-3 flex items-center justify-between gap-2">
        <PanelTitle icon={<Database size={16} />} title="Datasets" />
        {catalogQuery.isFetching && <InlineSpinner label="Loading" />}
      </div>
      {datasets.length === 0 ? (
        <EmptyState
          icon={<Database size={22} />}
          label="No datasets here yet."
          description="Upload one on the Datasets page and it appears in this list."
        />
      ) : (
        <ul className="nb-dataset-list">
          {datasets.map((dataset) => (
            <li key={dataset.id}>
              <div className="nb-dataset-row">
                <strong title={dataset.name}>{dataset.name}</strong>
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => copy(dataset.id)}
                  title={`Copy orinth.datasets.load("${dataset.id}", "train")`}
                >
                  <ClipboardCopy size={13} />
                </button>
              </div>
              <div className="nb-dataset-meta">
                <ReadinessBadge readiness={dataset.readiness} />
                <span>{formatDatasetTask(dataset.task_type)}</span>
              </div>
              <code className="nb-dataset-id">
                {copied === dataset.id ? "copied to clipboard" : dataset.id}
              </code>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
