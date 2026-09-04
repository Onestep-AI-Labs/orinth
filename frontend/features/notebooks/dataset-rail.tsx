"use client";

import { useState } from "react";
import { ClipboardCopy, Database } from "lucide-react";
import { useDatasetCatalogQuery } from "@/features/datasets/hooks";
import { ReadinessBadge } from "@/features/datasets/readiness-badge";
import { toast } from "@/features/platform/toast";
import { EmptyState, InlineSpinner, ListSkeleton, PanelTitle } from "@/features/platform/ui";
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
 *
 * The catalog costs seconds to build — it walks every split of every dataset —
 * so this rail asks for a long stale window. The first cut used the page
 * default and re-fetched on every visit, which meant the panel spent those
 * seconds showing a spinner (or, worse, "No datasets here yet") for a list that
 * had not changed since the last time it was looked at.
 */

//: Five minutes. Any mutation that changes the catalog invalidates the key, so
//: this only governs how often an *unchanged* list is rebuilt from scratch.
const RAIL_STALE_TIME = 5 * 60_000;

export function DatasetRail({ projectId }: { projectId: string }) {
  const catalogQuery = useDatasetCatalogQuery(projectId, {
    staleTime: RAIL_STALE_TIME,
    // No polling. The rail lists ids to copy; a prep run's progress belongs to
    // the Datasets page, and watching one from here re-read the whole catalog
    // every four seconds for as long as *any* dataset was mid-prep — including
    // one left mid-prep by a backend restart weeks ago.
    poll: false
  });
  const [copied, setCopied] = useState("");
  const datasets = catalogQuery.data ?? [];
  //: The contract: nothing yet is a skeleton, a refresh over existing rows is
  //: the inline spinner. Showing the empty state while the first request is
  //: still out told the user there were no datasets when nobody had looked.
  const loading = catalogQuery.isPending || (catalogQuery.isFetching && datasets.length === 0);

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
        {catalogQuery.isFetching && datasets.length > 0 && <InlineSpinner label="Refreshing" />}
      </div>
      {loading ? (
        <ListSkeleton rows={3} />
      ) : datasets.length === 0 ? (
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
