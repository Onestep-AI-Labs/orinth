"use client";

import { BarChart3 } from "lucide-react";
import { DatasetConfigTab } from "@/features/datasets/config-tab";
import { EdaPanel } from "@/features/datasets/dataset-components";
import type {
  useCreateDatasetVersionMutation,
  useDatasetCatalogQuery,
  useDatasetEdaQuery,
  useDatasetItemDetailQuery,
  useDatasetVersionsQuery,
  usePreviewDatasetPreprocessMutation,
  useProcessDatasetMutation
} from "@/features/datasets/hooks";
import { PanelTitle } from "@/features/platform/ui";
import type { useConfirmationDialog } from "@/features/platform/ui";
import type {
  DatasetPreprocessConfig,
  DatasetSplitConfig,
  DatasetSplitFilter,
  DatasetSummary
} from "@/types/api";

/**
 * Prepare tab: what the data looks like, then how it is prepared.
 *
 * Phase 21 merged the old EDA and Config tabs. They were always one question
 * asked twice — "is this data balanced" and "how should it be split" — and
 * splitting them across two tabs meant the answer sat on a different screen from
 * the control it should inform.
 *
 * Both panels are the existing components, moved rather than rewritten: the
 * agent fills these values in now, and this is where they are inspected and
 * overridden.
 */
export function DatasetPrepareTab({
  dataset,
  nlp,
  split,
  setSplit,
  edaQuery,
  ...configProps
}: {
  dataset: DatasetSummary;
  nlp: boolean;
  split: DatasetSplitFilter;
  setSplit: (value: DatasetSplitFilter) => void;
  edaQuery: ReturnType<typeof useDatasetEdaQuery>;
  catalogQuery: ReturnType<typeof useDatasetCatalogQuery>;
  preprocessConfig: DatasetPreprocessConfig;
  setPreprocessConfig: (value: DatasetPreprocessConfig) => void;
  detailQuery: ReturnType<typeof useDatasetItemDetailQuery>;
  previewUrl: string | null;
  previewText: string | null;
  previewMutation: ReturnType<typeof usePreviewDatasetPreprocessMutation>;
  splitConfig: DatasetSplitConfig;
  setSplitConfig: (value: DatasetSplitConfig) => void;
  versionName: string;
  setVersionName: (value: string) => void;
  versionsQuery: ReturnType<typeof useDatasetVersionsQuery>;
  processMutation: ReturnType<typeof useProcessDatasetMutation>;
  createVersionMutation: ReturnType<typeof useCreateDatasetVersionMutation>;
  confirm: ReturnType<typeof useConfirmationDialog>["confirm"];
}) {
  return (
    <div className="space-y-5">
      <section className="panel">
        <PanelTitle icon={<BarChart3 size={18} />} title="What is in this data" />
        <EdaPanel
          eda={edaQuery.data}
          loading={edaQuery.isLoading}
          dataset={dataset}
          split={split}
          setSplit={setSplit}
        />
      </section>
      <DatasetConfigTab dataset={dataset} nlp={nlp} {...configProps} />
    </div>
  );
}
