"use client";

import { Save } from "lucide-react";
import { DatasetPreprocessPanel } from "@/features/datasets/preprocess-panel";
import type {
  useCreateDatasetVersionMutation,
  useDatasetCatalogQuery,
  useDatasetItemDetailQuery,
  useDatasetVersionsQuery,
  usePreviewDatasetPreprocessMutation,
  useProcessDatasetMutation
} from "@/features/datasets/hooks";
import { DatasetSplitVersionPanel } from "@/features/datasets/split-version-panel";
import { InlineSpinner, MutationError, PanelTitle, useConfirmationDialog } from "@/features/platform/ui";
import type { DatasetPreprocessConfig, DatasetSplitConfig, DatasetSummary } from "@/types/api";

/**
 * Config tab: preprocessing + split/versioning panels. State and
 * mutations stay owned by DatasetPage; this is a thin composition of
 * the already-extracted preprocess and split/version panels.
 */
export function DatasetConfigTab({
  dataset,
  nlp,
  catalogQuery,
  preprocessConfig,
  setPreprocessConfig,
  detailQuery,
  previewUrl,
  previewText,
  previewMutation,
  splitConfig,
  setSplitConfig,
  versionName,
  setVersionName,
  versionsQuery,
  processMutation,
  createVersionMutation,
  confirm
}: {
  dataset: DatasetSummary;
  nlp: boolean;
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
    <section className="panel">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <PanelTitle icon={<Save size={18} />} title="Config" />
        {catalogQuery.isFetching && <InlineSpinner label="Refreshing" />}
      </div>
      <div className="config-panel-stack">
        <DatasetPreprocessPanel
          dataset={dataset}
          config={preprocessConfig}
          setConfig={setPreprocessConfig}
          nlp={nlp}
          detailItem={detailQuery.data}
          previewUrl={previewUrl}
          previewText={previewText}
          previewMutation={previewMutation}
        />
        <DatasetSplitVersionPanel
          dataset={dataset}
          splitConfig={splitConfig}
          setSplitConfig={setSplitConfig}
          preprocessConfig={preprocessConfig}
          versionName={versionName}
          setVersionName={setVersionName}
          versions={versionsQuery.data ?? []}
          versionsLoading={versionsQuery.isLoading}
          processMutation={processMutation}
          createVersionMutation={createVersionMutation}
          confirm={confirm}
        />
      </div>
      <MutationError mutations={[previewMutation, createVersionMutation, processMutation]} />
    </section>
  );
}
