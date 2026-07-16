"use client";

import { PreprocessPanel } from "@/features/datasets/dataset-components";
import type { usePreviewDatasetPreprocessMutation } from "@/features/datasets/hooks";
import type { DatasetItemDetail, DatasetPreprocessConfig, DatasetSummary } from "@/types/api";

/**
 * Preprocessing panel for the dataset "Config" tab. The preview mutation
 * (and the previewUrl/previewText state it populates via onSuccess) stays
 * owned by DatasetPage since previewMutation also feeds the combined
 * MutationError banner shared with the split/version panel.
 */
export function DatasetPreprocessPanel({
  dataset,
  config,
  setConfig,
  nlp,
  detailItem,
  previewUrl,
  previewText,
  previewMutation
}: {
  dataset: DatasetSummary;
  config: DatasetPreprocessConfig;
  setConfig: (value: DatasetPreprocessConfig) => void;
  nlp: boolean;
  detailItem: DatasetItemDetail | undefined;
  previewUrl: string | null;
  previewText: string | null;
  previewMutation: ReturnType<typeof usePreviewDatasetPreprocessMutation>;
}) {
  function previewPreprocess() {
    if (!detailItem) return;
    previewMutation.mutate({ datasetId: dataset.id, item: detailItem });
  }

  return (
    <PreprocessPanel
      config={config}
      setConfig={setConfig}
      editable={dataset.editable}
      onPreview={previewPreprocess}
      previewUrl={previewUrl}
      previewText={previewText}
      previewPending={previewMutation.isPending}
      nlp={nlp}
    />
  );
}
