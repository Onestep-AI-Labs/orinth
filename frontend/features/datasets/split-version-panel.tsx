"use client";

import { SplitConfigPanel, VersionPanel } from "@/features/datasets/dataset-components";
import type { useCreateDatasetVersionMutation, useProcessDatasetMutation } from "@/features/datasets/hooks";
import type { useConfirmationDialog } from "@/features/platform/ui";
import type { DatasetPreprocessConfig, DatasetSplitConfig, DatasetSummary, DatasetVersionSummary } from "@/types/api";

/**
 * Split configuration + version-artifact panel for the dataset "Config"
 * tab. The underlying process/create-version mutations (and the
 * confirmation dialog they share with the rest of the page) are owned
 * by DatasetPage so cache-invalidation sequencing and the combined
 * MutationError banner stay identical to the pre-split behavior.
 */
export function DatasetSplitVersionPanel({
  dataset,
  splitConfig,
  setSplitConfig,
  preprocessConfig,
  versionName,
  setVersionName,
  versions,
  versionsLoading,
  processMutation,
  createVersionMutation,
  confirm
}: {
  dataset: DatasetSummary;
  splitConfig: DatasetSplitConfig;
  setSplitConfig: (value: DatasetSplitConfig) => void;
  preprocessConfig: DatasetPreprocessConfig;
  versionName: string;
  setVersionName: (value: string) => void;
  versions: DatasetVersionSummary[];
  versionsLoading: boolean;
  processMutation: ReturnType<typeof useProcessDatasetMutation>;
  createVersionMutation: ReturnType<typeof useCreateDatasetVersionMutation>;
  confirm: ReturnType<typeof useConfirmationDialog>["confirm"];
}) {
  function confirmProcessDataset() {
    confirm({
      title: "Proceed with dataset processing?",
      message: `This saves preprocessing settings and distributes inbox items in "${dataset.name}" into train, valid, and test.`,
      confirmLabel: "Proceed",
      tone: "warning",
      onConfirm: () => processMutation.mutate({ datasetId: dataset.id, preprocess: preprocessConfig, split: splitConfig })
    });
  }

  function createVersion() {
    createVersionMutation.mutate({
      datasetId: dataset.id,
      name: versionName.trim() || undefined,
      config: preprocessConfig
    });
  }

  return (
    <>
      <SplitConfigPanel
        config={splitConfig}
        setConfig={setSplitConfig}
        editable={dataset.editable}
        dataset={dataset}
        onProceed={confirmProcessDataset}
        pending={processMutation.isPending}
      />
      <VersionPanel
        versions={versions}
        loading={versionsLoading}
        versionName={versionName}
        setVersionName={setVersionName}
        config={preprocessConfig}
        onCreate={createVersion}
        pending={createVersionMutation.isPending}
        editable={dataset.editable}
      />
    </>
  );
}
