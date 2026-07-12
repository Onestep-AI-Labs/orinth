"use client";

import { UploadDropCard as DatasetUploadDropCard } from "@/features/datasets/upload-drop-card";
import type { useUploadDatasetMutation } from "@/features/datasets/hooks";
import type { DatasetSplitFilter, DatasetSummary } from "@/types/api";

/**
 * Upload flow for the dataset "images"/"texts" tab: file selection state
 * lives with the caller (the mutation's onSuccess resets it, and it is
 * also read by the shared MutationError banner further down the page),
 * so this component only owns the upload JSX + the submit handler.
 */
export function DatasetUploadPanel({
  dataset,
  split,
  files,
  setFiles,
  uploadClassId,
  setUploadClassId,
  errors,
  uploadMutation
}: {
  dataset: DatasetSummary;
  split: DatasetSplitFilter;
  files: File[];
  setFiles: (files: File[]) => void;
  uploadClassId: number;
  setUploadClassId: (value: number) => void;
  errors: Array<Record<string, string>>;
  uploadMutation: ReturnType<typeof useUploadDatasetMutation>;
}) {
  if (!dataset.editable) return null;

  function uploadImages() {
    if (!dataset.editable || files.length === 0) return;
    const form = new FormData();
    form.append("split", split === "all" ? "unassigned" : split);
    files.forEach((file) => form.append("files", file));
    if (dataset.task_type === "classification" || dataset.task_type === "text_classification") {
      form.append("class_id", String(uploadClassId));
    }
    uploadMutation.mutate({ datasetId: dataset.id, form });
  }

  return (
    <DatasetUploadDropCard
      files={files}
      setFiles={setFiles}
      dataset={dataset}
      uploadClassId={uploadClassId}
      setUploadClassId={setUploadClassId}
      onUpload={uploadImages}
      pending={uploadMutation.isPending}
      errors={errors}
    />
  );
}
