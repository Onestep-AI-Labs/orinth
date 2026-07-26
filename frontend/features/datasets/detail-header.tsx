"use client";

import type { Dispatch, SetStateAction } from "react";
import { Copy, Database, MoreVertical, Save, Trash2 } from "lucide-react";
import type { useCloneDatasetMutation, useDeleteDatasetMutation, useUpdateDatasetMutation } from "@/features/datasets/hooks";
import { Field, PageHeader } from "@/features/platform/ui";
import type { DatasetPreprocessConfig, DatasetSummary } from "@/types/api";

/**
 * Dataset detail header: title, back-to-catalog, options menu, and the
 * inline rename/duplicate dialogs. State (open/closed dialogs, draft
 * text) and mutations stay owned by DatasetPage.
 */
export function DatasetDetailHeader({
  dataset,
  onBackToCatalog,
  showDatasetOptions,
  setShowDatasetOptions,
  showRename,
  setShowRename,
  showDuplicate,
  setShowDuplicate,
  editName,
  setEditName,
  cloneName,
  setCloneName,
  preprocessConfig,
  updateDatasetMutation,
  cloneMutation,
  deleteDatasetMutation,
  onDeleteDataset
}: {
  dataset: DatasetSummary;
  onBackToCatalog: () => void;
  showDatasetOptions: boolean;
  setShowDatasetOptions: Dispatch<SetStateAction<boolean>>;
  showRename: boolean;
  setShowRename: (value: boolean) => void;
  showDuplicate: boolean;
  setShowDuplicate: (value: boolean) => void;
  editName: string;
  setEditName: (value: string) => void;
  cloneName: string;
  setCloneName: (value: string) => void;
  preprocessConfig: DatasetPreprocessConfig;
  updateDatasetMutation: ReturnType<typeof useUpdateDatasetMutation>;
  cloneMutation: ReturnType<typeof useCloneDatasetMutation>;
  deleteDatasetMutation: ReturnType<typeof useDeleteDatasetMutation>;
  onDeleteDataset: (dataset: DatasetSummary) => void;
}) {
  function saveDatasetSettings() {
    if (!dataset.editable) return;
    updateDatasetMutation.mutate({
      datasetId: dataset.id,
      name: editName.trim() || dataset.name,
      preprocess: preprocessConfig
    });
  }

  return (
    <>
      <div className="dataset-detail-header" data-tour="dataset-studio-header">
        <PageHeader title={dataset.name} subtitle="Dataset workspace" icon={<Database size={20} />} />
        <div className="dataset-header-actions">
          <button className="secondary-button" onClick={onBackToCatalog}>
            Back to catalog
          </button>
          <div className="dataset-options">
            <button
              className="icon-button"
              onClick={() => setShowDatasetOptions((value) => !value)}
              title="Dataset options"
              type="button"
              aria-expanded={showDatasetOptions}
            >
              <MoreVertical size={16} />
            </button>
            {showDatasetOptions && (
              <div className="option-menu" role="menu">
                <button
                  type="button"
                  onClick={() => {
                    setShowRename(true);
                    setShowDuplicate(false);
                    setShowDatasetOptions(false);
                  }}
                >
                  <Save size={15} /> Rename
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowDuplicate(true);
                    setShowRename(false);
                    setShowDatasetOptions(false);
                  }}
                >
                  <Copy size={15} /> Duplicate
                </button>
                <button
                  className="danger-menu-item"
                  type="button"
                  onClick={() => onDeleteDataset(dataset)}
                  disabled={!dataset.editable || deleteDatasetMutation.isPending}
                >
                  <Trash2 size={15} /> Delete
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
      {showRename && (
        <section className="inline-dialog dataset-action-panel">
          <Field label="Dataset name">
            <input value={editName} onChange={(event) => setEditName(event.target.value)} disabled={!dataset.editable} />
          </Field>
          <div className="flex flex-wrap gap-2">
            <button className="primary-button" onClick={saveDatasetSettings} disabled={!dataset.editable || updateDatasetMutation.isPending}>
              <Save size={16} /> Save
            </button>
            <button className="secondary-button" onClick={() => setShowRename(false)}>
              Cancel
            </button>
          </div>
        </section>
      )}
      {showDuplicate && (
        <section className="inline-dialog dataset-action-panel">
          <Field label="Duplicate name">
            <input value={cloneName} onChange={(event) => setCloneName(event.target.value)} placeholder={`${dataset.name} Copy`} />
          </Field>
          <div className="flex flex-wrap gap-2">
            <button
              className="primary-button"
              onClick={() => cloneMutation.mutate({ datasetId: dataset.id, name: cloneName || undefined })}
              disabled={cloneMutation.isPending}
            >
              <Copy size={16} /> Duplicate
            </button>
            <button className="secondary-button" onClick={() => setShowDuplicate(false)}>
              Cancel
            </button>
          </div>
        </section>
      )}
    </>
  );
}
