"use client";

import type { Dispatch, ReactNode, SetStateAction } from "react";
import { Copy, Database, FilePlus2, FolderOpen, MoreVertical, RefreshCw, Save, Trash2, X } from "lucide-react";
import { DatasetCreatePanel } from "@/features/datasets/dataset-components";
import type {
  useCloneDatasetMutation,
  useCreateDatasetMutation,
  useDatasetCatalogQuery,
  useDeleteDatasetMutation,
  useUpdateDatasetMutation
} from "@/features/datasets/hooks";
import { SPLITS, TRAINING_SPLITS } from "@/features/platform/constants";
import { formatDatasetFormat, formatDatasetTask, isNlpTask } from "@/features/platform/utils";
import { EmptyState, Field, InlineSpinner, MutationError, PageHeader, PanelTitle, StatusBadge } from "@/features/platform/ui";
import type { DatasetSummary, TaskType } from "@/types/api";

/**
 * Dataset catalog (no dataset selected) view. Owns the create-dataset
 * form toggle, the per-card options menu, and the inline rename
 * popover; the underlying state and mutations stay with DatasetPage so
 * they behave identically whether the catalog is currently mounted.
 */
export function DatasetCatalogView({
  projectName,
  projectId,
  datasets,
  catalogQuery,
  showCreate,
  setShowCreate,
  newDatasetName,
  setNewDatasetName,
  taskType,
  setTaskType,
  allowedTaskTypes,
  labelDraft,
  setLabelDraft,
  createMutation,
  openDataset,
  catalogMenuDatasetId,
  setCatalogMenuDatasetId,
  catalogRenameDatasetId,
  setCatalogRenameDatasetId,
  catalogRenameDraft,
  setCatalogRenameDraft,
  cloneMutation,
  updateDatasetMutation,
  deleteDatasetMutation,
  onDeleteDataset,
  confirmationDialog
}: {
  projectName: string | undefined;
  projectId: string;
  datasets: DatasetSummary[];
  catalogQuery: ReturnType<typeof useDatasetCatalogQuery>;
  showCreate: boolean;
  setShowCreate: Dispatch<SetStateAction<boolean>>;
  newDatasetName: string;
  setNewDatasetName: (value: string) => void;
  taskType: TaskType;
  setTaskType: (value: TaskType) => void;
  allowedTaskTypes: TaskType[];
  labelDraft: string;
  setLabelDraft: (value: string) => void;
  createMutation: ReturnType<typeof useCreateDatasetMutation>;
  openDataset: (datasetId: string) => void;
  catalogMenuDatasetId: string;
  setCatalogMenuDatasetId: Dispatch<SetStateAction<string>>;
  catalogRenameDatasetId: string;
  setCatalogRenameDatasetId: (value: string) => void;
  catalogRenameDraft: string;
  setCatalogRenameDraft: (value: string) => void;
  cloneMutation: ReturnType<typeof useCloneDatasetMutation>;
  updateDatasetMutation: ReturnType<typeof useUpdateDatasetMutation>;
  deleteDatasetMutation: ReturnType<typeof useDeleteDatasetMutation>;
  onDeleteDataset: (dataset: DatasetSummary) => void;
  confirmationDialog: ReactNode;
}) {
  function createDataset() {
    const labels = labelDraft
      .split(",")
      .map((label) => label.trim())
      .filter(Boolean);
    if (!newDatasetName.trim()) return;
    // Summarization and QA have a fixed label; the draft belongs to whatever
    // task was selected before and must not leak into them — that is how
    // summarization datasets ended up saved with labels: ["object"].
    const isFixedLabelTask = taskType === "summarization" || taskType === "question_answering";
    if (!isFixedLabelTask && labels.length === 0) return;
    const taskLabels =
      taskType === "summarization" ? ["summary"] : taskType === "question_answering" ? ["answer"] : labels;
    createMutation.mutate({
      project_id: projectId,
      name: newDatasetName.trim(),
      task_type: taskType,
      format: isNlpTask(taskType) ? "text_folder" : taskType === "classification" ? "image_folder" : "yolo",
      labels: taskLabels
    });
  }

  function openCatalogDatasetRename(dataset: DatasetSummary) {
    if (!dataset.editable) return;
    setCatalogRenameDatasetId(dataset.id);
    setCatalogRenameDraft(dataset.name);
    setCatalogMenuDatasetId("");
  }

  function saveCatalogDatasetRename() {
    if (!catalogRenameDatasetId || !catalogRenameDraft.trim()) return;
    updateDatasetMutation.mutate({
      datasetId: catalogRenameDatasetId,
      name: catalogRenameDraft.trim()
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Datasets" subtitle={projectName ?? "Project"} icon={<Database size={20} />} />
      <section className="panel">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <PanelTitle icon={<Database size={18} />} title="Dataset Catalog" />
          <div className="flex flex-wrap gap-2">
            {catalogQuery.isFetching && <InlineSpinner label="Refreshing" />}
            <button className="secondary-button" onClick={() => catalogQuery.refetch()}>
              <RefreshCw size={16} /> Refresh
            </button>
            <button className="primary-button" onClick={() => setShowCreate((value) => !value)}>
              <FilePlus2 size={16} /> Add Dataset
            </button>
          </div>
        </div>
        {showCreate && (
          <DatasetCreatePanel
            newDatasetName={newDatasetName}
            setNewDatasetName={setNewDatasetName}
            taskType={taskType}
            setTaskType={setTaskType}
            allowedTaskTypes={allowedTaskTypes}
            labelDraft={labelDraft}
            setLabelDraft={setLabelDraft}
            onCreate={createDataset}
            pending={createMutation.isPending}
          />
        )}
        <div className="dataset-catalog-grid">
          {datasets.map((dataset) => {
            const totalImages = SPLITS.reduce((total, splitName) => total + (dataset.splits[splitName]?.item_count ?? dataset.splits[splitName]?.image_count ?? 0), 0);
            return (
              <article
                className={`dataset-card dataset-card-${dataset.task_type.replaceAll("_", "-")} ${
                  dataset.editable ? "dataset-card-editable" : "dataset-card-readonly"
                }`}
                key={dataset.id}
              >
                <button className="dataset-card-main" type="button" onClick={() => openDataset(dataset.id)}>
                  <div className="dataset-card-header">
                    <span className="dataset-card-icon"><Database size={17} /></span>
                    <div className="dataset-card-title">
                      <strong title={dataset.name}>{dataset.name}</strong>
                      <span title={`${formatDatasetTask(dataset.task_type)} / ${formatDatasetFormat(dataset.format)}`}>
                        {formatDatasetTask(dataset.task_type)} / {formatDatasetFormat(dataset.format)}
                      </span>
                    </div>
                  </div>
                  <div className="dataset-card-summary">
                    <span><strong>{totalImages}</strong> {isNlpTask(dataset.task_type) ? "texts" : "images"}</span>
                    <span><strong>{dataset.labels.length}</strong> labels</span>
                  </div>
                  <div className="dataset-card-stats">
                    {TRAINING_SPLITS.map((splitName) => (
                      <span key={splitName}>{splitName}: {dataset.splits[splitName]?.item_count ?? dataset.splits[splitName]?.image_count ?? 0}</span>
                    ))}
                  </div>
                </button>
                <StatusBadge status={dataset.editable ? "editable" : "read-only"} />
                <div className="card-menu">
                  <button
                    className="icon-button"
                    onClick={() => setCatalogMenuDatasetId((value) => (value === dataset.id ? "" : dataset.id))}
                    title="Dataset options"
                    type="button"
                    aria-expanded={catalogMenuDatasetId === dataset.id}
                  >
                    <MoreVertical size={16} />
                  </button>
                  {catalogMenuDatasetId === dataset.id && (
                    <div className="option-menu" role="menu">
                      <button type="button" onClick={() => openDataset(dataset.id)}>
                        <FolderOpen size={15} /> Open
                      </button>
                      <button
                        type="button"
                        onClick={() => openCatalogDatasetRename(dataset)}
                        disabled={!dataset.editable}
                      >
                        <Save size={15} /> Rename
                      </button>
                      <button
                        type="button"
                        onClick={() => cloneMutation.mutate({ datasetId: dataset.id })}
                        disabled={cloneMutation.isPending}
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
                {catalogRenameDatasetId === dataset.id && (
                  <div className="card-rename-popover">
                    <Field label="Dataset name">
                      <input
                        value={catalogRenameDraft}
                        onChange={(event) => setCatalogRenameDraft(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") saveCatalogDatasetRename();
                          if (event.key === "Escape") setCatalogRenameDatasetId("");
                        }}
                        autoFocus
                      />
                    </Field>
                    <div className="card-rename-actions">
                      <button className="primary-button" onClick={saveCatalogDatasetRename} disabled={!catalogRenameDraft.trim() || updateDatasetMutation.isPending}>
                        <Save size={16} /> Save
                      </button>
                      <button className="secondary-button" onClick={() => setCatalogRenameDatasetId("")}>
                        <X size={16} /> Cancel
                      </button>
                    </div>
                  </div>
                )}
              </article>
            );
          })}
          {datasets.length === 0 && <EmptyState label="No datasets yet." icon={<Database size={30} />} centered description="Create or import a dataset to start annotating and training models." />}
        </div>
        <MutationError mutations={[createMutation, updateDatasetMutation, cloneMutation, deleteDatasetMutation]} />
      </section>
      {confirmationDialog}
    </div>
  );
}
