"use client";

import Link from "next/link";
import { useState } from "react";
import { Activity, Download, MoreVertical, RefreshCw, Save, Trash2, X } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { formatDatasetTask, labelColor } from "@/features/platform/utils";
import { CardGridSkeleton, EmptyState, Field, Metric, MutationError, PageHeader, PanelTitle, StatusBadge, useConfirmationDialog } from "@/features/platform/ui";
import type { ModelInfo } from "@/types/api";

export function ModelsPage() {
  const { projectId, project } = useProject();
  const [openMenuId, setOpenMenuId] = useState("");
  const [renameModelId, setRenameModelId] = useState("");
  const [renameDraft, setRenameDraft] = useState("");
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const modelsQuery = useQuery({
    queryKey: ["models", "catalog", projectId],
    queryFn: () => api.models(false, projectId)
  });
  const renameModel = useMutation({
    mutationFn: ({ modelId, name }: { modelId: string; name: string }) => api.renameModel(modelId, name),
    onSuccess: async () => {
      setRenameModelId("");
      setRenameDraft("");
      setOpenMenuId("");
      await modelsQuery.refetch();
    }
  });
  const deleteModel = useMutation({
    mutationFn: api.deleteModel,
    onSuccess: async () => {
      setOpenMenuId("");
      setRenameModelId("");
      await modelsQuery.refetch();
    }
  });
  const models = modelsQuery.data ?? [];
  const availableCount = models.filter((model) => model.available).length;

  function openRename(model: ModelInfo) {
    setRenameModelId(model.id);
    setRenameDraft(model.name);
    setOpenMenuId("");
  }

  function saveRename() {
    if (!renameModelId || !renameDraft.trim()) return;
    renameModel.mutate({ modelId: renameModelId, name: renameDraft.trim() });
  }

  function confirmDeleteModel(model: ModelInfo) {
    confirm({
      title: "Delete model?",
      message: `This will remove "${model.name}" from the model catalog and delete owned trained-model files when managed by the registry.`,
      confirmLabel: "Delete model",
      onConfirm: () => deleteModel.mutate(model.id)
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Models" subtitle={project?.name ?? "Available trained models"} icon={<Activity size={20} />} />
      <section className="panel">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <PanelTitle icon={<Activity size={18} />} title="Available Models" />
          <div className="flex flex-wrap gap-2">
            <Metric label="Registered" value={models.length} />
            <Metric label="Available" value={availableCount} />
            <button className="secondary-button" onClick={() => modelsQuery.refetch()}>
              <RefreshCw size={16} /> Refresh
            </button>
          </div>
        </div>
        {modelsQuery.isLoading ? (
          <CardGridSkeleton count={4} />
        ) : (
          <div className="model-grid">
            {models.map((model) => {
              const editable = model.source !== "reference";
              return (
                <article className={`model-card model-card-${model.task_type.replaceAll("_", "-")} model-card-source-${model.source}`} key={model.id}>
                  <div className="model-card-header">
                    <span className="model-card-icon"><Activity size={17} /></span>
                    <div className="model-card-copy">
                      <strong title={model.name}>{model.name}</strong>
                      <span>{model.description}</span>
                    </div>
                    <StatusBadge status={model.available ? "available" : "missing"} />
                  </div>
                  <div className="card-menu">
                    <button
                      className="icon-button"
                      onClick={() => setOpenMenuId((value) => (value === model.id ? "" : model.id))}
                      title="Model options"
                      type="button"
                      aria-expanded={openMenuId === model.id}
                    >
                      <MoreVertical size={16} />
                    </button>
                    {openMenuId === model.id && (
                      <div className="option-menu" role="menu">
                        <button type="button" onClick={() => openRename(model)} disabled={!editable}>
                          <Save size={15} /> Rename
                        </button>
                        <a href={api.modelDownloadUrl(model.id)} download onClick={() => setOpenMenuId("")}>
                          <Download size={15} /> Download
                        </a>
                        <button
                          className="danger-menu-item"
                          type="button"
                          onClick={() => confirmDeleteModel(model)}
                          disabled={!editable || deleteModel.isPending}
                        >
                          <Trash2 size={15} /> Delete
                        </button>
                      </div>
                    )}
                  </div>
                  {renameModelId === model.id && (
                    <div className="model-rename-popover">
                      <Field label="Model name">
                        <input
                          value={renameDraft}
                          onChange={(event) => setRenameDraft(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") saveRename();
                            if (event.key === "Escape") setRenameModelId("");
                          }}
                          autoFocus
                        />
                      </Field>
                      <div className="model-rename-actions">
                        <button className="primary-button" onClick={saveRename} disabled={!renameDraft.trim() || renameModel.isPending}>
                          <Save size={16} /> Save
                        </button>
                        <button className="secondary-button" onClick={() => setRenameModelId("")}>
                          <X size={16} /> Cancel
                        </button>
                      </div>
                    </div>
                  )}
                  <div className="dataset-meta-chips">
                    <span><strong>Task</strong>{formatDatasetTask(model.task_type)}</span>
                    <span><strong>Family</strong>{model.family}</span>
                    <span><strong>Source</strong>{model.source}</span>
                  </div>
                  <div className="label-chip-row tag-row-compact">
                    {model.labels.map((label, index) => (
                      <span className="label-chip" key={label}>
                        <span className="class-dot" style={{ backgroundColor: labelColor(index) }} />
                        {label}
                      </span>
                    ))}
                  </div>
                  <div className="model-card-actions">
                    <Link className="secondary-button" href="/inference">Inference</Link>
                    <Link className="secondary-button" href="/testing">Testing</Link>
                    {model.training_job_id && <Link className="secondary-button" href={`/training/${model.training_job_id}`}>Training run</Link>}
                  </div>
                </article>
              );
            })}
            {models.length === 0 && <EmptyState label="No models registered" />}
          </div>
        )}
        <MutationError mutations={[renameModel, deleteModel]} />
      </section>
      {confirmationDialog}
    </div>
  );
}
