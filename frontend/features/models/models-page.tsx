"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Activity, Boxes, Download, MoreVertical, RefreshCw, Save, Trash2, UploadCloud, X } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { formatBytes, formatDatasetTask, isLlmModelFamily, labelColor } from "@/features/platform/utils";
import {
  Badge,
  Button,
  ButtonLink,
  CardGridSkeleton,
  EmptyState,
  Field,
  Metric,
  MutationError,
  PageHeader,
  PanelTitle,
  StatusBadge,
  useConfirmationDialog
} from "@/features/platform/ui";
import { ModelsTabs } from "./models-tabs";
import { UploadModelDialog } from "./upload-model-dialog";
import type { ModelInfo } from "@/types/api";

type SourceGroup = { key: string; title: string; hint: string; models: ModelInfo[] };

const SOURCE_ORDER: Array<{ key: string; title: string; hint: string; match: (source: string) => boolean }> = [
  { key: "reference", title: "Reference", hint: "Built-in baselines shipped with the platform.", match: (s) => s === "reference" },
  { key: "trained", title: "Trained", hint: "Models produced by your training runs.", match: (s) => s === "trained" || s === "promoted" },
  { key: "uploaded", title: "Uploaded", hint: "Custom weights you brought in.", match: (s) => s === "uploaded" }
];

function sourceTone(source: string): "neutral" | "info" | "ok" {
  if (source === "uploaded") return "ok";
  if (source === "trained" || source === "promoted") return "info";
  return "neutral";
}

export function ModelsPage() {
  const { projectId, project } = useProject();
  const router = useRouter();
  const [openMenuId, setOpenMenuId] = useState("");
  const [renameModelId, setRenameModelId] = useState("");
  const [renameDraft, setRenameDraft] = useState("");
  const [showUpload, setShowUpload] = useState(false);
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
  const groups: SourceGroup[] = SOURCE_ORDER.map((group) => ({
    key: group.key,
    title: group.title,
    hint: group.hint,
    models: models.filter((model) => group.match(model.source))
  })).filter((group) => group.models.length > 0);

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
      message: `This will remove "${model.name}" from the model catalog and delete owned files when managed by the registry.`,
      confirmLabel: "Delete model",
      onConfirm: () => deleteModel.mutate(model.id)
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Models"
        subtitle={project?.name ?? "Available trained models"}
        icon={<Boxes size={20} />}
        actions={
          <Button onClick={() => setShowUpload(true)} data-tour="models-upload">
            <UploadCloud size={16} /> Upload model
          </Button>
        }
      />
      <ModelsTabs active="catalog" />
      <section className="panel">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <PanelTitle icon={<Boxes size={18} />} title="Model catalog" dataTour="models-catalog" />
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
        ) : models.length === 0 ? (
          <EmptyState
            label="No models yet"
            icon={<Boxes size={30} />}
            centered
            description="Train a model on one of your dataset versions, or upload custom weights to see it here."
            action={
              <div className="flex gap-2">
                <ButtonLink href="/training">
                  <Activity size={16} /> Run a training job
                </ButtonLink>
                <Button variant="secondary" onClick={() => setShowUpload(true)}>
                  <UploadCloud size={16} /> Upload model
                </Button>
              </div>
            }
          />
        ) : (
          <div className="model-sections">
            {groups.map((group) => (
              <div className="model-section" key={group.key}>
                <div className="model-section-head">
                  <h3>{group.title}</h3>
                  <span>{group.hint}</span>
                </div>
                <div className="model-grid">
                  {group.models.map((model) => (
                    <ModelCard
                      key={model.id}
                      model={model}
                      isMenuOpen={openMenuId === model.id}
                      onToggleMenu={() => setOpenMenuId((value) => (value === model.id ? "" : model.id))}
                      isRenaming={renameModelId === model.id}
                      renameDraft={renameDraft}
                      onRenameDraft={setRenameDraft}
                      onOpenRename={() => openRename(model)}
                      onSaveRename={saveRename}
                      onCancelRename={() => setRenameModelId("")}
                      renamePending={renameModel.isPending}
                      onDelete={() => confirmDeleteModel(model)}
                      deletePending={deleteModel.isPending}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
        <MutationError mutations={[renameModel, deleteModel]} />
      </section>
      {showUpload && (
        <UploadModelDialog
          projectId={projectId}
          onClose={() => setShowUpload(false)}
          onUploaded={(result) => {
            setShowUpload(false);
            modelsQuery.refetch();
            router.push(`/models/${result.model.id}`);
          }}
        />
      )}
      {confirmationDialog}
    </div>
  );
}

function ModelCard({
  model,
  isMenuOpen,
  onToggleMenu,
  isRenaming,
  renameDraft,
  onRenameDraft,
  onOpenRename,
  onSaveRename,
  onCancelRename,
  renamePending,
  onDelete,
  deletePending
}: {
  model: ModelInfo;
  isMenuOpen: boolean;
  onToggleMenu: () => void;
  isRenaming: boolean;
  renameDraft: string;
  onRenameDraft: (value: string) => void;
  onOpenRename: () => void;
  onSaveRename: () => void;
  onCancelRename: () => void;
  renamePending: boolean;
  onDelete: () => void;
  deletePending: boolean;
}) {
  const editable = model.source !== "reference";
  const gated = isLlmModelFamily(model.family);
  return (
    <article className={`model-card model-card-${model.task_type.replaceAll("_", "-")} model-card-source-${model.source}`}>
      <div className="model-card-header">
        <span className="model-card-icon"><Activity size={17} /></span>
        <div className="model-card-copy">
          <Link href={`/models/${model.id}`} className="model-card-title-link" title={model.name}>
            <strong>{model.name}</strong>
          </Link>
          <span>{model.description}</span>
        </div>
        <StatusBadge status={model.available ? "available" : "missing"} />
      </div>
      <div className="card-menu">
        <button
          className="icon-button"
          onClick={onToggleMenu}
          title="Model options"
          type="button"
          aria-expanded={isMenuOpen}
        >
          <MoreVertical size={16} />
        </button>
        {isMenuOpen && (
          <div className="option-menu" role="menu">
            <button type="button" onClick={onOpenRename} disabled={!editable}>
              <Save size={15} /> Rename
            </button>
            <a href={api.modelDownloadUrl(model.id)} download>
              <Download size={15} /> Download
            </a>
            <button className="danger-menu-item" type="button" onClick={onDelete} disabled={!editable || deletePending}>
              <Trash2 size={15} /> Delete
            </button>
          </div>
        )}
      </div>
      {isRenaming && (
        <div className="model-rename-popover">
          <Field label="Model name">
            <input
              value={renameDraft}
              onChange={(event) => onRenameDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") onSaveRename();
                if (event.key === "Escape") onCancelRename();
              }}
              autoFocus
            />
          </Field>
          <div className="model-rename-actions">
            <Button onClick={onSaveRename} disabled={!renameDraft.trim() || renamePending}>
              <Save size={16} /> Save
            </Button>
            <Button variant="secondary" onClick={onCancelRename}>
              <X size={16} /> Cancel
            </Button>
          </div>
        </div>
      )}
      <div className="dataset-meta-chips">
        <span><strong>Task</strong>{formatDatasetTask(model.task_type)}</span>
        <span><strong>Family</strong>{model.family}</span>
        <span><strong>Size</strong>{formatBytes(model.size_bytes)}</span>
        <span className="model-source-chip"><strong>Source</strong><Badge tone={sourceTone(model.source)}>{model.source}</Badge></span>
      </div>
      {model.labels.length > 0 && (
        <div className="label-chip-row tag-row-compact">
          {model.labels.map((label, index) => (
            <span className="label-chip" key={label}>
              <span className="class-dot" style={{ backgroundColor: labelColor(index) }} />
              {label}
            </span>
          ))}
        </div>
      )}
      {gated && <p className="model-gate-note">{llmFamilyNote(model.family)}</p>}
      <div className="model-card-actions">
        {gated ? (
          model.family === "llm_gguf" ? (
            <>
              <Link className="secondary-button" href="/inference/chat">Serve &amp; chat</Link>
              <Link className="secondary-button" href={`/models/${model.id}`}>Export</Link>
            </>
          ) : (
            <Link className="secondary-button" href={`/models/${model.id}`}>Export &amp; serve</Link>
          )
        ) : (
          <>
            <Link className="secondary-button" href="/inference">Inference</Link>
            <Link className="secondary-button" href="/testing">Testing</Link>
          </>
        )}
        {model.training_job_id && <Link className="secondary-button" href={`/training/${model.training_job_id}`}>Training run</Link>}
      </div>
    </article>
  );
}

/** Plain-language note on the model card explaining what an LLM artifact is and
 *  what to do with it — answers "why is this only an adapter?" inline. */
function llmFamilyNote(family: string): string {
  if (family === "llm_adapter") {
    return "LoRA adapter — the small tuned delta over its base model, not a standalone model. Export to GGUF to serve and chat, or to merged 16-bit to download a full model.";
  }
  if (family === "llm_hf") {
    return "Full Hugging Face checkpoint. Export to GGUF to serve it in the chat runtime.";
  }
  return "GGUF — a quantized, self-contained model. Serve it and chat, or re-quantize to a smaller size.";
}
