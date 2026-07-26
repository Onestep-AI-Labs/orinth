"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowLeft, Boxes, Download, Save, Trash2, X } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { formatBytes, formatDatasetTask, isLlmModelFamily, labelColor } from "@/features/platform/utils";
import {
  Badge,
  Button,
  Field,
  MutationError,
  PageHeader,
  PageSkeleton,
  useConfirmationDialog
} from "@/features/platform/ui";
import { toast } from "@/features/platform/toast";
import { ExportPanel, ExportToGgufCta, ServingPanel } from "@/features/models/llm-panels";
import type { ModelInfo } from "@/types/api";

function sourceTone(source: string): "neutral" | "info" | "ok" {
  if (source === "uploaded") return "ok";
  if (source === "trained" || source === "promoted") return "info";
  return "neutral";
}

export function ModelDetailPage({ modelId }: { modelId: string }) {
  const router = useRouter();
  const [renaming, setRenaming] = useState(false);
  const [renameDraft, setRenameDraft] = useState("");
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const modelsQuery = useQuery({
    queryKey: ["models", "catalog", "all"],
    queryFn: () => api.models(false)
  });
  const model = modelsQuery.data?.find((item) => item.id === modelId);

  const renameModel = useMutation({
    mutationFn: (name: string) => api.renameModel(modelId, name),
    onSuccess: async () => {
      setRenaming(false);
      toast.success("Model renamed");
      await modelsQuery.refetch();
    }
  });
  const deleteModel = useMutation({
    mutationFn: () => api.deleteModel(modelId),
    onSuccess: () => {
      toast.success("Model deleted");
      router.push("/models");
    }
  });

  if (modelsQuery.isLoading) return <PageSkeleton title="Loading model" />;

  if (!model) {
    return (
      <div className="space-y-5">
        <PageHeader
          title="Model not found"
          subtitle="This model is no longer in the registry."
          icon={<Boxes size={20} />}
          actions={<Link className="secondary-button" href="/models"><ArrowLeft size={16} /> Back to models</Link>}
        />
      </div>
    );
  }

  const editable = model.source !== "reference";
  const gated = isLlmModelFamily(model.family);
  const artifacts = Object.entries(model.paths);

  function startRename() {
    if (!model) return;
    setRenameDraft(model.name);
    setRenaming(true);
  }

  function confirmDelete() {
    if (!model) return;
    confirm({
      title: "Delete model?",
      message: `This removes "${model.name}" from the registry and deletes its owned files.`,
      confirmLabel: "Delete model",
      onConfirm: () => deleteModel.mutate()
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Model"
        title={model.name}
        subtitle={model.description}
        icon={<Boxes size={20} />}
        actions={
          <div className="flex flex-wrap gap-2">
            <Link className="secondary-button" href="/models"><ArrowLeft size={16} /> Back</Link>
            <a className="secondary-button" href={api.modelDownloadUrl(model.id)} download>
              <Download size={16} /> Download
            </a>
            {editable && (
              <>
                <Button variant="secondary" onClick={startRename}><Save size={16} /> Rename</Button>
                <Button variant="danger" onClick={confirmDelete} disabled={deleteModel.isPending}>
                  <Trash2 size={16} /> Delete
                </Button>
              </>
            )}
          </div>
        }
      />

      {renaming && (
        <section className="panel model-rename-panel">
          <Field label="Model name">
            <input
              value={renameDraft}
              onChange={(event) => setRenameDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && renameDraft.trim()) renameModel.mutate(renameDraft.trim());
                if (event.key === "Escape") setRenaming(false);
              }}
              autoFocus
            />
          </Field>
          <div className="model-rename-actions">
            <Button onClick={() => renameModel.mutate(renameDraft.trim())} disabled={!renameDraft.trim() || renameModel.isPending}>
              <Save size={16} /> Save
            </Button>
            <Button variant="secondary" onClick={() => setRenaming(false)}><X size={16} /> Cancel</Button>
          </div>
        </section>
      )}

      <section className="panel">
        <div className="model-detail-grid">
          <MetaRow label="Family" value={model.family} />
          <MetaRow label="Task" value={formatDatasetTask(model.task_type)} />
          <MetaRow label="Source" value={<Badge tone={sourceTone(model.source)}>{model.source}</Badge>} />
          <MetaRow label="Availability" value={<Badge tone={model.available ? "ok" : "fail"}>{model.available ? "available" : "missing"}</Badge>} />
          <MetaRow label="Size on disk" value={formatBytes(model.size_bytes)} />
          {model.format && <MetaRow label="Format" value={model.format} />}
          {model.base_model_id && <MetaRow label="Base model" value={model.base_model_id} />}
          {model.training_job_id && (
            <MetaRow label="Training run" value={<Link className="model-detail-link" href={`/training/${model.training_job_id}`}>{model.training_job_id}</Link>} />
          )}
          {model.created_at && <MetaRow label="Registered" value={new Date(model.created_at).toLocaleString()} />}
        </div>

        <div className="label-chip-row tag-row-compact model-detail-labels">
          {model.labels.map((label, index) => (
            <span className="label-chip" key={label}>
              <span className="class-dot" style={{ backgroundColor: labelColor(index) }} />
              {label}
            </span>
          ))}
        </div>
      </section>

      <section className="panel">
        <h3 className="model-detail-section-title">Artifacts</h3>
        {artifacts.length === 0 ? (
          <p className="model-detail-empty">This model has no on-disk artifacts (built-in baseline).</p>
        ) : (
          <ul className="model-artifact-list">
            {artifacts.map(([key, path]) => (
              <li key={key}>
                <span className="model-artifact-key">{key}</span>
                <span className="model-artifact-path" title={path}>{path}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      {gated && model.available && (
        <>
          <ExportPanel model={model} />
          {model.family === "llm_gguf" ? <ServingPanel model={model} /> : <ExportToGgufCta />}
        </>
      )}

      <MutationError mutations={[renameModel, deleteModel]} />
      {confirmationDialog}
    </div>
  );
}

function MetaRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="model-meta-row">
      <span className="model-meta-label">{label}</span>
      <span className="model-meta-value">{value}</span>
    </div>
  );
}
