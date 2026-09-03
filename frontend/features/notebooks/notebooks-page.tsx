"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Copy, FileCode2, MoreVertical, NotebookPen, Trash2 } from "lucide-react";
import { useProject } from "@/components/app-shell";
import {
  useCreateNotebookMutation,
  useDeleteNotebookMutation,
  useDuplicateNotebookMutation,
  useNotebookTemplatesQuery,
  useNotebooksQuery
} from "@/features/notebooks/hooks";
import { RuntimeBanner } from "@/features/notebooks/runtime-banner";
import {
  Badge,
  Button,
  CardGridSkeleton,
  EmptyState,
  Field,
  MutationError,
  PageHeader,
  PanelTitle,
  useConfirmationDialog
} from "@/features/platform/ui";
import type { NotebookSummary, NotebookTemplate } from "@/types/api";

/**
 * The notebooks in this project, and the four ways to start one.
 *
 * Templates lead rather than sitting behind a "new from template" menu, because
 * they are the SDK's documentation: a user discovers what `orinth` can do by
 * opening "Dataset EDA", not by reading a signature list. That is also the
 * constraint that keeps the SDK small — if it does not fit in four notebooks,
 * it is too big.
 */

function relative(iso: string | null | undefined): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const minutes = Math.floor((Date.now() - then) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function TemplateCard({
  template,
  onPick,
  pending
}: {
  template: NotebookTemplate;
  onPick: () => void;
  pending: boolean;
}) {
  return (
    <button type="button" className="nb-template" onClick={onPick} disabled={pending}>
      <FileCode2 size={15} aria-hidden="true" />
      <strong>{template.name}</strong>
      <span className="form-caption">{template.description}</span>
    </button>
  );
}

export function NotebooksPage() {
  const router = useRouter();
  const { projectId, project } = useProject();
  const [name, setName] = useState("");
  const [menuId, setMenuId] = useState("");
  const { confirm, confirmationDialog } = useConfirmationDialog();

  const notebooksQuery = useNotebooksQuery(projectId);
  const templatesQuery = useNotebookTemplatesQuery();
  const createMutation = useCreateNotebookMutation((id) => router.push(`/notebooks/${id}`));
  const duplicateMutation = useDuplicateNotebookMutation();
  const deleteMutation = useDeleteNotebookMutation();

  const notebooks = notebooksQuery.data ?? [];

  function create(templateId?: string) {
    const template = (templatesQuery.data ?? []).find((entry) => entry.id === templateId);
    createMutation.mutate({
      project_id: projectId,
      // A template-started notebook is named after the template unless the user
      // typed something, so the list does not fill with "Untitled".
      name: name.trim() || template?.name || "Untitled notebook",
      template_id: templateId ?? null
    });
    setName("");
  }

  function renderCard(notebook: NotebookSummary) {
    return (
      <article className="nb-card" key={notebook.id}>
        <button
          type="button"
          className="nb-card-main"
          onClick={() => router.push(`/notebooks/${notebook.id}`)}
        >
          <div className="nb-card-head">
            <NotebookPen size={15} aria-hidden="true" />
            <strong title={notebook.name}>{notebook.name}</strong>
          </div>
          <div className="nb-card-meta">
            <span>
              {notebook.cell_count} cell{notebook.cell_count === 1 ? "" : "s"}
            </span>
            {notebook.updated_at && <span>Updated {relative(notebook.updated_at)}</span>}
            {!notebook.valid && (
              <Badge tone="fail" title="The .ipynb on disk could not be parsed.">
                unreadable
              </Badge>
            )}
          </div>
        </button>
        <div className="card-menu">
          <button
            className="icon-button"
            type="button"
            title="Notebook options"
            aria-expanded={menuId === notebook.id}
            onClick={() => setMenuId((value) => (value === notebook.id ? "" : notebook.id))}
          >
            <MoreVertical size={16} />
          </button>
          {menuId === notebook.id && (
            <div className="option-menu" role="menu">
              <button
                type="button"
                onClick={() => {
                  duplicateMutation.mutate(notebook.id);
                  setMenuId("");
                }}
              >
                <Copy size={15} /> Duplicate
              </button>
              <button
                className="danger-menu-item"
                type="button"
                onClick={() => {
                  setMenuId("");
                  confirm({
                    title: `Delete "${notebook.name}"?`,
                    message:
                      "The notebook, its outputs, and every run logged from it are removed. " +
                      "Datasets it registered are not affected.",
                    confirmLabel: "Delete",
                    tone: "danger",
                    onConfirm: () => deleteMutation.mutate(notebook.id)
                  });
                }}
              >
                <Trash2 size={15} /> Delete
              </button>
            </div>
          )}
        </div>
      </article>
    );
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Notebooks"
        subtitle={project?.name ?? "Project"}
        icon={<NotebookPen size={20} />}
      />

      <RuntimeBanner />

      <section className="panel">
        <PanelTitle icon={<FileCode2 size={18} />} title="Start a notebook" />
        <p className="form-caption nb-intro">
          Cells run in this workspace&apos;s own Python. <code>import orinth</code> reaches every
          dataset and model here, and a cleaned dataframe registers back as a trainable dataset.
        </p>
        <Field label="Name" hint="optional">
          <input
            className="text-input"
            value={name}
            placeholder="Named after the template if left blank"
            onChange={(event) => setName(event.target.value)}
          />
        </Field>
        <div className="nb-template-grid">
          {(templatesQuery.data ?? []).map((template) => (
            <TemplateCard
              key={template.id}
              template={template}
              pending={createMutation.isPending}
              onPick={() => create(template.id)}
            />
          ))}
        </div>
        <MutationError mutations={[createMutation, duplicateMutation, deleteMutation]} />
      </section>

      <section className="panel">
        <PanelTitle icon={<NotebookPen size={18} />} title="Your notebooks" />
        {notebooksQuery.isLoading ? (
          <CardGridSkeleton count={3} />
        ) : notebooks.length === 0 ? (
          <EmptyState
            icon={<NotebookPen size={26} />}
            label="No notebooks in this project yet."
            description="Pick a template above — Dataset EDA is the shortest way to see what is in your data."
          />
        ) : (
          <div className="nb-grid">{notebooks.map(renderCard)}</div>
        )}
      </section>

      {confirmationDialog}
    </div>
  );
}
