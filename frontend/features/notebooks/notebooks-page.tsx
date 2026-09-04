"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import {
  Copy,
  FileCode2,
  LayoutTemplate,
  MoreVertical,
  NotebookPen,
  Pencil,
  Plus,
  Trash2
} from "lucide-react";
import { useProject } from "@/components/app-shell";
import {
  useCreateNotebookMutation,
  useDeleteNotebookMutation,
  useDuplicateNotebookMutation,
  useNotebookTemplatesQuery,
  useNotebooksQuery,
  useRenameNotebookMutation
} from "@/features/notebooks/hooks";
import {
  Badge,
  Button,
  CardGridSkeleton,
  EmptyState,
  Field,
  Modal,
  MutationError,
  PageHeader,
  Pager,
  PanelTitle,
  useConfirmationDialog
} from "@/features/platform/ui";
import type { NotebookSummary, NotebookTemplate } from "@/types/api";

/**
 * The notebooks in this project, and the ways to start one.
 *
 * Templates lead rather than sitting behind a "new from template" menu, because
 * they are the SDK's documentation: a user discovers what `orinth` can do by
 * opening "Dataset EDA", not by reading a signature list.
 *
 * Twenty-two of them stacked in category sections was a wall to scroll, so they
 * are **tabbed** by category with an All tab in front, and paged six at a time.
 * Tabs rather than sections because the categories are alternatives — you are
 * looking for a training notebook *or* a data one — and a tab bar says that
 * while a stack of headings makes you scan all of them to find out.
 *
 * The category list is derived from what the backend sent, in the order it sent
 * it. A template that names a new category appears under a new tab with no
 * change here.
 */

//: Six fills two rows of the card grid at most widths, which is the point where
//: a page of options stops being scannable at a glance.
const TEMPLATES_PER_PAGE = 6;
//: Nine is three rows of cards. The notebook list grows without bound — one per
//: experiment — and an unpaged list of forty is a scroll, not a chooser.
const NOTEBOOKS_PER_PAGE = 9;
const ALL_TAB = "All";

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

/**
 * The category tabs, in the order the backend sent the templates.
 *
 * Derived rather than declared: the server owns the ordering (it is the order
 * the work happens in, not alphabetical), and a hardcoded list here would be a
 * second definition to keep in sync — and would silently drop a category the
 * next template introduces.
 */
export function categoriesOf(templates: NotebookTemplate[]): string[] {
  const seen: string[] = [];
  for (const template of templates) {
    if (!seen.includes(template.category)) seen.push(template.category);
  }
  return seen;
}

export function pageOf<T>(items: T[], page: number, size: number): T[] {
  return items.slice(page * size, page * size + size);
}

export function pageCountOf(total: number, size: number): number {
  return Math.max(1, Math.ceil(total / size));
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
  const [tab, setTab] = useState(ALL_TAB);
  //: The notebook being renamed inline, and its draft name. One at a time —
  //: two open editors on one list is a state nobody asked for.
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null);
  const [templatesOpen, setTemplatesOpen] = useState(false);
  const [templatePage, setTemplatePage] = useState(0);
  const [notebookPage, setNotebookPage] = useState(0);
  const { confirm, confirmationDialog } = useConfirmationDialog();

  const notebooksQuery = useNotebooksQuery(projectId);
  const templatesQuery = useNotebookTemplatesQuery();
  const createMutation = useCreateNotebookMutation((id) => router.push(`/notebooks/${id}`));
  const duplicateMutation = useDuplicateNotebookMutation();
  const deleteMutation = useDeleteNotebookMutation();
  const renameMutation = useRenameNotebookMutation();

  const notebooks = notebooksQuery.data ?? [];
  const templates = useMemo(() => templatesQuery.data ?? [], [templatesQuery.data]);
  const categories = useMemo(() => categoriesOf(templates), [templates]);
  const visibleTemplates = useMemo(
    () => (tab === ALL_TAB ? templates : templates.filter((entry) => entry.category === tab)),
    [tab, templates]
  );

  // A tab with fewer pages than the one before it would otherwise leave the
  // pager pointing past the end, which renders as an empty grid.
  const templatePages = pageCountOf(visibleTemplates.length, TEMPLATES_PER_PAGE);
  const currentTemplatePage = Math.min(templatePage, templatePages - 1);
  const notebookPages = pageCountOf(notebooks.length, NOTEBOOKS_PER_PAGE);
  const currentNotebookPage = Math.min(notebookPage, notebookPages - 1);

  function create(templateId?: string) {
    const template = templates.find((entry) => entry.id === templateId);
    setTemplatesOpen(false);
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
    if (renaming?.id === notebook.id) {
      return (
        <article className="nb-card" key={notebook.id}>
          <form
            className="nb-card-rename"
            onSubmit={(event) => {
              event.preventDefault();
              const name = renaming.name.trim();
              setRenaming(null);
              // Empty or unchanged is a cancel. The API would accept "" and
              // leave a blank row in the list.
              if (!name || name === notebook.name) return;
              renameMutation.mutate({ notebookId: notebook.id, name });
            }}
          >
            <input
              className="text-input"
              value={renaming.name}
              autoFocus
              aria-label={`Rename ${notebook.name}`}
              onChange={(event) => setRenaming({ id: notebook.id, name: event.target.value })}
              onBlur={(event) => event.currentTarget.form?.requestSubmit()}
              onKeyDown={(event) => {
                if (event.key === "Escape") setRenaming(null);
              }}
            />
            <p className="form-caption">Enter to save · Esc to cancel</p>
          </form>
        </article>
      );
    }
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
                  setMenuId("");
                  setRenaming({ id: notebook.id, name: notebook.name });
                }}
              >
                <Pencil size={15} /> Rename
              </button>
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

      {/* Two ways in, side by side. The runtime lives in the notebook now —
          you start one because you are about to run something, and that is
          there, not here. */}
      <section className="panel nb-start">
        <div className="nb-start-copy">
          <PanelTitle icon={<FileCode2 size={18} />} title="Start a notebook" />
          <p className="form-caption nb-intro">
            Cells run in this workspace&apos;s own Python. <code>import orinth</code> reaches
            every dataset and model here, and a cleaned dataframe registers back as a trainable
            dataset.
          </p>
        </div>
        <div className="nb-start-actions">
          <Button
            variant="primary"
            onClick={() => create()}
            disabled={createMutation.isPending}
            title="An empty notebook with orinth imported"
          >
            <Plus size={16} /> New notebook
          </Button>
          <Button
            variant="secondary"
            onClick={() => setTemplatesOpen(true)}
            disabled={createMutation.isPending}
          >
            <LayoutTemplate size={16} /> Open a template
            <span className="nb-start-count">{templates.length}</span>
          </Button>
        </div>
        <MutationError
          mutations={[createMutation, duplicateMutation, deleteMutation, renameMutation]}
        />
      </section>

      {templatesOpen && (
        <Modal
          title="Start from a template"
          subtitle="Each one runs against this workspace — they are the SDK's documentation."
          onClose={() => setTemplatesOpen(false)}
          className="nb-template-modal"
        >
          <div className="modal-body-padded">
            <Field label="Name" hint="optional">
              <input
                className="text-input"
                value={name}
                placeholder="Named after the template if left blank"
                onChange={(event) => setName(event.target.value)}
              />
            </Field>
            {templatesQuery.isLoading ? (
              <CardGridSkeleton count={3} />
            ) : (
              <>
                <div className="nb-template-bar">
                  <nav
                    className="segmented-control nb-template-tabs"
                    aria-label="Template categories"
                  >
                    {[ALL_TAB, ...categories].map((category) => {
                      const count =
                        category === ALL_TAB
                          ? templates.length
                          : templates.filter((entry) => entry.category === category).length;
                      return (
                        <button
                          key={category}
                          type="button"
                          className={tab === category ? "segmented-active" : ""}
                          aria-current={tab === category ? "true" : undefined}
                          onClick={() => {
                            setTab(category);
                            setTemplatePage(0);
                          }}
                        >
                          {category}
                          <span className="segmented-count">{count}</span>
                        </button>
                      );
                    })}
                  </nav>
                  <Pager
                    page={currentTemplatePage}
                    pageCount={templatePages}
                    onChange={setTemplatePage}
                    label="Template pages"
                    unit="templates"
                  />
                </div>
                <div className="nb-template-grid">
                  {pageOf(visibleTemplates, currentTemplatePage, TEMPLATES_PER_PAGE).map(
                    (template) => (
                      <TemplateCard
                        key={template.id}
                        template={template}
                        pending={createMutation.isPending}
                        onPick={() => create(template.id)}
                      />
                    )
                  )}
                </div>
              </>
            )}
          </div>
        </Modal>
      )}

      <section className="panel">
        <div className="nb-list-head">
          <PanelTitle icon={<NotebookPen size={18} />} title="Your notebooks" />
          <Pager
            page={currentNotebookPage}
            pageCount={notebookPages}
            onChange={setNotebookPage}
            label="Notebook pages"
            unit="notebooks"
          />
        </div>
        {notebooksQuery.isLoading ? (
          <CardGridSkeleton count={3} />
        ) : notebooks.length === 0 ? (
          <EmptyState
            icon={<NotebookPen size={26} />}
            label="No notebooks in this project yet."
            description="Pick a template above — Dataset EDA is the shortest way to see what is in your data."
          />
        ) : (
          <div className="nb-grid">
            {pageOf(notebooks, currentNotebookPage, NOTEBOOKS_PER_PAGE).map(renderCard)}
          </div>
        )}
      </section>

      {confirmationDialog}
    </div>
  );
}
