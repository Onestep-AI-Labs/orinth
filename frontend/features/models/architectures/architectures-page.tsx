"use client";

import { useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Copy, Network, Plus, Search, Trash2, Upload } from "lucide-react";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import {
  Badge,
  Button,
  ButtonLink,
  CardGridSkeleton,
  EmptyState,
  Field,
  IconButton,
  MutationError,
  PageHeader,
  TaskSelect,
  useConfirmationDialog
} from "@/features/platform/ui";
import { allowedTaskTypesForProject, formatDatasetTask } from "@/features/platform/utils";
import type { TaskType } from "@/types/api";
import { ModelsTabs } from "../models-tabs";

// Enough to fill a wide grid twice over without becoming a wall of cards.
const PAGE_SIZE = 12;
// The picker sits inside a panel above the list, so it pages sooner: eight
// presets plus the pinned blank card is three rows on a wide screen.
const TEMPLATE_PAGE_SIZE = 8;

export function ArchitecturesPage() {
  const router = useRouter();
  const { projectId, project } = useProject();
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const fileInput = useRef<HTMLInputElement>(null);
  const [creating, setCreating] = useState(false);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [templateSearch, setTemplateSearch] = useState("");
  const [templatePage, setTemplatePage] = useState(1);
  // The task a blank canvas is created for. Null means "whatever the project
  // works in" — the auto-detected default — until the user picks something.
  const [task, setTask] = useState<TaskType | null>(null);
  // Opt-in narrowing. Off by default: hiding presets by task is what made the
  // NLP and LLM ones invisible.
  const [onlyTask, setOnlyTask] = useState(false);

  const architecturesQuery = useQuery({
    queryKey: ["architectures", projectId],
    queryFn: () => api.architectures(projectId)
  });
  const templatesQuery = useQuery({
    queryKey: ["architecture-templates"],
    queryFn: () => api.architectureTemplates()
  });

  const createMutation = useMutation({
    mutationFn: (input: { name: string; templateId?: string; taskType: TaskType }) =>
      api.createArchitecture({
        name: input.name,
        project_id: projectId,
        task_type: input.taskType,
        template_id: input.templateId ?? null,
        description: null,
        graph: null
      }),
    onSuccess: (created) => router.push(`/models/architectures/${created.id}`)
  });

  const importMutation = useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      form.append("project_id", projectId);
      return api.importArchitecture(form);
    },
    onSuccess: (imported) => router.push(`/models/architectures/${imported.id}`)
  });

  const duplicateMutation = useMutation({
    mutationFn: (architectureId: string) => api.duplicateArchitecture(architectureId),
    onSuccess: () => architecturesQuery.refetch()
  });

  const deleteMutation = useMutation({
    mutationFn: (architectureId: string) => api.deleteArchitecture(architectureId),
    onSuccess: () => architecturesQuery.refetch()
  });

  const all = architecturesQuery.data ?? [];

  // Filtering and paging happen client-side: the list endpoint returns
  // summaries without graphs, so even a few hundred architectures are a small
  // payload and a round trip per keystroke would be worse than no search.
  const matches = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return all;
    return all.filter((item) =>
      `${item.name} ${item.description ?? ""} ${item.task_type}`.toLowerCase().includes(query)
    );
  }, [all, search]);

  const pageCount = Math.max(1, Math.ceil(matches.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount);
  const architectures = matches.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE
  );

  const projectTasks = allowedTaskTypesForProject(project);

  /**
   * Tasks the studio can build, which is not the same set as the project's.
   *
   * `language_modeling` is the reason: it exists for architectures composed
   * here and no project declares it, so gating the picker on the project's
   * tasks made all seventeen LLM presets — GPT-2, Llama, Qwen, DeepSeek, Kimi —
   * unreachable in every project. The project's tasks lead the list because
   * they are the likely intent; the rest stay selectable.
   */
  const studioTasks = useMemo(() => {
    const fromPresets = (templatesQuery.data ?? []).map(
      (template) => template.task_type as TaskType
    );
    return [...new Set<TaskType>([...projectTasks, ...fromPresets])];
  }, [templatesQuery.data, projectTasks.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectedTask: TaskType = task ?? projectTasks[0] ?? "classification";

  /**
   * Presets are sorted by the selected task, never hidden by it.
   *
   * Filtering them was the bug: a preset is the fastest way to learn what the
   * canvas can express, and a vision project that could not reach BERT could
   * not discover that the studio builds text models at all. `onlyTask` is an
   * opt-in narrowing for when the list is genuinely in the way.
   */
  const allTemplates = useMemo(() => {
    const entries = [...(templatesQuery.data ?? [])];
    if (onlyTask) return entries.filter((template) => template.task_type === selectedTask);
    return entries.sort((a, b) => {
      const rank = (task_type: string) => (task_type === selectedTask ? 0 : 1);
      return rank(a.task_type) - rank(b.task_type);
    });
  }, [templatesQuery.data, onlyTask, selectedTask]);

  const templates = useMemo(() => {
    const query = templateSearch.trim().toLowerCase();
    if (!query) return allTemplates;
    return allTemplates.filter((template) =>
      `${template.name} ${template.description} ${template.task_type}`
        .toLowerCase()
        .includes(query)
    );
  }, [allTemplates, templateSearch]);
  const templatePageCount = Math.max(1, Math.ceil(templates.length / TEMPLATE_PAGE_SIZE));
  const currentTemplatePage = Math.min(templatePage, templatePageCount);
  const templatePageItems = templates.slice(
    (currentTemplatePage - 1) * TEMPLATE_PAGE_SIZE,
    currentTemplatePage * TEMPLATE_PAGE_SIZE
  );

  return (
    <div className="space-y-5">
      <PageHeader
        eyebrow="Models"
        title="Architectures"
        subtitle={
          project?.name
            ? `Visual model graphs in ${project.name}`
            : "Compose a model as a graph, then train and export it"
        }
        icon={<Network size={20} />}
        actions={
          <>
            <input
              ref={fileInput}
              type="file"
              accept="application/json,.json"
              className="visually-hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) importMutation.mutate(file);
                event.target.value = "";
              }}
            />
            <Button
              variant="secondary"
              onClick={() => fileInput.current?.click()}
              disabled={importMutation.isPending}
            >
              <Upload size={16} aria-hidden /> Import
            </Button>
            <Button onClick={() => setCreating(true)}>
              <Plus size={16} aria-hidden /> New architecture
            </Button>
          </>
        }
      />

      <ModelsTabs active="architectures" />

      <MutationError mutations={[createMutation, importMutation, duplicateMutation, deleteMutation]} />

      {creating && (
        <section className="panel arch-template-picker">
          <div className="arch-template-head">
            <div className="arch-template-head-title">
              <p className="panel-title">Start from</p>
              {allTemplates.length > 0 && (
                <p className="arch-list-count">
                  {templates.length === allTemplates.length
                    ? `${allTemplates.length} preset${allTemplates.length === 1 ? "" : "s"}`
                    : `${templates.length} of ${allTemplates.length}`}
                </p>
              )}
            </div>
            <div className="arch-template-head-actions">
              <div className="arch-search arch-search-compact">
                <Search size={15} aria-hidden />
                <input
                  type="search"
                  value={templateSearch}
                  placeholder="Search presets"
                  aria-label="Search presets"
                  onChange={(event) => {
                    setTemplateSearch(event.target.value);
                    setTemplatePage(1);
                  }}
                />
              </div>
              <Button variant="ghost" size="sm" onClick={() => setCreating(false)}>
                Cancel
              </Button>
            </div>
          </div>

          {/* A preset already knows its task, so choosing one is the
              auto-detecting path. This picker is what a blank canvas needs —
              the task decides which nodes the palette offers and which training
              runner the Train button can reach — and it doubles as the preset
              filter. */}
          <div className="arch-template-task">
            <Field label="Task" hint="Sets the blank canvas and sorts presets">
              <TaskSelect
                value={selectedTask}
                options={studioTasks}
                onChange={(next) => {
                  setTask(next);
                  setTemplatePage(1);
                }}
              />
            </Field>
            <Button
              variant="ghost"
              size="sm"
              aria-pressed={onlyTask}
              onClick={() => {
                setOnlyTask((value) => !value);
                setTemplatePage(1);
              }}
            >
              {onlyTask ? "Show every preset" : `Only ${formatDatasetTask(selectedTask)}`}
            </Button>
          </div>

          <div className="arch-template-grid">
            <button
              type="button"
              className="arch-template-card arch-template-blank"
              disabled={createMutation.isPending}
              onClick={() =>
                createMutation.mutate({ name: "Untitled architecture", taskType: selectedTask })
              }
            >
              <p className="arch-template-name">Blank canvas</p>
              <p className="arch-template-desc">
                Start with nothing and wire it up from the palette.
              </p>
              <Badge tone="neutral">{formatDatasetTask(selectedTask)}</Badge>
            </button>
            {templatePageItems.map((template) => (
              <button
                key={template.id}
                type="button"
                className="arch-template-card"
                disabled={createMutation.isPending}
                onClick={() =>
                  createMutation.mutate({
                    name: template.name,
                    templateId: template.id,
                    taskType: template.task_type as TaskType
                  })
                }
              >
                <p className="arch-template-name">{template.name}</p>
                <p className="arch-template-desc">{template.description}</p>
                <Badge tone="neutral">{formatDatasetTask(template.task_type)}</Badge>
              </button>
            ))}
          </div>
          {templatePageCount > 1 && (
            <nav className="arch-pagination arch-template-pagination" aria-label="Preset pages">
              <Button
                variant="secondary"
                size="sm"
                disabled={currentTemplatePage <= 1}
                onClick={() => setTemplatePage(currentTemplatePage - 1)}
              >
                <ChevronLeft size={15} aria-hidden /> Previous
              </Button>
              <p className="arch-pagination-status">
                Page {currentTemplatePage} of {templatePageCount}
              </p>
              <Button
                variant="secondary"
                size="sm"
                disabled={currentTemplatePage >= templatePageCount}
                onClick={() => setTemplatePage(currentTemplatePage + 1)}
              >
                Next <ChevronRight size={15} aria-hidden />
              </Button>
            </nav>
          )}
        </section>
      )}

      {all.length > 0 && (
        <div className="arch-list-controls">
          <div className="arch-search">
            <Search size={15} aria-hidden />
            <input
              type="search"
              value={search}
              placeholder="Search architectures"
              aria-label="Search architectures"
              onChange={(event) => {
                setSearch(event.target.value);
                setPage(1);
              }}
            />
          </div>
          <p className="arch-list-count">
            {matches.length === all.length
              ? `${all.length} architecture${all.length === 1 ? "" : "s"}`
              : `${matches.length} of ${all.length}`}
          </p>
        </div>
      )}

      {architecturesQuery.isLoading ? (
        <CardGridSkeleton count={3} />
      ) : architectures.length === 0 ? (
        <EmptyState
          centered
          icon={<Network size={28} />}
          label="No architectures yet"
          description="Build a model visually — drag layers onto a canvas, wire them together, then train it like any other model."
          action={<Button onClick={() => setCreating(true)}>New architecture</Button>}
        />
      ) : matches.length === 0 ? (
        <EmptyState
          centered
          icon={<Search size={28} />}
          label={`No architecture matches “${search}”`}
          description="Try a different name, or clear the search to see everything."
          action={
            <Button variant="secondary" onClick={() => setSearch("")}>
              Clear search
            </Button>
          }
        />
      ) : (
        <div className="card-grid">
          {architectures.map((architecture) => (
            <article className="panel arch-card" key={architecture.id}>
              <div className="arch-card-head">
                <Link
                  className="arch-card-name"
                  href={`/models/architectures/${architecture.id}`}
                >
                  {architecture.name}
                </Link>
                <Badge tone="neutral">{formatDatasetTask(architecture.task_type)}</Badge>
              </div>
              {architecture.description && (
                <p className="arch-card-desc">{architecture.description}</p>
              )}
              <dl className="arch-card-meta">
                <div>
                  <dt>Nodes</dt>
                  <dd>{architecture.node_count}</dd>
                </div>
                <div>
                  <dt>Version</dt>
                  <dd>v{architecture.version}</dd>
                </div>
                <div>
                  <dt>Updated</dt>
                  <dd>{new Date(architecture.updated_at).toLocaleDateString()}</dd>
                </div>
              </dl>
              <div className="arch-card-actions">
                <ButtonLink
                  variant="secondary"
                  size="sm"
                  href={`/models/architectures/${architecture.id}`}
                >
                  Open studio
                </ButtonLink>
                <IconButton
                  aria-label={`Duplicate ${architecture.name}`}
                  onClick={() => duplicateMutation.mutate(architecture.id)}
                >
                  <Copy size={16} />
                </IconButton>
                <IconButton
                  danger
                  aria-label={`Delete ${architecture.name}`}
                  onClick={() =>
                    confirm({
                      title: "Delete architecture?",
                      message: `This removes “${architecture.name}” and its saved versions. Models already trained from it keep working.`,
                      confirmLabel: "Delete architecture",
                      onConfirm: () => deleteMutation.mutate(architecture.id)
                    })
                  }
                >
                  <Trash2 size={16} />
                </IconButton>
              </div>
            </article>
          ))}
        </div>
      )}

      {pageCount > 1 && (
        <nav className="arch-pagination" aria-label="Architecture pages">
          <Button
            variant="secondary"
            size="sm"
            disabled={currentPage <= 1}
            onClick={() => setPage(currentPage - 1)}
          >
            <ChevronLeft size={15} aria-hidden /> Previous
          </Button>
          <p className="arch-pagination-status">
            Page {currentPage} of {pageCount}
          </p>
          <Button
            variant="secondary"
            size="sm"
            disabled={currentPage >= pageCount}
            onClick={() => setPage(currentPage + 1)}
          >
            Next <ChevronRight size={15} aria-hidden />
          </Button>
        </nav>
      )}

      {confirmationDialog}
    </div>
  );
}
