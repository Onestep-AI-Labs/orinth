"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Archive,
  ArchiveRestore,
  ArrowLeft,
  Boxes,
  Database,
  FolderOpen,
  Info,
  Save,
  Settings2,
  Trash2,
  Wand2
} from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { ALL_TASK_TYPES } from "@/features/platform/utils";
import { WORKSPACE_HREF } from "@/features/auth/routes";
import { TaskTypePicker, domainForTasks, initialDomainForTasks } from "./task-type-picker";
import {
  Badge,
  Button,
  ButtonLink,
  EmptyState,
  Field,
  InlineSpinner,
  Metric,
  MutationError,
  PageHeader,
  PanelTitle,
  useConfirmationDialog
} from "@/features/platform/ui";
import type { TaskType } from "@/types/api";

const DEFAULT_PROJECT_ID = "default-research-project";
const DESCRIPTION_LIMIT = 500;
/**
 * Reserved metadata keys the form must not round-trip verbatim: `archived` is
 * service-owned, and `domain` is derived from the task selection on every save
 * so it cannot go stale when a project's task types change domain.
 */
const SERVICE_OWNED_METADATA = ["archived", "domain"];
const PREP_MODEL_KEY = "default_openrouter_model";

export function ProjectSettingsPage({ projectId }: { projectId: string }) {
  const router = useRouter();
  const {
    projectId: activeProjectId,
    projects,
    projectsLoading,
    setProjectId,
    refreshProjects
  } = useProject();
  const project = projects.find((item) => item.id === projectId) ?? null;
  const isDefaultProject = projectId === DEFAULT_PROJECT_ID;
  const { confirm, confirmationDialog } = useConfirmationDialog();

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tasks, setTasks] = useState<TaskType[]>([]);
  const [prepModel, setPrepModel] = useState("");

  // Reseed whenever the stored project changes — on first load, and after a save
  // refreshes the shell's copy — so the form never shows a stale draft.
  useEffect(() => {
    if (!project) return;
    setName(project.name);
    setDescription(project.description ?? "");
    setTasks(project.task_types);
    setPrepModel(String(project.metadata?.[PREP_MODEL_KEY] ?? ""));
  }, [project?.id, project?.updated_at]);

  // A deep link to /projects/{id}/settings must make that project the active one,
  // or the sidebar names one project while this form edits another.
  //
  // Adopt the route exactly once per project id. The switcher sets the active
  // project and *then* navigates, so there is a render where the URL still names
  // the old project while the context names the new one. Re-running here would
  // read that intermediate state as a deep link and pull the user back.
  const adoptedProjectId = useRef<string | null>(null);
  useEffect(() => {
    if (!project || adoptedProjectId.current === projectId) return;
    adoptedProjectId.current = projectId;
    if (project.id !== activeProjectId) setProjectId(project.id);
  }, [project, projectId, activeProjectId, setProjectId]);

  const statsQuery = useQuery({
    queryKey: ["project-stats", projectId],
    queryFn: () => api.projectStats(projectId),
    enabled: Boolean(project)
  });
  const catalogQuery = useQuery({
    queryKey: ["dataset-catalog", projectId],
    queryFn: () => api.datasetCatalog(projectId),
    enabled: Boolean(project)
  });

  const updateProject = useMutation({
    mutationFn: (payload: Parameters<typeof api.updateProject>[1]) =>
      api.updateProject(projectId, payload),
    onSuccess: async () => {
      await refreshProjects();
      await statsQuery.refetch();
    }
  });
  const deleteProject = useMutation({
    mutationFn: () => api.deleteProject(projectId),
    onSuccess: async () => {
      setProjectId(DEFAULT_PROJECT_ID);
      await refreshProjects();
      router.push(WORKSPACE_HREF);
    }
  });

  const trimmedName = name.trim();
  const orderedTasks = useMemo(
    () => ALL_TASK_TYPES.filter((task) => tasks.includes(task)),
    [tasks]
  );
  const dirty =
    Boolean(project) &&
    (trimmedName !== project!.name ||
      description.trim() !== (project!.description ?? "") ||
      orderedTasks.join() !== project!.task_types.join() ||
      prepModel.trim() !== String(project!.metadata?.[PREP_MODEL_KEY] ?? ""));
  const canSave = dirty && trimmedName.length > 0 && orderedTasks.length > 0;

  // Datasets owned by this project whose task type is being dropped. Removing the
  // type does not delete them, but the project can no longer create more of them.
  const strandedDatasets = useMemo(() => {
    const removed = (project?.task_types ?? []).filter((task) => !orderedTasks.includes(task));
    if (!removed.length) return [];
    return (catalogQuery.data ?? []).filter(
      (dataset) => dataset.editable && removed.includes(dataset.task_type)
    );
  }, [project?.task_types, orderedTasks, catalogQuery.data]);

  function save() {
    if (!canSave || !project) return;
    const metadata = Object.fromEntries(
      Object.entries(project.metadata ?? {}).filter(([key]) => !SERVICE_OWNED_METADATA.includes(key))
    );
    const trimmedModel = prepModel.trim();
    if (trimmedModel) metadata[PREP_MODEL_KEY] = trimmedModel;
    else delete metadata[PREP_MODEL_KEY];
    // Recomputed, never round-tripped: moving a project from vision to NLP task
    // types must not leave metadata.domain claiming the domain it just left.
    metadata.domain = domainForTasks(orderedTasks);

    const payload = {
      name: trimmedName,
      description: description.trim(),
      task_types: orderedTasks,
      metadata
    };

    if (strandedDatasets.length) {
      confirm({
        tone: "warning",
        title: "Remove task types in use?",
        message: `${strandedDatasets.length} dataset${strandedDatasets.length === 1 ? "" : "s"} in this project ${strandedDatasets.length === 1 ? "uses" : "use"} a task type you are removing. Existing datasets are kept, but this project can no longer create new ones of that type.`,
        confirmLabel: "Save changes",
        onConfirm: () => updateProject.mutate(payload)
      });
      return;
    }
    updateProject.mutate(payload);
  }

  /** Return the form to the stored project — the same seeding the effect does. */
  function reset() {
    if (!project) return;
    setName(project.name);
    setDescription(project.description ?? "");
    setTasks(project.task_types);
    setPrepModel(String(project.metadata?.[PREP_MODEL_KEY] ?? ""));
  }

  function toggleArchived() {
    updateProject.mutate({ archived: !project?.archived });
  }

  function confirmDelete() {
    confirm({
      title: "Delete project?",
      message: `This permanently deletes "${project?.name}". This action cannot be undone.`,
      confirmLabel: "Delete project",
      onConfirm: () => deleteProject.mutate()
    });
  }

  if (!project) {
    return (
      <div className="space-y-5">
        <PageHeader title="Project settings" subtitle="Workspace configuration" icon={<Settings2 size={20} />} />
        <section className="panel">
          {projectsLoading ? (
            <InlineSpinner label="Loading project" />
          ) : (
            <EmptyState
              centered
              icon={<FolderOpen size={32} />}
              label="Project not found"
              description="This project may have been deleted, or the link may be out of date."
              action={
                <ButtonLink href={WORKSPACE_HREF}>
                  <ArrowLeft size={16} /> Back to projects
                </ButtonLink>
              }
            />
          )}
        </section>
      </div>
    );
  }

  const stats = statsQuery.data;
  const blockers = stats?.blockers ?? [];
  const deleteBlocked = !stats || !stats.deletable;

  return (
    <div className="space-y-5">
      {/* No back link here — the project sidebar owns the route back to the list. */}
      <PageHeader
        title={project.name}
        subtitle="Workspace name, task types, and lifecycle"
        icon={<Settings2 size={20} />}
        eyebrow="Project settings"
        actions={project.archived ? <Badge tone="warn">Archived</Badge> : undefined}
      />

      <div className="settings-layout">
        <div className="settings-column">
          <section className="panel">
            <PanelTitle icon={<Settings2 size={18} />} title="General" />
            <div className="create-project-form">
              <Field label="Project name">
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Name this workspace"
                />
              </Field>
              <Field label="Short description" hint={`${description.length}/${DESCRIPTION_LIMIT}`}>
                <textarea
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="What this workspace is for"
                  maxLength={DESCRIPTION_LIMIT}
                  rows={3}
                />
              </Field>
            </div>
          </section>

          <section className="panel">
            <PanelTitle icon={<Boxes size={18} />} title="Task types" />
            <p className="field-hint">
              Task types decide which datasets and bundled samples this project can use. Both
              domains stay available — add an NLP type to a vision project to start working in it.
            </p>
            <div className="field mt-4">
              <TaskTypePicker
                value={orderedTasks}
                onChange={setTasks}
                initialDomain={initialDomainForTasks(project.task_types)}
                resetKey={project.id}
              />
            </div>
            {strandedDatasets.length ? (
              <p className="field-hint mt-3">
                {strandedDatasets.length} existing dataset
                {strandedDatasets.length === 1 ? "" : "s"} use a task type you are removing. They are
                kept, but this project can no longer create more of that type.
              </p>
            ) : null}
            {orderedTasks.length === 0 ? (
              <p className="field-hint mt-3">Select at least one task type to save.</p>
            ) : null}
          </section>

          <section className="panel">
            <PanelTitle icon={<Wand2 size={18} />} title="Auto data prep" />
            <div>
              <Field label="Default model">
                <input
                  value={prepModel}
                  onChange={(event) => setPrepModel(event.target.value)}
                  placeholder="Provider model id"
                  autoComplete="off"
                />
              </Field>
              <p className="field-hint mt-2">
                Saved with the project and used as the default when auto data prep ships. It has no
                effect yet.
              </p>
            </div>
          </section>

          <section className="panel danger-zone">
            <h2 className="danger-zone-head">
              <Trash2 size={18} /> Danger zone
            </h2>
            <div className="danger-zone-row">
              <div className="danger-zone-copy">
                <strong>{project.archived ? "Restore project" : "Archive project"}</strong>
                <p>
                  {isDefaultProject
                    ? "The default workspace cannot be archived — the switcher would have nothing to fall back to."
                    : project.archived
                      ? "Return this project to the workspace switcher and the project list."
                      : "Hide this project from the switcher and the project list. Nothing is deleted, and this is reversible."}
                </p>
              </div>
              <Button
                variant="secondary"
                onClick={toggleArchived}
                disabled={isDefaultProject || updateProject.isPending}
              >
                {project.archived ? <ArchiveRestore size={16} /> : <Archive size={16} />}
                {project.archived ? "Restore" : "Archive"}
              </Button>
            </div>
            <div className="danger-zone-row">
              <div className="danger-zone-copy">
                <strong>Delete project</strong>
                <p>
                  {isDefaultProject
                    ? "The default workspace cannot be deleted."
                    : statsQuery.isError
                      ? "Project totals could not be loaded, so deletion stays disabled until they are known."
                      : blockers.length
                        ? `This project still has ${blockers.join(", ")}. Delete those first.`
                        : "Permanently delete this project. This cannot be undone."}
                </p>
              </div>
              <Button
                variant="danger"
                onClick={confirmDelete}
                disabled={isDefaultProject || deleteBlocked || deleteProject.isPending}
              >
                <Trash2 size={16} /> Delete
              </Button>
            </div>
            <div className="danger-zone-error">
              <MutationError mutations={[deleteProject]} />
            </div>
          </section>
        </div>

        <div className="settings-column settings-aside">
          <section className="panel">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <PanelTitle icon={<Database size={18} />} title="Contents" />
              {statsQuery.isFetching ? <InlineSpinner label="Refreshing" /> : null}
            </div>
            <div className="metric-grid">
              <Metric label="Datasets" value={stats?.datasets ?? "—"} />
              <Metric label="Training runs" value={stats?.training_jobs ?? "—"} />
              <Metric label="Test runs" value={stats?.evaluation_jobs ?? "—"} />
              <Metric label="Inference results" value={stats?.inference_runs ?? "—"} />
            </div>
            <div className="action-row mt-4">
              <ButtonLink variant="secondary" href="/datasets">
                <Database size={16} /> Open datasets
              </ButtonLink>
            </div>
          </section>

          <section className="panel">
            <PanelTitle icon={<Info size={18} />} title="Identity" />
            <dl className="settings-identity">
              <div>
                <dt>Project id</dt>
                <dd className="settings-identity-id">{project.id}</dd>
              </div>
              <div>
                <dt>Created</dt>
                <dd>{new Date(project.created_at).toLocaleString()}</dd>
              </div>
              <div>
                <dt>Updated</dt>
                <dd>{new Date(project.updated_at).toLocaleString()}</dd>
              </div>
            </dl>
          </section>
        </div>
      </div>

      {/* Follows the user down a long form. Always present rather than appearing
          on first edit: Save is a fixed target you can find before you touch
          anything, and the spec's contract is that it is disabled when clean,
          not absent. */}
      <div className="settings-savebar">
        <span className="settings-savebar-note">
          {!dirty
            ? "No unsaved changes"
            : canSave
              ? "Unsaved changes"
              : "A name and at least one task type are required"}
        </span>
        <div className="settings-savebar-actions">
          <Button variant="secondary" onClick={reset} disabled={!dirty || updateProject.isPending}>
            Discard
          </Button>
          <Button onClick={save} disabled={!canSave || updateProject.isPending}>
            <Save size={16} /> Save changes
          </Button>
        </div>
      </div>
      <MutationError mutations={[updateProject]} />
      {confirmationDialog}
    </div>
  );
}
