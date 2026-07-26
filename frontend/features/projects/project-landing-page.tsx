"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import {
  Archive,
  ArchiveRestore,
  Database,
  FilePlus2,
  FolderOpen,
  MoreVertical,
  Settings2,
  Trash2
} from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { formatDatasetTask } from "@/features/platform/utils";
import {
  Badge,
  ButtonLink,
  CardGridSkeleton,
  EmptyState,
  IconButton,
  MutationError,
  PageHeader,
  PanelTitle,
  useConfirmationDialog
} from "@/features/platform/ui";
import type { ProjectSummary } from "@/types/api";

const DEFAULT_PROJECT_ID = "default-research-project";

export function ProjectLandingPage() {
  const router = useRouter();
  const { projectId, projects, projectsLoading, setProjectId, refreshProjects } = useProject();
  const [openMenuId, setOpenMenuId] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const updateProject = useMutation({
    mutationFn: ({ targetProjectId, archived }: { targetProjectId: string; archived: boolean }) =>
      api.updateProject(targetProjectId, { archived }),
    onSuccess: async (_result, variables) => {
      // Archiving the active project would leave the switcher pointing at a
      // workspace it no longer lists, so fall back to the default.
      if (variables.archived && variables.targetProjectId === projectId) {
        setProjectId(DEFAULT_PROJECT_ID);
      }
      setOpenMenuId("");
      await refreshProjects();
    }
  });
  const deleteProject = useMutation({
    mutationFn: api.deleteProject,
    onSuccess: async (_result, deletedId) => {
      if (deletedId === projectId) setProjectId(DEFAULT_PROJECT_ID);
      setOpenMenuId("");
      await refreshProjects();
    }
  });

  const archivedCount = useMemo(
    () => projects.filter((project) => project.archived).length,
    [projects]
  );
  const visibleProjects = useMemo(
    () => (showArchived ? projects : projects.filter((project) => !project.archived)),
    [projects, showArchived]
  );

  function openProject(nextProjectId: string) {
    setProjectId(nextProjectId);
    router.push("/datasets");
  }

  function confirmDeleteProject(project: ProjectSummary) {
    confirm({
      title: "Delete project?",
      message: `This will delete "${project.name}" if it has no owned datasets or history. This action cannot be undone.`,
      confirmLabel: "Delete project",
      onConfirm: () => deleteProject.mutate(project.id)
    });
  }

  return (
    <div className="space-y-5">
      <PageHeader title="Projects" subtitle="Choose a workspace" icon={<Database size={20} />} />
      <section className="panel">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <PanelTitle icon={<Database size={18} />} title="Project List" />
          <div className="action-row">
            {archivedCount > 0 && (
              <label className="check-row">
                <input
                  type="checkbox"
                  checked={showArchived}
                  onChange={(event) => setShowArchived(event.target.checked)}
                />
                <span>Show archived ({archivedCount})</span>
              </label>
            )}
            <ButtonLink href="/projects/new" data-tour="projects-new">
              <FilePlus2 size={16} /> New Project
            </ButtonLink>
          </div>
        </div>
        {projectsLoading ? (
          <CardGridSkeleton count={3} />
        ) : (
          <div className="project-grid">
            {visibleProjects.map((project) => (
              <article className="project-card" key={project.id} data-tour="projects-grid">
                <button className="project-card-main" onClick={() => openProject(project.id)} type="button">
                  <div className="project-card-title">
                    <strong title={project.name}>{project.name}</strong>
                    <span>{project.id === DEFAULT_PROJECT_ID ? "Default workspace" : "Project workspace"}</span>
                  </div>
                  <p className="project-card-description">
                    {project.description ?? "Local AI workspace"}
                  </p>
                  <div className="label-chip-row tag-row-compact">
                    {project.archived ? <Badge tone="warn">Archived</Badge> : null}
                    {project.task_types.map((task) => (
                      <span className="label-chip" key={task}>{formatDatasetTask(task)}</span>
                    ))}
                  </div>
                </button>
                <div className="card-menu">
                  <IconButton
                    aria-label="Project options"
                    onClick={() => setOpenMenuId((value) => (value === project.id ? "" : project.id))}
                    aria-expanded={openMenuId === project.id}
                  >
                    <MoreVertical size={16} />
                  </IconButton>
                  {openMenuId === project.id && (
                    <div className="option-menu" role="menu">
                      <button type="button" onClick={() => openProject(project.id)}>
                        <FolderOpen size={15} /> Open
                      </button>
                      <Link href={`/projects/${project.id}/settings`} onClick={() => setOpenMenuId("")}>
                        <Settings2 size={15} /> Settings
                      </Link>
                      <button
                        type="button"
                        onClick={() =>
                          updateProject.mutate({
                            targetProjectId: project.id,
                            archived: !project.archived
                          })
                        }
                        disabled={project.id === DEFAULT_PROJECT_ID || updateProject.isPending}
                      >
                        {project.archived ? <ArchiveRestore size={15} /> : <Archive size={15} />}
                        {project.archived ? "Restore" : "Archive"}
                      </button>
                      <button
                        className="danger-menu-item"
                        type="button"
                        onClick={() => confirmDeleteProject(project)}
                        disabled={project.id === DEFAULT_PROJECT_ID || deleteProject.isPending}
                      >
                        <Trash2 size={15} /> Delete
                      </button>
                    </div>
                  )}
                </div>
              </article>
            ))}
            {visibleProjects.length === 0 && (
              <EmptyState
                centered
                icon={archivedCount > 0 ? <Archive size={32} /> : <FolderOpen size={32} />}
                label={archivedCount > 0 ? "No active projects" : "No projects yet"}
                description={
                  archivedCount > 0
                    ? "Every project is archived. Show archived to restore one, or create a new project."
                    : "A project holds your datasets, training runs, and results."
                }
                action={
                  <ButtonLink href="/projects/new">
                    <FilePlus2 size={16} /> Create your first project
                  </ButtonLink>
                }
              />
            )}
          </div>
        )}
        <MutationError mutations={[updateProject, deleteProject]} />
      </section>
      {confirmationDialog}
    </div>
  );
}
