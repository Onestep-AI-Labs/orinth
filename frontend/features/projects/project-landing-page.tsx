"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Database, FilePlus2, FolderOpen, MoreVertical, Save, Trash2, X } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { formatDatasetTask } from "@/features/platform/utils";
import { CardGridSkeleton, EmptyState, Field, MutationError, PageHeader, PanelTitle, useConfirmationDialog } from "@/features/platform/ui";
import type { ProjectSummary } from "@/types/api";

const DEFAULT_PROJECT_ID = "default-research-project";

export function ProjectLandingPage() {
  const router = useRouter();
  const { projectId, projects, projectsLoading, setProjectId, refreshProjects } = useProject();
  const [openMenuId, setOpenMenuId] = useState("");
  const [renameProjectId, setRenameProjectId] = useState("");
  const [renameProjectDraft, setRenameProjectDraft] = useState("");
  const { confirm, confirmationDialog } = useConfirmationDialog();
  const updateProject = useMutation({
    mutationFn: ({ targetProjectId, name }: { targetProjectId: string; name: string }) =>
      api.updateProject(targetProjectId, { name }),
    onSuccess: async () => {
      setOpenMenuId("");
      setRenameProjectId("");
      setRenameProjectDraft("");
      await refreshProjects();
    }
  });
  const deleteProject = useMutation({
    mutationFn: api.deleteProject,
    onSuccess: async (_result, deletedId) => {
      if (deletedId === projectId) setProjectId(DEFAULT_PROJECT_ID);
      setOpenMenuId("");
      setRenameProjectId("");
      await refreshProjects();
    }
  });

  function openProject(projectId: string) {
    setProjectId(projectId);
    router.push("/datasets");
  }

  function openProjectRename(project: ProjectSummary) {
    setRenameProjectId(project.id);
    setRenameProjectDraft(project.name);
    setOpenMenuId("");
  }

  function saveProjectRename() {
    if (!renameProjectId || !renameProjectDraft.trim()) return;
    updateProject.mutate({ targetProjectId: renameProjectId, name: renameProjectDraft.trim() });
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
          <Link className="primary-button" href="/projects/new">
            <FilePlus2 size={16} /> New Project
          </Link>
        </div>
        {projectsLoading ? (
          <CardGridSkeleton count={3} />
        ) : (
          <div className="project-grid">
            {projects.map((project) => (
              <article className="project-card" key={project.id}>
                <button className="project-card-main" onClick={() => openProject(project.id)} type="button">
                  <div className="project-card-title">
                    <strong title={project.name}>{project.name}</strong>
                    <span>{project.id === DEFAULT_PROJECT_ID ? "Default workspace" : "Project workspace"}</span>
                  </div>
                  <p className="project-card-description">
                    {project.description ?? "Local image workspace"}
                  </p>
                  <div className="label-chip-row tag-row-compact">
                    {project.task_types.map((task) => (
                      <span className="label-chip" key={task}>{formatDatasetTask(task)}</span>
                    ))}
                  </div>
                </button>
                <div className="card-menu">
                  <button
                    className="icon-button"
                    onClick={() => setOpenMenuId((value) => (value === project.id ? "" : project.id))}
                    title="Project options"
                    type="button"
                    aria-expanded={openMenuId === project.id}
                  >
                    <MoreVertical size={16} />
                  </button>
                  {openMenuId === project.id && (
                    <div className="option-menu" role="menu">
                      <button type="button" onClick={() => openProject(project.id)}>
                        <FolderOpen size={15} /> Open
                      </button>
                      <button type="button" onClick={() => openProjectRename(project)}>
                        <Save size={15} /> Rename
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
                {renameProjectId === project.id && (
                  <div className="card-rename-popover">
                    <Field label="Project name">
                      <input
                        value={renameProjectDraft}
                        onChange={(event) => setRenameProjectDraft(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") saveProjectRename();
                          if (event.key === "Escape") setRenameProjectId("");
                        }}
                        autoFocus
                      />
                    </Field>
                    <div className="card-rename-actions">
                      <button className="primary-button" onClick={saveProjectRename} disabled={!renameProjectDraft.trim() || updateProject.isPending}>
                        <Save size={16} /> Save
                      </button>
                      <button className="secondary-button" onClick={() => setRenameProjectId("")}>
                        <X size={16} /> Cancel
                      </button>
                    </div>
                  </div>
                )}
              </article>
            ))}
            {projects.length === 0 && <EmptyState label="No projects" />}
          </div>
        )}
        <MutationError mutations={[updateProject, deleteProject]} />
      </section>
      {confirmationDialog}
    </div>
  );
}

