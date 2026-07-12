"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowLeft, BarChart3, Database, FileImage, FilePlus2 } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { formatDatasetTask, taskDescription } from "@/features/platform/utils";
import { Field, MutationError } from "@/features/platform/ui";
import type { TaskType } from "@/types/api";

const IMAGE_TASK_TYPES: TaskType[] = ["classification", "object_detection", "segmentation"];
const PROJECT_DESCRIPTION_LIMIT = 160;

export function ProjectCreatePage() {
  const router = useRouter();
  const { setProjectId, refreshProjects } = useProject();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [selectedTasks, setSelectedTasks] = useState<TaskType[]>(IMAGE_TASK_TYPES);
  const trimmedName = name.trim();
  const trimmedDescription = description.trim();
  const canCreateProject = Boolean(trimmedName && trimmedDescription && selectedTasks.length > 0);
  const createProject = useMutation({
    mutationFn: api.createProject,
    onSuccess: async (project) => {
      setProjectId(project.id);
      await refreshProjects();
      router.push("/datasets");
    }
  });

  function toggleTaskCard(task: TaskType) {
    setSelectedTasks((tasks) => {
      if (tasks.includes(task)) {
        const next = tasks.filter((item) => item !== task);
        return next.length ? next : tasks;
      }
      return [...tasks, task];
    });
  }

  function submitProject() {
    if (!canCreateProject) return;
    createProject.mutate({
      name: trimmedName,
      description: trimmedDescription,
      task_types: selectedTasks,
      metadata: { domain: "image" }
    });
  }

  function taskIcon(task: TaskType) {
    if (task === "classification") return <FileImage size={20} />;
    if (task === "object_detection") return <Database size={20} />;
    return <BarChart3 size={20} />;
  }

  return (
    <div className="create-project-page">
      <section className="create-project-hero">
        <Link className="project-back-link" href="/">
          <ArrowLeft size={16} />
          <span>Projects</span>
        </Link>
        <div className="create-project-hero-copy">
          <span className="brand-kicker">Onestep AI Platform</span>
          <h2>Create an image intelligence workspace</h2>
          <p>Shape a focused project for datasets, annotations, training runs, model tests, and inspection workflows.</p>
        </div>
      </section>
      <section className="panel create-project-panel">
        <div className="create-project-form">
          <Field label="Project name">
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Example: Dental Radiograph Workspace" />
          </Field>
          <Field label="Short description">
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Example: Dental X-ray segmentation experiments for YOLO and U-Net models"
              maxLength={PROJECT_DESCRIPTION_LIMIT}
              rows={3}
            />
            <span className="field-hint">{description.length}/{PROJECT_DESCRIPTION_LIMIT}</span>
          </Field>
        </div>
        <div className="field">
          <label>Project type</label>
          <div className="task-choice-grid task-choice-grid-premium">
            {IMAGE_TASK_TYPES.map((task) => {
              const active = selectedTasks.includes(task);
              return (
                <button
                  type="button"
                  aria-pressed={active}
                  className={`task-choice task-choice-premium task-choice-${task.replaceAll("_", "-")} ${
                    active ? "task-choice-active" : ""
                  }`}
                  key={task}
                  onClick={() => toggleTaskCard(task)}
                >
                  <span className="task-choice-icon">{taskIcon(task)}</span>
                  <span className="task-choice-copy">
                    <strong>{formatDatasetTask(task)}</strong>
                    <small>{taskDescription(task)}</small>
                    <em>{active ? "Included" : "Add type"}</em>
                  </span>
                </button>
              );
            })}
          </div>
        </div>
        <div className="create-project-actions">
          <button className="primary-button" onClick={submitProject} disabled={createProject.isPending || !canCreateProject}>
            <FilePlus2 size={16} /> Create Project
          </button>
        </div>
        <MutationError mutations={[createProject]} />
      </section>
    </div>
  );
}

