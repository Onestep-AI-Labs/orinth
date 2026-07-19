"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowLeft, BarChart3, Database, FileImage, FilePlus2, FileText, MessageSquareText } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { NLP_TASK_TYPES, VISION_TASK_TYPES, formatDatasetTask, taskDescription } from "@/features/platform/utils";
import { Badge, Button, Field, MutationError, PageHeader } from "@/features/platform/ui";
import { toast } from "@/features/platform/toast";
import type { TaskType } from "@/types/api";

type ProjectDomain = "vision" | "nlp";
const PROJECT_DESCRIPTION_LIMIT = 160;

export function ProjectCreatePage() {
  const router = useRouter();
  const { setProjectId, refreshProjects } = useProject();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  // No preselected domain or tasks: the workspace type is a deliberate choice,
  // and defaulting to vision quietly created vision projects for NLP users.
  const [domain, setDomain] = useState<ProjectDomain | null>(null);
  const [selectedTasks, setSelectedTasks] = useState<TaskType[]>([]);
  const trimmedName = name.trim();
  const trimmedDescription = description.trim();
  const canCreateProject = Boolean(trimmedName && trimmedDescription && selectedTasks.length > 0);
  const createProject = useMutation({
    mutationFn: api.createProject,
    onSuccess: async (project) => {
      setProjectId(project.id);
      await refreshProjects();
      toast.success(`Project "${project.name}" created`);
      router.push("/datasets");
    }
  });

  function toggleTaskCard(task: TaskType) {
    setSelectedTasks((tasks) =>
      tasks.includes(task) ? tasks.filter((item) => item !== task) : [...tasks, task]
    );
  }

  function submitProject() {
    if (!canCreateProject) return;
    createProject.mutate({
      name: trimmedName,
      description: trimmedDescription,
      task_types: selectedTasks,
      metadata: { domain: selectedDomain }
    });
  }

  function taskIcon(task: TaskType) {
    if (task === "classification") return <FileImage size={20} />;
    if (task === "object_detection") return <Database size={20} />;
    if (task === "text_classification") return <FileText size={20} />;
    if (task === "summarization") return <BarChart3 size={20} />;
    if (task === "question_answering") return <MessageSquareText size={20} />;
    return <BarChart3 size={20} />;
  }

  const domainTasks = domain === null ? [] : domain === "vision" ? VISION_TASK_TYPES : NLP_TASK_TYPES;
  const visionCount = selectedTasks.filter((task) => VISION_TASK_TYPES.includes(task)).length;
  const nlpCount = selectedTasks.filter((task) => NLP_TASK_TYPES.includes(task)).length;
  // Task types are the source of truth downstream; `domain` only records which
  // side the workspace leans to, so a cross-domain project reads as "mixed".
  const selectedDomain: ProjectDomain | "mixed" =
    visionCount > 0 && nlpCount > 0 ? "mixed" : nlpCount > 0 ? "nlp" : "vision";
  const orderedSelection = [...VISION_TASK_TYPES, ...NLP_TASK_TYPES].filter((task) =>
    selectedTasks.includes(task)
  );

  return (
    <div className="create-project-page">
      <div className="create-project-heading">
        <Link className="project-back-link" href="/">
          <ArrowLeft size={16} />
          <span>Projects</span>
        </Link>
        <PageHeader
          title="Create an AI workspace"
          subtitle="Datasets, annotations, training runs, model tests, and inspection workflows."
          icon={<FilePlus2 size={20} />}
        />
      </div>
      <section className="panel create-project-panel">
        <div className="create-project-form">
          <Field label="Project name">
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Name this workspace" />
          </Field>
          <Field label="Short description">
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="What this workspace is for"
              maxLength={PROJECT_DESCRIPTION_LIMIT}
              rows={3}
            />
            <span className="field-hint">{description.length}/{PROJECT_DESCRIPTION_LIMIT}</span>
          </Field>
        </div>
        <div className="field">
          <label>Project type</label>
          <div className="segmented-control mb-3">
            <button className={domain === "vision" ? "segmented-active" : ""} type="button" onClick={() => setDomain("vision")}>
              Vision{visionCount > 0 ? <span className="segmented-count">{visionCount}</span> : null}
            </button>
            <button className={domain === "nlp" ? "segmented-active" : ""} type="button" onClick={() => setDomain("nlp")}>
              NLP{nlpCount > 0 ? <span className="segmented-count">{nlpCount}</span> : null}
            </button>
          </div>
          {domain === null ? (
            <p className="hint-text">Choose Vision or NLP to see the task types available.</p>
          ) : (
          <div className="task-choice-grid task-choice-grid-premium">
            {domainTasks.map((task) => {
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
          )}
          {orderedSelection.length > 0 ? (
            <div className="task-selection-summary">
              <span className="task-selection-label">Included task types</span>
              <div className="task-selection-chips">
                {orderedSelection.map((task) => (
                  <Badge key={task} tone="neutral">
                    {formatDatasetTask(task)}
                  </Badge>
                ))}
              </div>
            </div>
          ) : null}
        </div>
        <div className="create-project-actions">
          <Button onClick={submitProject} disabled={createProject.isPending || !canCreateProject}>
            <FilePlus2 size={16} /> Create Project
          </Button>
        </div>
        <MutationError mutations={[createProject]} />
      </section>
    </div>
  );
}
