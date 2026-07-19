"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowLeft, FilePlus2 } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useProject } from "@/components/app-shell";
import { Button, Field, MutationError, PageHeader } from "@/features/platform/ui";
import { toast } from "@/features/platform/toast";
import { TaskTypePicker, domainForTasks } from "./task-type-picker";
import type { TaskType } from "@/types/api";

const PROJECT_DESCRIPTION_LIMIT = 160;

export function ProjectCreatePage() {
  const router = useRouter();
  const { setProjectId, refreshProjects } = useProject();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
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

  function submitProject() {
    if (!canCreateProject) return;
    createProject.mutate({
      name: trimmedName,
      description: trimmedDescription,
      task_types: selectedTasks,
      metadata: { domain: domainForTasks(selectedTasks) }
    });
  }

  return (
    <div className="create-project-page">
      <div className="create-project-heading">
        <Link className="project-back-link" href="/projects">
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
          {/* No preselected domain: the workspace type is a deliberate choice, and
              defaulting to vision quietly created vision projects for NLP users. */}
          <TaskTypePicker value={selectedTasks} onChange={setSelectedTasks} initialDomain={null} />
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
