import type { DeleteResponse, ProjectSummary, TaskType } from "@/types/api";
import { jsonFetch } from "@/lib/api/client";

export const projectsApi = {
  projects: () => jsonFetch<ProjectSummary[]>("/projects"),
  createProject: (payload: {
    name: string;
    description?: string | null;
    task_types?: TaskType[];
    metadata?: Record<string, any>;
  }) =>
    jsonFetch<ProjectSummary>("/projects", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  updateProject: (projectId: string, payload: Partial<ProjectSummary>) =>
    jsonFetch<ProjectSummary>(`/projects/${projectId}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    }),
  deleteProject: (projectId: string) =>
    jsonFetch<DeleteResponse>(`/projects/${projectId}`, {
      method: "DELETE"
    })
};
