import type { TaskType } from "@/types/api";
import type { components } from "@/types/generated/api";
import { jsonFetch, jsonFetchChecked } from "@/lib/api/client";
import { projectListSchema } from "@/lib/api/schemas";

type ProjectSummary = components["schemas"]["ProjectSummary"];
type ProjectStats = components["schemas"]["ProjectStats"];
type DeleteResponse = components["schemas"]["DeleteResponse"];

export const projectsApi = {
  projects: () => jsonFetchChecked<ProjectSummary[]>("/projects", projectListSchema),
  projectStats: (projectId: string) => jsonFetch<ProjectStats>(`/projects/${projectId}/stats`),
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
