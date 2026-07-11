import type { DeleteResponse, ModelInfo, TaskType } from "@/types/api";
import { API_BASE, jsonFetch, query } from "@/lib/api/client";

export const modelsApi = {
  models: (availableOnly = false, projectId?: string, taskType?: TaskType) =>
    jsonFetch<ModelInfo[]>(
      `/models${query({
        available_only: availableOnly ? "true" : undefined,
        project_id: projectId,
        task_type: taskType
      })}`
    ),
  renameModel: (modelId: string, name: string) =>
    jsonFetch<ModelInfo>(`/models/${modelId}`, {
      method: "PATCH",
      body: JSON.stringify({ name })
    }),
  deleteModel: (modelId: string) =>
    jsonFetch<DeleteResponse>(`/models/${modelId}`, {
      method: "DELETE"
    }),
  modelDownloadUrl: (modelId: string) => `${API_BASE}/models/${encodeURIComponent(modelId)}/download`
};
