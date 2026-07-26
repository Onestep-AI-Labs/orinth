import type { TaskType } from "@/types/api";
import type { components } from "@/types/generated/api";
import { API_BASE, jsonFetch, query } from "@/lib/api/client";

type ModelInfo = components["schemas"]["ModelInfo"];
type DeleteResponse = components["schemas"]["DeleteResponse"];
type ModelUploadOption = components["schemas"]["ModelUploadOption"];
type ModelUploadResult = components["schemas"]["ModelUploadResult"];
type ModelExportStatus = components["schemas"]["ModelExportStatus"];
type ModelExportFormat = components["schemas"]["ModelExportRequest"]["format"];

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
  modelDownloadUrl: (modelId: string) => `${API_BASE}/models/${encodeURIComponent(modelId)}/download`,
  modelUploadOptions: () => jsonFetch<ModelUploadOption[]>("/models/upload-options"),
  uploadModel: (form: FormData) =>
    jsonFetch<ModelUploadResult>("/models/upload", {
      method: "POST",
      body: form
    }),
  modelExports: (modelId: string) =>
    jsonFetch<ModelExportStatus[]>(`/models/${encodeURIComponent(modelId)}/exports`),
  createModelExport: (modelId: string, format: ModelExportFormat) =>
    jsonFetch<ModelExportStatus>(`/models/${encodeURIComponent(modelId)}/export`, {
      method: "POST",
      body: JSON.stringify({ format })
    }),
  modelExportDownloadUrl: (modelId: string, exportId: string) =>
    `${API_BASE}/models/${encodeURIComponent(modelId)}/exports/${encodeURIComponent(exportId)}/download`
};
