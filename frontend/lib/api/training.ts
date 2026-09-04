import type { TaskType } from "@/types/api";
import type { components } from "@/types/generated/api";
import { jsonFetch, jsonFetchChecked, query } from "@/lib/api/client";
import { jobListSchema, jobWithProgressSchema } from "@/lib/api/schemas";

type TrainingJob = components["schemas"]["TrainingJobRead"];
type TrainingModelOption = components["schemas"]["TrainingModelOption"];
type ModelAssetStatus = components["schemas"]["ModelAssetStatus"];
type DeleteResponse = components["schemas"]["DeleteResponse"];
type LlmEnvironment = components["schemas"]["LlmEnvironment"];
type LlmModelInfo = components["schemas"]["LlmModelInfo"];
type ComputeEnvironment = components["schemas"]["ComputeEnvironment"];
type ComputePlan = components["schemas"]["ComputePlan"];
type ComputeProvider = components["schemas"]["ComputeProvider"];

export const trainingApi = {
  /** What this machine can train on, per framework. Cached server-side. */
  computeEnvironment: () => jsonFetch<ComputeEnvironment>("/training/compute"),
  /** Resolve `device: auto` into what a run would actually do, before it starts. */
  computePlan: (params: {
    task_type: string;
    model_family: string;
    device?: string;
    batch_size?: number;
  }) => jsonFetch<ComputePlan>(`/training/compute/plan${query({ ...params })}`),
  /** Where a run could execute. Only `local` is available today. */
  computeProviders: () => jsonFetch<ComputeProvider[]>("/training/compute/providers"),
  trainingOptions: (taskType?: TaskType) =>
    jsonFetch<TrainingModelOption[]>(`/training/model-options${query({ task_type: taskType })}`),
  llmEnvironment: () => jsonFetch<LlmEnvironment>("/training/llm/environment"),
  llmModelInfo: (modelRef: string) =>
    jsonFetch<LlmModelInfo>(`/training/llm/model-info${query({ model_ref: modelRef })}`),
  prepareModelAsset: (optionId: string, download = true, modelRef?: string) =>
    jsonFetch<ModelAssetStatus>("/training/model-assets/prepare", {
      method: "POST",
      body: JSON.stringify({ option_id: optionId, download, model_ref: modelRef ?? null })
    }),
  trainingJobs: (projectId?: string) =>
    jsonFetchChecked<TrainingJob[]>(`/training/jobs${query({ project_id: projectId })}`, jobListSchema),
  createTrainingJob: (payload: {
    project_id: string;
    task_type: TaskType;
    model_family: string;
    model_option_id: string;
    model_name?: string | null;
    base_model?: string | null;
    epochs: number;
    image_size?: number;
    batch_size: number;
    dataset_id?: string;
    optimizer?: string;
    learning_rate?: number;
    architecture?: Record<string, any>;
    hyperparameters?: Record<string, any>;
    device?: string;
    cache?: "disk" | "ram" | "none";
    workers?: number;
    patience?: number;
  }) =>
    jsonFetch<TrainingJob>("/training/jobs", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  trainingJob: (jobId: string) => jsonFetchChecked<TrainingJob>(`/training/jobs/${jobId}`, jobWithProgressSchema),
  cancelTrainingJob: (jobId: string) =>
    jsonFetch<TrainingJob>(`/training/jobs/${jobId}/cancel`, {
      method: "POST"
    }),
  promoteTrainingJob: (jobId: string) =>
    jsonFetch<{ model_id: string }>(`/training/jobs/${jobId}/promote`, {
      method: "POST"
    }),
  deleteTrainingJobs: (ids: string[], projectId?: string, clearAll = false) =>
    jsonFetch<DeleteResponse>("/training/jobs/delete", {
      method: "POST",
      body: JSON.stringify({ ids, project_id: projectId, clear_all: clearAll })
    })
};
