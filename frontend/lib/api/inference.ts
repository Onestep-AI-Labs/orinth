import type { components } from "@/types/generated/api";
import { jsonFetch, query } from "@/lib/api/client";

type InferenceResult = components["schemas"]["InferenceResult"];
type InferenceJob = components["schemas"]["InferenceJobRead"];
type DeleteResponse = components["schemas"]["DeleteResponse"];

export const inferenceApi = {
  inferenceHistory: (projectId?: string) =>
    jsonFetch<InferenceResult[]>(`/inference${query({ project_id: projectId })}`),
  createInferenceJob: (form: FormData) =>
    jsonFetch<InferenceJob>("/inference/jobs", {
      method: "POST",
      body: form
    }),
  inferenceJob: (jobId: string) => jsonFetch<InferenceJob>(`/inference/jobs/${jobId}`),
  deleteInference: (ids: string[], projectId?: string, clearAll = false) =>
    jsonFetch<DeleteResponse>("/inference/delete", {
      method: "POST",
      body: JSON.stringify({ ids, project_id: projectId, clear_all: clearAll })
    })
};
