import type { components } from "@/types/generated/api";
import { jsonFetch, jsonFetchChecked, query } from "@/lib/api/client";
import { inferenceResultListSchema, jobWithProgressSchema } from "@/lib/api/schemas";

type InferenceResult = components["schemas"]["InferenceResult"];
type InferenceJob = components["schemas"]["InferenceJobRead"];
type DeleteResponse = components["schemas"]["DeleteResponse"];

export const inferenceApi = {
  inferenceHistory: (projectId?: string) =>
    jsonFetchChecked<InferenceResult[]>(`/inference${query({ project_id: projectId })}`, inferenceResultListSchema),
  createInferenceJob: (form: FormData) =>
    jsonFetch<InferenceJob>("/inference/jobs", {
      method: "POST",
      body: form
    }),
  inferenceJob: (jobId: string) => jsonFetchChecked<InferenceJob>(`/inference/jobs/${jobId}`, jobWithProgressSchema),
  deleteInference: (ids: string[], projectId?: string, clearAll = false) =>
    jsonFetch<DeleteResponse>("/inference/delete", {
      method: "POST",
      body: JSON.stringify({ ids, project_id: projectId, clear_all: clearAll })
    })
};
