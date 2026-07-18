import type { TaskType } from "@/types/api";
import type { components } from "@/types/generated/api";
import { jsonFetch, jsonFetchChecked, query } from "@/lib/api/client";
import { jobListSchema, jobWithProgressSchema } from "@/lib/api/schemas";

type EvaluationDataset = components["schemas"]["EvaluationDatasetInfo"];
type EvaluationJob = components["schemas"]["EvaluationJobRead"];
type EvaluationComparison = components["schemas"]["EvaluationComparisonRead"];
type EvaluationPerImageRow = components["schemas"]["EvaluationPerImageRow"];
type DeleteResponse = components["schemas"]["DeleteResponse"];

export const testingApi = {
  datasets: (projectId?: string, taskType?: TaskType) =>
    jsonFetch<EvaluationDataset[]>(`/testing/datasets${query({ project_id: projectId, task_type: taskType })}`),
  testingJobs: (projectId?: string) =>
    jsonFetchChecked<EvaluationJob[]>(`/testing/jobs${query({ project_id: projectId })}`, jobListSchema),
  testingJob: (jobId: string) => jsonFetchChecked<EvaluationJob>(`/testing/jobs/${jobId}`, jobWithProgressSchema),
  testingComparison: (jobId: string) =>
    jsonFetch<EvaluationComparison>(`/testing/jobs/${jobId}/comparison`),
  testingPerImage: (jobId: string) =>
    jsonFetch<EvaluationPerImageRow[]>(`/testing/jobs/${jobId}/per-image`),
  createTestingJob: (payload: {
    project_id: string;
    model_id: string;
    dataset_key: string;
    limit?: number | null;
  }) =>
    jsonFetch<EvaluationJob>("/testing/jobs", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  createTestingJobsBatch: (payload: {
    project_id: string;
    model_ids: string[];
    dataset_key: string;
    limit?: number | null;
  }) =>
    jsonFetch<EvaluationJob[]>("/testing/jobs/batch", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  deleteTestingJobs: (ids: string[], projectId?: string, clearAll = false) =>
    jsonFetch<DeleteResponse>("/testing/jobs/delete", {
      method: "POST",
      body: JSON.stringify({ ids, project_id: projectId, clear_all: clearAll })
    })
};
