import type {
  DatasetAnnotation,
  DatasetEdaSummary,
  DatasetFormat,
  DatasetItemDetail,
  DatasetItemSummary,
  DatasetPreprocessConfig,
  DatasetPreprocessPreview,
  DatasetProcessResponse,
  DatasetSplitConfig,
  DatasetSummary,
  DatasetVersionSummary,
  DeleteResponse,
  EvaluationComparison,
  EvaluationDataset,
  EvaluationJob,
  EvaluationPerImageRow,
  InferenceJob,
  InferenceResult,
  ModelAssetStatus,
  ModelInfo,
  ProjectSummary,
  TaskType,
  TrainingJob,
  TrainingModelOption
} from "@/types/api";

export const API_ORIGIN = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
const API_BASE = `${API_ORIGIN}/api`;

export function mediaUrl(path: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${API_ORIGIN}${path}`;
}

export function apiAssetUrl(path: string | null): string | null {
  return mediaUrl(path);
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" })
    }
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // keep status text
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json() as Promise<T>;
}

function query(params: Record<string, string | number | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  }
  const value = search.toString();
  return value ? `?${value}` : "";
}

export const api = {
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
    }),

  models: (availableOnly = false, projectId?: string, taskType?: TaskType) =>
    jsonFetch<ModelInfo[]>(
      `/models${query({
        available_only: availableOnly ? "true" : undefined,
        project_id: projectId,
        task_type: taskType
      })}`
    ),

  datasetCatalog: (projectId?: string) =>
    jsonFetch<DatasetSummary[]>(`/datasets${query({ project_id: projectId })}`),
  createDataset: (payload: {
    project_id: string;
    name: string;
    task_type: TaskType;
    format?: DatasetFormat;
    labels?: string[];
  }) =>
    jsonFetch<DatasetSummary>("/datasets", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  updateDataset: (
    datasetId: string,
    payload: {
      name?: string;
      metadata?: Record<string, any>;
      preprocess?: DatasetPreprocessConfig;
    }
  ) =>
    jsonFetch<DatasetSummary>(`/datasets/${datasetId}`, {
      method: "PATCH",
      body: JSON.stringify(payload)
    }),
  processDataset: (
    datasetId: string,
    payload: {
      preprocess?: DatasetPreprocessConfig;
      split?: DatasetSplitConfig;
    }
  ) =>
    jsonFetch<DatasetProcessResponse>(`/datasets/${datasetId}/process`, {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  importDataset: (payload: {
    project_id: string;
    path: string;
    name?: string | null;
    task_type: TaskType;
    format?: DatasetFormat;
    labels?: string[];
  }) =>
    jsonFetch<DatasetSummary>("/datasets/import", {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  cloneDataset: (datasetId: string, payload: { name?: string | null; project_id?: string | null }) =>
    jsonFetch<DatasetSummary>(`/datasets/${datasetId}/clone`, {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  deleteDataset: (datasetId: string) =>
    jsonFetch<DeleteResponse>(`/datasets/${datasetId}`, {
      method: "DELETE"
    }),
  addDatasetLabel: (datasetId: string, name: string) =>
    jsonFetch<DatasetSummary>(`/datasets/${datasetId}/labels`, {
      method: "POST",
      body: JSON.stringify({ name })
    }),
  renameDatasetLabel: (datasetId: string, labelIndex: number, name: string) =>
    jsonFetch<DatasetSummary>(`/datasets/${datasetId}/labels/${labelIndex}`, {
      method: "PATCH",
      body: JSON.stringify({ name })
    }),
  deleteDatasetLabel: (datasetId: string, labelIndex: number, force = false) =>
    jsonFetch<DatasetSummary>(`/datasets/${datasetId}/labels/${labelIndex}${query({ force: String(force) })}`, {
      method: "DELETE"
    }),
  datasetItems: (
    datasetId: string,
    params: { split: string; class_name?: string; unlabeled?: boolean; limit?: number; offset?: number }
  ) =>
    jsonFetch<DatasetItemSummary[]>(
      `/datasets/${datasetId}/items${query({
        split: params.split,
        class_name: params.class_name,
        unlabeled: params.unlabeled ? "true" : undefined,
        limit: params.limit ?? 240,
        offset: params.offset
      })}`
    ),
  datasetItem: (datasetId: string, split: string, itemId: string) =>
    jsonFetch<DatasetItemDetail>(
      `/datasets/${datasetId}/items/${split}/${encodeURIComponent(itemId)}`
    ),
  uploadDatasetImage: (datasetId: string, form: FormData) =>
    jsonFetch<DatasetItemDetail>(`/datasets/${datasetId}/items`, {
      method: "POST",
      body: form
    }),
  uploadDatasetImages: (datasetId: string, form: FormData) =>
    jsonFetch<{ uploaded: DatasetItemDetail[]; errors: Array<Record<string, string>> }>(
      `/datasets/${datasetId}/items/batch`,
      {
        method: "POST",
        body: form
      }
    ),
  deleteDatasetItems: (datasetId: string, payload: { split: string; ids: string[] }) =>
    jsonFetch<DeleteResponse>(`/datasets/${datasetId}/items/delete`, {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  moveDatasetItems: (
    datasetId: string,
    payload: { source_split: string; target_split: string; ids: string[] }
  ) =>
    jsonFetch<{ moved: number; missing: string[]; items: DatasetItemSummary[] }>(
      `/datasets/${datasetId}/items/move`,
      {
        method: "POST",
        body: JSON.stringify(payload)
      }
    ),
  updateDatasetItemLabel: (
    datasetId: string,
    split: string,
    itemId: string,
    payload: { class_id?: number; class_name?: string }
  ) =>
    jsonFetch<DatasetItemDetail>(
      `/datasets/${datasetId}/items/${split}/${encodeURIComponent(itemId)}/label`,
      {
        method: "PATCH",
        body: JSON.stringify(payload)
      }
    ),
  updateDatasetItemLabels: (
    datasetId: string,
    split: string,
    payload: { ids: string[]; class_id?: number; class_name?: string }
  ) =>
    jsonFetch<{ updated: number; missing: string[]; items: DatasetItemSummary[] }>(
      `/datasets/${datasetId}/items/${split}/labels`,
      {
        method: "PATCH",
        body: JSON.stringify(payload)
      }
    ),
  previewDatasetPreprocess: (
    datasetId: string,
    split: string,
    itemId: string,
    config?: DatasetPreprocessConfig
  ) =>
    jsonFetch<DatasetPreprocessPreview>(
      `/datasets/${datasetId}/items/${split}/${encodeURIComponent(itemId)}/preprocess-preview`,
      {
        method: "POST",
        body: JSON.stringify({ config })
      }
    ),
  datasetVersions: (datasetId: string) =>
    jsonFetch<DatasetVersionSummary[]>(`/datasets/${datasetId}/versions`),
  createDatasetVersion: (
    datasetId: string,
    payload: {
      name?: string | null;
      config?: DatasetPreprocessConfig;
      splits?: string[];
      augmentation_splits?: string[];
    }
  ) =>
    jsonFetch<DatasetVersionSummary>(`/datasets/${datasetId}/versions`, {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  datasetEda: (datasetId: string, split: string) =>
    jsonFetch<DatasetEdaSummary>(`/datasets/${datasetId}/eda${query({ split })}`),
  saveDatasetAnnotations: (
    datasetId: string,
    split: string,
    itemId: string,
    annotations: DatasetAnnotation[]
  ) =>
    jsonFetch<DatasetItemDetail>(
      `/datasets/${datasetId}/items/${split}/${encodeURIComponent(itemId)}/annotations`,
      {
        method: "PUT",
        body: JSON.stringify({ annotations })
      }
    ),

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
    }),

  datasets: (projectId?: string) =>
    jsonFetch<EvaluationDataset[]>(`/testing/datasets${query({ project_id: projectId })}`),
  testingJobs: (projectId?: string) =>
    jsonFetch<EvaluationJob[]>(`/testing/jobs${query({ project_id: projectId })}`),
  testingJob: (jobId: string) => jsonFetch<EvaluationJob>(`/testing/jobs/${jobId}`),
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
    }),

  trainingOptions: (taskType?: TaskType) =>
    jsonFetch<TrainingModelOption[]>(`/training/model-options${query({ task_type: taskType })}`),
  prepareModelAsset: (optionId: string, download = true) =>
    jsonFetch<ModelAssetStatus>("/training/model-assets/prepare", {
      method: "POST",
      body: JSON.stringify({ option_id: optionId, download })
    }),
  trainingJobs: (projectId?: string) =>
    jsonFetch<TrainingJob[]>(`/training/jobs${query({ project_id: projectId })}`),
  createTrainingJob: (payload: {
    project_id: string;
    task_type: TaskType;
    model_family: string;
    model_option_id: string;
    base_model?: string | null;
    epochs: number;
    image_size: number;
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
  trainingJob: (jobId: string) => jsonFetch<TrainingJob>(`/training/jobs/${jobId}`),
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
