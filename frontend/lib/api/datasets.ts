import type { DatasetFormat, TaskType } from "@/types/api";
import type { components } from "@/types/generated/api";
import { jsonFetch, jsonFetchChecked, query } from "@/lib/api/client";
import { datasetItemPageSchema } from "@/lib/api/schemas";

type DatasetSummary = components["schemas"]["DatasetSummary"];
type DatasetPreprocessConfig = components["schemas"]["DatasetPreprocessConfig"];
type DatasetSplitConfig = components["schemas"]["DatasetSplitConfig"];
type DatasetProcessResponse = components["schemas"]["DatasetProcessResponse"];
type DatasetItemPage = components["schemas"]["DatasetItemPage"];
type DatasetItemDetail = components["schemas"]["DatasetItemDetail"];
type DatasetItemSummary = components["schemas"]["DatasetItemSummary"];
type DatasetPreprocessPreview = components["schemas"]["DatasetPreprocessPreview"];
type DatasetVersionSummary = components["schemas"]["DatasetVersionSummary"];
type DatasetEdaSummary = components["schemas"]["DatasetEdaSummary"];
type DatasetAnnotation = components["schemas"]["DatasetAnnotation"];
type DeleteResponse = components["schemas"]["DeleteResponse"];
type DatasetHubSearchResponse = components["schemas"]["DatasetHubSearchResponse"];
type DatasetHubPreview = components["schemas"]["DatasetHubPreview"];
type DatasetHubImportRequest = components["schemas"]["DatasetHubImportRequest"];
type DatasetHubImportResponse = components["schemas"]["DatasetHubImportResponse"];

export const datasetsApi = {
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
    jsonFetch<DatasetSummary>(
      `/datasets/${datasetId}/labels/${labelIndex}${query({ force: String(force) })}`,
      {
        method: "DELETE"
      }
    ),
  datasetItems: (
    datasetId: string,
    params: { split: string; class_name?: string; unlabeled?: boolean; limit?: number; offset?: number }
  ) =>
    jsonFetchChecked<DatasetItemPage>(
      `/datasets/${datasetId}/items${query({
        split: params.split,
        class_name: params.class_name,
        unlabeled: params.unlabeled ? "true" : undefined,
        limit: params.limit ?? 240,
        offset: params.offset
      })}`,
      datasetItemPageSchema
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
  createDatasetRecord: (datasetId: string, payload: { split: string; record: Record<string, any> }) =>
    jsonFetch<DatasetItemDetail>(`/datasets/${datasetId}/records`, {
      method: "POST",
      body: JSON.stringify(payload)
    }),
  saveDatasetRecord: (datasetId: string, split: string, itemId: string, record: Record<string, any>) =>
    jsonFetch<DatasetItemDetail>(
      `/datasets/${datasetId}/records/${split}/${encodeURIComponent(itemId)}`,
      {
        method: "PUT",
        body: JSON.stringify({ record })
      }
    ),
  uploadDatasetRecords: (datasetId: string, form: FormData) =>
    jsonFetch<{ imported: number; skipped: number; warnings: string[] }>(
      `/datasets/${datasetId}/records/upload`,
      {
        method: "POST",
        body: form
      }
    ),
  searchDatasetHub: (params: { query?: string; task?: string; limit?: number }) =>
    jsonFetch<DatasetHubSearchResponse>(
      `/datasets/hub/search${query({ query: params.query, task: params.task, limit: params.limit })}`
    ),
  previewDatasetHub: (params: { hub_id: string; config?: string; split?: string; limit?: number }) =>
    jsonFetch<DatasetHubPreview>(
      `/datasets/hub/preview${query({
        hub_id: params.hub_id,
        config: params.config,
        split: params.split,
        limit: params.limit
      })}`
    ),
  importDatasetHub: (payload: DatasetHubImportRequest) =>
    jsonFetch<DatasetHubImportResponse>("/datasets/import/hub", {
      method: "POST",
      body: JSON.stringify(payload)
    })
};
