import type { DatasetFormat, TaskType } from "@/types/api";
import type { components } from "@/types/generated/api";
import { jsonFetch, jsonFetchChecked, query, uploadFetch } from "@/lib/api/client";
import { datasetItemPageSchema } from "@/lib/api/schemas";

type DatasetSummary = components["schemas"]["DatasetSummary"];
type DatasetDetection = components["schemas"]["DatasetDetection"];
type DatasetPrepPlan = components["schemas"]["DatasetPrepPlan"];
type DatasetPrepResponse = components["schemas"]["DatasetPrepResponse"];
type DatasetPrepStatus = components["schemas"]["DatasetPrepStatus"];
type DatasetReadiness = components["schemas"]["DatasetReadiness"];
type DatasetPreprocessConfig = components["schemas"]["DatasetPreprocessConfig"];
type DatasetSplitConfig = components["schemas"]["DatasetSplitConfig"];
type DatasetProcessResponse = components["schemas"]["DatasetProcessResponse"];
type DatasetItemPage = components["schemas"]["DatasetItemPage"];
type DatasetTablePage = components["schemas"]["DatasetTablePage"];
type DatasetItemDetail = components["schemas"]["DatasetItemDetail"];
type DatasetItemSummary = components["schemas"]["DatasetItemSummary"];
type DatasetPreprocessPreview = components["schemas"]["DatasetPreprocessPreview"];
type DatasetVersionSummary = components["schemas"]["DatasetVersionSummary"];
type DatasetEdaSummary = components["schemas"]["DatasetEdaSummary"];
type DatasetAnnotation = components["schemas"]["DatasetAnnotation"];
type DeleteResponse = components["schemas"]["DeleteResponse"];
type DatasetHubSearchResponse = components["schemas"]["DatasetHubSearchResponse"];
type DatasetHubFacets = components["schemas"]["DatasetHubFacets"];
type DatasetHubPreview = components["schemas"]["DatasetHubPreview"];
type DatasetHubImportRequest = components["schemas"]["DatasetHubImportRequest"];
type DatasetHubImportResponse = components["schemas"]["DatasetHubImportResponse"];
type DatasetHubIngestRequest = components["schemas"]["DatasetHubIngestRequest"];

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
  /**
   * The grid view of a dataset — same rows as `datasetItems`, projected into
   * columns the table can render for any modality. Its own call rather than a
   * client-side projection of `datasetItems` because which columns exist is a
   * property of the dataset's task, and only the server knows that without
   * loading the whole catalog.
   */
  datasetTable: (
    datasetId: string,
    params: { split: string; class_name?: string; unlabeled?: boolean; limit?: number; offset?: number }
  ) =>
    jsonFetch<DatasetTablePage>(
      `/datasets/${datasetId}/table${query({
        split: params.split,
        class_name: params.class_name,
        unlabeled: params.unlabeled ? "true" : undefined,
        limit: params.limit ?? 50,
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
  /** The browse vocabulary — filter terms plus the sentence explaining each. */
  datasetHubFacets: () => jsonFetch<DatasetHubFacets>("/datasets/hub/facets"),
  searchDatasetHub: (params: {
    query?: string;
    task?: string;
    limit?: number;
    modality?: string[];
    format?: string[];
    size?: string[];
    task_category?: string[];
    sort?: string;
  }) =>
    jsonFetch<DatasetHubSearchResponse>(
      `/datasets/hub/search${query({
        query: params.query,
        task: params.task,
        limit: params.limit,
        modality: params.modality,
        format: params.format,
        size: params.size,
        task_category: params.task_category,
        sort: params.sort
      })}`
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
    }),
  /**
   * Import a Hub split as it is: no column mapping, no declared task.
   *
   * Returns a draft dataset immediately (202) with the download already
   * queued — progress comes from `datasetPrepStatus`, the same readout an
   * uploaded folder gets, because on the server it is the same run.
   */
  ingestDatasetHub: (payload: DatasetHubIngestRequest) =>
    jsonFetch<DatasetSummary>("/datasets/import/hub/as-is", {
      method: "POST",
      body: JSON.stringify(payload)
    }),

  // ---- prep agent (phase 21) ----
  // `form` must carry a `relative_paths` entry per file, in the same order as
  // `files`: the directory layout is what detection reads, and a flat list of
  // names looks the same for a YOLO export and a folder of holiday photos.
  ingestDataset: (form: FormData, onProgress?: (fraction: number) => void) =>
    uploadFetch<DatasetSummary>("/datasets/ingest", form, onProgress),
  ingestIntoDataset: (datasetId: string, form: FormData) =>
    jsonFetch<DatasetSummary>(`/datasets/${datasetId}/ingest`, { method: "POST", body: form }),
  detectDataset: (datasetId: string) =>
    jsonFetch<DatasetDetection>(`/datasets/${datasetId}/prep/detect`),
  /** Queues a run and returns at once; the dataset comes back already `planning`. */
  runDatasetPrep: (datasetId: string, payload: { auto_apply?: boolean } = {}) =>
    jsonFetch<DatasetSummary>(`/datasets/${datasetId}/prep`, {
      method: "POST",
      body: JSON.stringify({ auto_apply: payload.auto_apply ?? true })
    }),
  /** Just the run state. Cheap by design — the catalog costs seconds to build. */
  datasetPrepStatus: (datasetId: string) =>
    jsonFetch<DatasetPrepStatus>(`/datasets/${datasetId}/prep/status`),
  datasetPrepPlan: (datasetId: string) =>
    jsonFetch<DatasetPrepPlan>(`/datasets/${datasetId}/prep`),
  applyDatasetPrep: (datasetId: string, plan: DatasetPrepPlan) =>
    jsonFetch<DatasetPrepResponse>(`/datasets/${datasetId}/prep/apply`, {
      method: "POST",
      body: JSON.stringify({ plan })
    }),
  undoDatasetPrep: (datasetId: string) =>
    jsonFetch<DatasetSummary>(`/datasets/${datasetId}/prep/undo`, { method: "POST" }),
  discardStagedFiles: (datasetId: string) =>
    jsonFetch<DatasetSummary>(`/datasets/${datasetId}/prep/discard-staged`, { method: "POST" }),
  datasetReadiness: (datasetId: string) =>
    jsonFetch<DatasetReadiness>(`/datasets/${datasetId}/readiness`)
};
