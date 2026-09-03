"use client";

import { keepPreviousData, useMutation, useQuery, type UseQueryResult } from "@tanstack/react-query";
import { api, apiAssetUrl } from "@/lib/api";
import { toast } from "@/features/platform/toast";
import { TRAINING_SPLITS } from "@/features/platform/constants";
import { preprocessFromDataset, splitConfigFromDataset } from "@/features/platform/utils";
import type {
  DatasetEdaSummary,
  DatasetItemDetail,
  DatasetItemPage,
  DatasetPreprocessConfig,
  DatasetSplitConfig,
  DatasetSplitFilter,
  DatasetSummary,
  DatasetVersionSummary,
  SplitKey
} from "@/types/api";

/**
 * Data-access hooks for the dataset page. Query keys and mutation
 * behavior are unchanged from the original inline implementation in
 * `dataset-page.tsx` -- this module only relocates the blocks so the
 * page component can consume them.
 */

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/** Prep states that mean the agent is still working on a dataset. */
const PREP_IN_FLIGHT = new Set(["detecting", "planning", "applying"]);

export function useDatasetCatalogQuery(projectId: string) {
  return useQuery({
    queryKey: ["dataset-catalog", projectId],
    queryFn: () => api.datasetCatalog(projectId),
    // A prep run outlives the click that started it, so a reload mid-run must
    // still converge. Only while one is actually in flight: building this
    // response walks every split of every dataset and costs seconds on a large
    // one, which is why the run itself is watched through `/prep/status`.
    refetchInterval: (query) =>
      query.state.data?.some((dataset) => PREP_IN_FLIGHT.has(dataset.prep?.state ?? ""))
        ? 4000
        : false
  });
}

export function useDatasetItemsQuery(
  datasetId: string | undefined,
  split: DatasetSplitFilter,
  classFilter: string,
  imagePage: number,
  imagesPerPage: number
) {
  return useQuery({
    queryKey: ["dataset-items", datasetId, split, classFilter, imagePage, imagesPerPage],
    queryFn: () =>
      api.datasetItems(datasetId ?? "", {
        split,
        class_name: classFilter && classFilter !== "__unlabeled__" ? classFilter : undefined,
        unlabeled: classFilter === "__unlabeled__",
        limit: imagesPerPage,
        offset: imagePage * imagesPerPage
      }),
    enabled: Boolean(datasetId),
    placeholderData: keepPreviousData
  });
}

export function useDatasetItemDetailQuery(
  datasetId: string | undefined,
  split: SplitKey | "",
  itemId: string,
  enabled: boolean
) {
  return useQuery({
    queryKey: ["dataset-item", datasetId, split, itemId],
    queryFn: () => api.datasetItem(datasetId ?? "", split, itemId),
    enabled
  });
}

export function useDatasetVersionsQuery(datasetId: string | undefined) {
  return useQuery({
    queryKey: ["dataset-versions", datasetId],
    queryFn: () => api.datasetVersions(datasetId ?? ""),
    enabled: Boolean(datasetId)
  });
}

export function useDatasetEdaQuery(datasetId: string | undefined, split: DatasetSplitFilter) {
  return useQuery({
    queryKey: ["dataset-eda", datasetId, split],
    queryFn: () => api.datasetEda(datasetId ?? "", split),
    enabled: Boolean(datasetId)
  });
}

// ---------------------------------------------------------------------------
// Mutations
// ---------------------------------------------------------------------------

export type SelectedDatasetItem = { id: string; split: SplitKey };

/**
 * Multi-select actions operate under the "all" filter, where selected items
 * can span several real splits even though the batch endpoints take one
 * split per call. Group by each item's actual split (captured at selection
 * time) and issue one request per group.
 */
function groupBySplit(items: SelectedDatasetItem[]): Map<SplitKey, string[]> {
  const groups = new Map<SplitKey, string[]>();
  for (const item of items) {
    const ids = groups.get(item.split);
    if (ids) ids.push(item.id);
    else groups.set(item.split, [item.id]);
  }
  return groups;
}

export function useCreateDatasetMutation(options: {
  openDataset: (datasetId: string) => void;
  setNewDatasetName: (value: string) => void;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
}) {
  return useMutation({
    mutationFn: api.createDataset,
    onSuccess: async (dataset) => {
      // Creating navigates into the dataset, so the source panel it was
      // submitted from unmounts on its own — it no longer has to be told to
      // close, which is why `setShowCreate` is gone.
      options.openDataset(dataset.id);
      options.setNewDatasetName("");
      await options.catalogQuery.refetch();
    }
  });
}

export function useUpdateDatasetMutation(options: {
  setEditName: (value: string) => void;
  setPreprocessConfig: (value: DatasetPreprocessConfig) => void;
  setShowRename: (value: boolean) => void;
  setShowDatasetOptions: (value: boolean) => void;
  setCatalogRenameDatasetId: (value: string) => void;
  setCatalogRenameDraft: (value: string) => void;
  setCatalogMenuDatasetId: (value: string) => void;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
}) {
  return useMutation({
    mutationFn: ({ datasetId, name, preprocess }: { datasetId: string; name?: string; preprocess?: DatasetPreprocessConfig }) =>
      api.updateDataset(datasetId, { name, preprocess }),
    onSuccess: async (dataset) => {
      options.setEditName(dataset.name);
      options.setPreprocessConfig(preprocessFromDataset(dataset));
      options.setShowRename(false);
      options.setShowDatasetOptions(false);
      options.setCatalogRenameDatasetId("");
      options.setCatalogRenameDraft("");
      options.setCatalogMenuDatasetId("");
      await options.catalogQuery.refetch();
    }
  });
}

export function useCloneDatasetMutation(options: {
  projectId: string;
  openDataset: (datasetId: string) => void;
  setCloneName: (value: string) => void;
  setShowDuplicate: (value: boolean) => void;
  setShowDatasetOptions: (value: boolean) => void;
  setCatalogMenuDatasetId: (value: string) => void;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
}) {
  return useMutation({
    mutationFn: ({ datasetId, name }: { datasetId: string; name?: string }) =>
      api.cloneDataset(datasetId, { name, project_id: options.projectId }),
    onSuccess: async (dataset) => {
      options.openDataset(dataset.id);
      options.setCloneName("");
      options.setShowDuplicate(false);
      options.setShowDatasetOptions(false);
      options.setCatalogMenuDatasetId("");
      await options.catalogQuery.refetch();
    }
  });
}

export function useUploadDatasetMutation(options: {
  setUploadFiles: (files: File[]) => void;
  setUploadErrors: (errors: Array<Record<string, string>>) => void;
  setSelectedItemId: (id: string) => void;
  setSelectedItemSplit: (split: SplitKey | "") => void;
  itemsQuery: UseQueryResult<DatasetItemPage>;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  edaQuery: UseQueryResult<DatasetEdaSummary>;
}) {
  return useMutation({
    mutationFn: ({ datasetId, form }: { datasetId: string; form: FormData }) =>
      api.uploadDatasetImages(datasetId, form),
    onSuccess: async (result) => {
      options.setUploadFiles([]);
      options.setUploadErrors(result.errors ?? []);
      const lastItem = result.uploaded.at(-1);
      if (lastItem) {
        options.setSelectedItemId(lastItem.id);
        options.setSelectedItemSplit(lastItem.split);
      }
      if (result.uploaded.length > 0) {
        toast.success(`${result.uploaded.length} ${result.uploaded.length === 1 ? "file" : "files"} added — annotate them next`);
      }
      await Promise.all([
        options.itemsQuery.refetch(),
        options.catalogQuery.refetch(),
        options.edaQuery.refetch()
      ]);
    }
  });
}

export function useDeleteDatasetItemsMutation(options: {
  setSelectedItemIds: (ids: string[]) => void;
  setSelectedItemSplits: (splits: Record<string, SplitKey>) => void;
  setSelectedItemId: (id: string) => void;
  setSelectedItemSplit: (split: SplitKey | "") => void;
  itemsQuery: UseQueryResult<DatasetItemPage>;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  edaQuery: UseQueryResult<DatasetEdaSummary>;
}) {
  return useMutation({
    mutationFn: async ({ datasetId, items }: { datasetId: string; items: SelectedDatasetItem[] }) => {
      const groups = groupBySplit(items);
      const responses = await Promise.all(
        Array.from(groups.entries()).map(([split, ids]) => api.deleteDatasetItems(datasetId, { split, ids }))
      );
      return responses.reduce(
        (total, response) => ({
          deleted: total.deleted + response.deleted,
          missing: [...total.missing, ...(response.missing ?? [])]
        }),
        { deleted: 0, missing: [] as string[] }
      );
    },
    onSuccess: async () => {
      options.setSelectedItemIds([]);
      options.setSelectedItemSplits({});
      options.setSelectedItemId("");
      options.setSelectedItemSplit("");
      await Promise.all([
        options.itemsQuery.refetch(),
        options.catalogQuery.refetch(),
        options.edaQuery.refetch()
      ]);
    }
  });
}

export function useBulkLabelDatasetItemsMutation(options: {
  itemsQuery: UseQueryResult<DatasetItemPage>;
  detailQuery: UseQueryResult<DatasetItemDetail>;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  edaQuery: UseQueryResult<DatasetEdaSummary>;
}) {
  return useMutation({
    mutationFn: async ({
      datasetId,
      items,
      classId
    }: {
      datasetId: string;
      items: SelectedDatasetItem[];
      classId: number;
    }) => {
      const groups = groupBySplit(items);
      const responses = await Promise.all(
        Array.from(groups.entries()).map(([split, ids]) =>
          api.updateDatasetItemLabels(datasetId, split, { ids, class_id: classId })
        )
      );
      return responses.reduce(
        (total, response) => ({
          updated: total.updated + response.updated,
          missing: [...total.missing, ...(response.missing ?? [])]
        }),
        { updated: 0, missing: [] as string[] }
      );
    },
    onSuccess: async () => {
      await Promise.all([
        options.itemsQuery.refetch(),
        options.detailQuery.refetch(),
        options.catalogQuery.refetch(),
        options.edaQuery.refetch()
      ]);
    }
  });
}

export function usePreviewDatasetPreprocessMutation(options: {
  preprocessConfig: DatasetPreprocessConfig;
  setPreviewUrl: (url: string | null) => void;
  setPreviewText: (text: string | null) => void;
}) {
  return useMutation({
    mutationFn: ({ datasetId, item }: { datasetId: string; item: DatasetItemDetail }) =>
      api.previewDatasetPreprocess(datasetId, item.split, item.id, options.preprocessConfig),
    onSuccess: (preview) => {
      options.setPreviewUrl(preview.media_type === "text" ? null : apiAssetUrl(preview.image_url));
      options.setPreviewText(preview.media_type === "text" ? preview.text_preview ?? "" : null);
    }
  });
}

export function useCreateDatasetVersionMutation(options: {
  setVersionName: (value: string) => void;
  versionsQuery: UseQueryResult<DatasetVersionSummary[]>;
}) {
  return useMutation({
    mutationFn: ({ datasetId, name, config }: { datasetId: string; name?: string; config: DatasetPreprocessConfig }) =>
      api.createDatasetVersion(datasetId, {
        name,
        config,
        splits: TRAINING_SPLITS,
        augmentation_splits: ["train"]
      }),
    onSuccess: async () => {
      options.setVersionName("");
      await options.versionsQuery.refetch();
    }
  });
}

export function useDeleteDatasetMutation(options: {
  showDatasetCatalog: () => void;
  setSelectedItemId: (id: string) => void;
  setSelectedItemSplit: (split: SplitKey | "") => void;
  setShowDatasetOptions: (value: boolean) => void;
  setCatalogMenuDatasetId: (value: string) => void;
  setCatalogRenameDatasetId: (value: string) => void;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
}) {
  return useMutation({
    mutationFn: api.deleteDataset,
    onSuccess: async () => {
      options.showDatasetCatalog();
      options.setSelectedItemId("");
      options.setSelectedItemSplit("");
      options.setShowDatasetOptions(false);
      options.setCatalogMenuDatasetId("");
      options.setCatalogRenameDatasetId("");
      await options.catalogQuery.refetch();
    }
  });
}

export function useProcessDatasetMutation(options: {
  setPreprocessConfig: (value: DatasetPreprocessConfig) => void;
  setSplitConfig: (value: DatasetSplitConfig) => void;
  setSplit: (value: DatasetSplitFilter) => void;
  setSelectedItemIds: (ids: string[]) => void;
  setSelectedItemSplits: (splits: Record<string, SplitKey>) => void;
  setSelectedItemId: (id: string) => void;
  setSelectedItemSplit: (split: SplitKey | "") => void;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  itemsQuery: UseQueryResult<DatasetItemPage>;
  edaQuery: UseQueryResult<DatasetEdaSummary>;
}) {
  return useMutation({
    mutationFn: ({ datasetId, preprocess, split }: { datasetId: string; preprocess: DatasetPreprocessConfig; split: DatasetSplitConfig }) =>
      api.processDataset(datasetId, { preprocess, split }),
    onSuccess: async (result) => {
      options.setPreprocessConfig(preprocessFromDataset(result.dataset));
      options.setSplitConfig(splitConfigFromDataset(result.dataset));
      options.setSplit("train");
      options.setSelectedItemIds([]);
      options.setSelectedItemSplits({});
      // Items were redistributed across splits; drop the single selection so its
      // detail query does not fetch a now-moved record from its old split.
      options.setSelectedItemId("");
      options.setSelectedItemSplit("");
      toast.success("Dataset version generated", { action: { label: "Start training", href: "/training" } });
      await Promise.all([
        options.catalogQuery.refetch(),
        options.itemsQuery.refetch(),
        options.edaQuery.refetch()
      ]);
    }
  });
}

export function useCreateDatasetRecordMutation(options: {
  itemsQuery: UseQueryResult<DatasetItemPage>;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  edaQuery: UseQueryResult<DatasetEdaSummary>;
  onCreated?: (item: DatasetItemDetail) => void;
}) {
  return useMutation({
    mutationFn: ({ datasetId, split, record }: { datasetId: string; split: SplitKey; record: Record<string, unknown> }) =>
      api.createDatasetRecord(datasetId, { split, record }),
    onSuccess: async (item) => {
      toast.success("Record added");
      options.onCreated?.(item);
      await Promise.all([options.itemsQuery.refetch(), options.catalogQuery.refetch(), options.edaQuery.refetch()]);
    }
  });
}

export function useSaveDatasetRecordMutation(options: {
  itemsQuery: UseQueryResult<DatasetItemPage>;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  edaQuery: UseQueryResult<DatasetEdaSummary>;
  detailQuery: UseQueryResult<DatasetItemDetail>;
  onSaved?: () => void;
}) {
  return useMutation({
    mutationFn: ({
      datasetId,
      split,
      itemId,
      record
    }: {
      datasetId: string;
      split: SplitKey;
      itemId: string;
      record: Record<string, unknown>;
    }) => api.saveDatasetRecord(datasetId, split, itemId, record),
    onSuccess: async () => {
      toast.success("Record saved");
      options.onSaved?.();
      await Promise.all([
        options.itemsQuery.refetch(),
        options.catalogQuery.refetch(),
        options.edaQuery.refetch(),
        options.detailQuery.refetch()
      ]);
    }
  });
}

export function useUploadDatasetRecordsMutation(options: {
  itemsQuery: UseQueryResult<DatasetItemPage>;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  edaQuery: UseQueryResult<DatasetEdaSummary>;
}) {
  return useMutation({
    mutationFn: ({ datasetId, form }: { datasetId: string; form: FormData }) =>
      api.uploadDatasetRecords(datasetId, form),
    onSuccess: async (result) => {
      const skipped = result.skipped ? `, ${result.skipped} skipped` : "";
      toast.success(`Uploaded ${result.imported} records${skipped}`);
      await Promise.all([options.itemsQuery.refetch(), options.catalogQuery.refetch(), options.edaQuery.refetch()]);
    }
  });
}

export function useImportHubDatasetMutation(options: {
  openDataset: (datasetId: string) => void;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  onImported?: () => void;
}) {
  return useMutation({
    mutationFn: api.importDatasetHub,
    onSuccess: async (response) => {
      const skipped = response.skipped_rows ? `, ${response.skipped_rows} skipped` : "";
      toast.success(`Imported ${response.imported_rows} rows${skipped}`);
      options.onImported?.();
      await options.catalogQuery.refetch();
      options.openDataset(response.dataset.id);
    },
    onError: async () => {
      // A slow import can outlive the dev proxy's socket timeout even though the
      // backend finished writing the dataset. Refresh the catalog so it still
      // appears under "Imported from HuggingFace" instead of seeming to vanish.
      await options.catalogQuery.refetch();
    }
  });
}

export function useMoveDatasetItemsMutation(options: {
  setSelectedItemIds: (ids: string[]) => void;
  setSelectedItemSplits: (splits: Record<string, SplitKey>) => void;
  itemsQuery: UseQueryResult<DatasetItemPage>;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  edaQuery: UseQueryResult<DatasetEdaSummary>;
}) {
  return useMutation({
    mutationFn: async ({
      datasetId,
      items,
      target
    }: {
      datasetId: string;
      items: SelectedDatasetItem[];
      target: SplitKey;
    }) => {
      const groups = groupBySplit(items.filter((item) => item.split !== target));
      const responses = await Promise.all(
        Array.from(groups.entries()).map(([split, ids]) =>
          api.moveDatasetItems(datasetId, { source_split: split, target_split: target, ids })
        )
      );
      return responses.reduce(
        (total, response) => ({
          moved: total.moved + response.moved,
          missing: [...total.missing, ...(response.missing ?? [])]
        }),
        { moved: 0, missing: [] as string[] }
      );
    },
    onSuccess: async () => {
      options.setSelectedItemIds([]);
      options.setSelectedItemSplits({});
      await Promise.all([
        options.itemsQuery.refetch(),
        options.catalogQuery.refetch(),
        options.edaQuery.refetch()
      ]);
    }
  });
}
