"use client";

import { useMutation, useQuery, type UseQueryResult } from "@tanstack/react-query";
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

export function useDatasetCatalogQuery(projectId: string) {
  return useQuery({
    queryKey: ["dataset-catalog", projectId],
    queryFn: () => api.datasetCatalog(projectId)
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
    enabled: Boolean(datasetId)
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

export function useCreateDatasetMutation(options: {
  openDataset: (datasetId: string) => void;
  setShowCreate: (value: boolean) => void;
  setNewDatasetName: (value: string) => void;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
}) {
  return useMutation({
    mutationFn: api.createDataset,
    onSuccess: async (dataset) => {
      options.openDataset(dataset.id);
      options.setShowCreate(false);
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
  split: DatasetSplitFilter;
  setSelectedItemIds: (ids: string[]) => void;
  setSelectedItemId: (id: string) => void;
  setSelectedItemSplit: (split: SplitKey | "") => void;
  itemsQuery: UseQueryResult<DatasetItemPage>;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  edaQuery: UseQueryResult<DatasetEdaSummary>;
}) {
  return useMutation({
    mutationFn: ({ datasetId, ids }: { datasetId: string; ids: string[] }) =>
      api.deleteDatasetItems(datasetId, { split: options.split === "all" ? "unassigned" : options.split, ids }),
    onSuccess: async () => {
      options.setSelectedItemIds([]);
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
  split: DatasetSplitFilter;
  itemsQuery: UseQueryResult<DatasetItemPage>;
  detailQuery: UseQueryResult<DatasetItemDetail>;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  edaQuery: UseQueryResult<DatasetEdaSummary>;
}) {
  return useMutation({
    mutationFn: ({ datasetId, ids, classId }: { datasetId: string; ids: string[]; classId: number }) =>
      api.updateDatasetItemLabels(datasetId, options.split === "all" ? "unassigned" : options.split, { ids, class_id: classId }),
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
      toast.success("Dataset version generated", { action: { label: "Start training", href: "/training" } });
      await Promise.all([
        options.catalogQuery.refetch(),
        options.itemsQuery.refetch(),
        options.edaQuery.refetch()
      ]);
    }
  });
}

export function useMoveDatasetItemsMutation(options: {
  split: DatasetSplitFilter;
  selectedItemIds: string[];
  setSelectedItemIds: (ids: string[]) => void;
  itemsQuery: UseQueryResult<DatasetItemPage>;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  edaQuery: UseQueryResult<DatasetEdaSummary>;
}) {
  return useMutation({
    mutationFn: ({ datasetId, target }: { datasetId: string; target: SplitKey }) =>
      api.moveDatasetItems(datasetId, {
        source_split: options.split === "all" ? "unassigned" : options.split,
        target_split: target,
        ids: options.selectedItemIds
      }),
    onSuccess: async () => {
      options.setSelectedItemIds([]);
      await Promise.all([
        options.itemsQuery.refetch(),
        options.catalogQuery.refetch(),
        options.edaQuery.refetch()
      ]);
    }
  });
}
