import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import {
  useBulkLabelDatasetItemsMutation,
  useCreateDatasetMutation,
  useDatasetCatalogQuery,
  useDatasetEdaQuery,
  useDatasetItemDetailQuery,
  useDatasetItemsQuery,
  useDatasetVersionsQuery,
  useDeleteDatasetItemsMutation,
  useMoveDatasetItemsMutation,
  useProcessDatasetMutation
} from "@/features/datasets/hooks";
import type { DatasetItemPage, DatasetSummary } from "@/types/api";

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return { Wrapper, queryClient };
}

function fakeDataset(overrides: Partial<DatasetSummary> = {}): DatasetSummary {
  return {
    id: "ds-1",
    name: "Dataset 1",
    task_type: "classification",
    editable: true,
    labels: ["cat", "dog"],
    counts: { train: 0, val: 0, test: 0, unassigned: 0 },
    ...overrides
  } as DatasetSummary;
}

function fakeItemPage(overrides: Partial<DatasetItemPage> = {}): DatasetItemPage {
  return { items: [], total: 0, limit: 50, offset: 0, ...overrides };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("dataset query hooks: query keys are unchanged", () => {
  it("useDatasetCatalogQuery uses ['dataset-catalog', projectId]", async () => {
    vi.spyOn(api, "datasetCatalog").mockResolvedValue([fakeDataset()]);
    const { Wrapper, queryClient } = createWrapper();

    const { result } = renderHook(() => useDatasetCatalogQuery("proj-1"), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.datasetCatalog).toHaveBeenCalledWith("proj-1");
    expect(queryClient.getQueryCache().find({ queryKey: ["dataset-catalog", "proj-1"] })).toBeTruthy();
  });

  it("useDatasetItemsQuery stays disabled without a datasetId", () => {
    const spy = vi.spyOn(api, "datasetItems").mockResolvedValue(fakeItemPage());
    const { Wrapper } = createWrapper();

    const { result } = renderHook(() => useDatasetItemsQuery(undefined, "train", "cat", 1, 25), {
      wrapper: Wrapper
    });

    expect(result.current.fetchStatus).toBe("idle");
    expect(spy).not.toHaveBeenCalled();
  });

  it("useDatasetItemsQuery uses ['dataset-items', datasetId, split, classFilter, imagePage, imagesPerPage] and forwards filters/pagination to the API", async () => {
    vi.spyOn(api, "datasetItems").mockResolvedValue(fakeItemPage());
    const { Wrapper, queryClient } = createWrapper();

    const { result } = renderHook(() => useDatasetItemsQuery("ds-1", "train", "cat", 1, 25), {
      wrapper: Wrapper
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.datasetItems).toHaveBeenCalledWith("ds-1", {
      split: "train",
      class_name: "cat",
      unlabeled: false,
      limit: 25,
      offset: 25
    });
    expect(
      queryClient.getQueryCache().find({ queryKey: ["dataset-items", "ds-1", "train", "cat", 1, 25] })
    ).toBeTruthy();
  });

  it("useDatasetItemDetailQuery uses ['dataset-item', datasetId, split, itemId] and respects the enabled flag", async () => {
    const detailSpy = vi.spyOn(api, "datasetItem");
    const { Wrapper } = createWrapper();

    const { result } = renderHook(() => useDatasetItemDetailQuery("ds-1", "train", "item-1", false), {
      wrapper: Wrapper
    });

    expect(result.current.fetchStatus).toBe("idle");
    expect(detailSpy).not.toHaveBeenCalled();
  });

  it("useDatasetVersionsQuery uses ['dataset-versions', datasetId]", async () => {
    vi.spyOn(api, "datasetVersions").mockResolvedValue([]);
    const { Wrapper, queryClient } = createWrapper();

    const { result } = renderHook(() => useDatasetVersionsQuery("ds-1"), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(queryClient.getQueryCache().find({ queryKey: ["dataset-versions", "ds-1"] })).toBeTruthy();
  });

  it("useDatasetEdaQuery uses ['dataset-eda', datasetId, split]", async () => {
    vi.spyOn(api, "datasetEda").mockResolvedValue({} as never);
    const { Wrapper, queryClient } = createWrapper();

    const { result } = renderHook(() => useDatasetEdaQuery("ds-1", "valid"), { wrapper: Wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(queryClient.getQueryCache().find({ queryKey: ["dataset-eda", "ds-1", "valid"] })).toBeTruthy();
  });
});

describe("dataset mutation hooks: onSuccess side effects", () => {
  it("useCreateDatasetMutation opens the new dataset, closes the create panel, clears the name, and refetches the catalog", async () => {
    const dataset = fakeDataset({ id: "new-ds" });
    vi.spyOn(api, "createDataset").mockResolvedValue(dataset);
    const { Wrapper } = createWrapper();
    const openDataset = vi.fn();
    const setShowCreate = vi.fn();
    const setNewDatasetName = vi.fn();
    const refetch = vi.fn().mockResolvedValue(undefined);

    const { result } = renderHook(
      () =>
        useCreateDatasetMutation({
          openDataset,
          setShowCreate,
          setNewDatasetName,
          catalogQuery: { refetch } as never
        }),
      { wrapper: Wrapper }
    );

    result.current.mutate({ name: "New", task_type: "classification", labels: [] } as never);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(openDataset).toHaveBeenCalledWith("new-ds");
    expect(setShowCreate).toHaveBeenCalledWith(false);
    expect(setNewDatasetName).toHaveBeenCalledWith("");
    expect(refetch).toHaveBeenCalled();
  });

  it("useDeleteDatasetItemsMutation groups selected items by their real split and resets selection on success", async () => {
    vi.spyOn(api, "deleteDatasetItems").mockResolvedValue({ deleted: 1, missing: [] } as never);
    const { Wrapper } = createWrapper();
    const setSelectedItemIds = vi.fn();
    const setSelectedItemSplits = vi.fn();
    const setSelectedItemId = vi.fn();
    const setSelectedItemSplit = vi.fn();
    const refetch = vi.fn().mockResolvedValue(undefined);
    const noopQuery = { refetch } as never;

    const { result } = renderHook(
      () =>
        useDeleteDatasetItemsMutation({
          setSelectedItemIds,
          setSelectedItemSplits,
          setSelectedItemId,
          setSelectedItemSplit,
          itemsQuery: noopQuery,
          catalogQuery: noopQuery,
          edaQuery: noopQuery
        }),
      { wrapper: Wrapper }
    );

    result.current.mutate({
      datasetId: "ds-1",
      items: [
        { id: "a", split: "train" },
        { id: "b", split: "valid" }
      ]
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.deleteDatasetItems).toHaveBeenCalledWith("ds-1", { split: "train", ids: ["a"] });
    expect(api.deleteDatasetItems).toHaveBeenCalledWith("ds-1", { split: "valid", ids: ["b"] });
    expect(setSelectedItemIds).toHaveBeenCalledWith([]);
    expect(setSelectedItemSplits).toHaveBeenCalledWith({});
    expect(setSelectedItemId).toHaveBeenCalledWith("");
    expect(setSelectedItemSplit).toHaveBeenCalledWith("");
    expect(refetch).toHaveBeenCalledTimes(3);
  });

  it("useMoveDatasetItemsMutation groups by source split, skips items already at the target, and clears selection on success", async () => {
    vi.spyOn(api, "moveDatasetItems").mockResolvedValue({ moved: 1, missing: [], items: [] } as never);
    const { Wrapper } = createWrapper();
    const setSelectedItemIds = vi.fn();
    const setSelectedItemSplits = vi.fn();
    const refetch = vi.fn().mockResolvedValue(undefined);
    const noopQuery = { refetch } as never;

    const { result } = renderHook(
      () =>
        useMoveDatasetItemsMutation({
          setSelectedItemIds,
          setSelectedItemSplits,
          itemsQuery: noopQuery,
          catalogQuery: noopQuery,
          edaQuery: noopQuery
        }),
      { wrapper: Wrapper }
    );

    result.current.mutate({
      datasetId: "ds-1",
      items: [
        { id: "x", split: "unassigned" },
        { id: "y", split: "train" }
      ],
      target: "train"
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.moveDatasetItems).toHaveBeenCalledTimes(1);
    expect(api.moveDatasetItems).toHaveBeenCalledWith("ds-1", {
      source_split: "unassigned",
      target_split: "train",
      ids: ["x"]
    });
    expect(setSelectedItemIds).toHaveBeenCalledWith([]);
    expect(setSelectedItemSplits).toHaveBeenCalledWith({});
  });

  it("useBulkLabelDatasetItemsMutation groups by split and refetches items, detail, catalog, and eda queries on success", async () => {
    vi.spyOn(api, "updateDatasetItemLabels").mockResolvedValue({ updated: 1, missing: [], items: [] } as never);
    const { Wrapper } = createWrapper();
    const itemsRefetch = vi.fn().mockResolvedValue(undefined);
    const detailRefetch = vi.fn().mockResolvedValue(undefined);
    const catalogRefetch = vi.fn().mockResolvedValue(undefined);
    const edaRefetch = vi.fn().mockResolvedValue(undefined);

    const { result } = renderHook(
      () =>
        useBulkLabelDatasetItemsMutation({
          itemsQuery: { refetch: itemsRefetch } as never,
          detailQuery: { refetch: detailRefetch } as never,
          catalogQuery: { refetch: catalogRefetch } as never,
          edaQuery: { refetch: edaRefetch } as never
        }),
      { wrapper: Wrapper }
    );

    result.current.mutate({ datasetId: "ds-1", items: [{ id: "a", split: "train" }], classId: 1 });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(api.updateDatasetItemLabels).toHaveBeenCalledWith("ds-1", "train", { ids: ["a"], class_id: 1 });
    expect(itemsRefetch).toHaveBeenCalled();
    expect(detailRefetch).toHaveBeenCalled();
    expect(catalogRefetch).toHaveBeenCalled();
    expect(edaRefetch).toHaveBeenCalled();
  });

  it("useProcessDatasetMutation applies the processed dataset's preprocess/split config, resets to the train split, and clears selection", async () => {
    const processedDataset = fakeDataset({ id: "ds-1" });
    vi.spyOn(api, "processDataset").mockResolvedValue({ dataset: processedDataset } as never);
    const { Wrapper } = createWrapper();
    const setPreprocessConfig = vi.fn();
    const setSplitConfig = vi.fn();
    const setSplit = vi.fn();
    const setSelectedItemIds = vi.fn();
    const setSelectedItemSplits = vi.fn();
    const setSelectedItemId = vi.fn();
    const setSelectedItemSplit = vi.fn();
    const refetch = vi.fn().mockResolvedValue(undefined);
    const noopQuery = { refetch } as never;

    const { result } = renderHook(
      () =>
        useProcessDatasetMutation({
          setPreprocessConfig,
          setSplitConfig,
          setSplit,
          setSelectedItemIds,
          setSelectedItemSplits,
          setSelectedItemId,
          setSelectedItemSplit,
          catalogQuery: noopQuery,
          itemsQuery: noopQuery,
          edaQuery: noopQuery
        }),
      { wrapper: Wrapper }
    );

    result.current.mutate({
      datasetId: "ds-1",
      preprocess: {} as never,
      split: {} as never
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(setSplit).toHaveBeenCalledWith("train");
    expect(setSelectedItemIds).toHaveBeenCalledWith([]);
    expect(setSelectedItemSplits).toHaveBeenCalledWith({});
    expect(setPreprocessConfig).toHaveBeenCalled();
    expect(setSplitConfig).toHaveBeenCalled();
    expect(refetch).toHaveBeenCalledTimes(3);
  });
});
