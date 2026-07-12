"use client";

import type { Dispatch, SetStateAction } from "react";
import { Copy, type LucideIcon, RefreshCw, Trash2 } from "lucide-react";
import {
  DatasetPagination,
  DatasetSummaryBar,
  DatasetThumb,
  LabelFilterChips
} from "@/features/datasets/dataset-components";
import type {
  useDatasetItemsQuery,
  useDeleteDatasetItemsMutation,
  useMoveDatasetItemsMutation,
  useUploadDatasetMutation
} from "@/features/datasets/hooks";
import { DatasetUploadPanel } from "@/features/datasets/upload-panel";
import { SPLITS } from "@/features/platform/constants";
import { CardGridSkeleton, EmptyState, MutationError, PanelTitle } from "@/features/platform/ui";
import type { useConfirmationDialog } from "@/features/platform/ui";
import type { DatasetItemPage, DatasetItemSummary, DatasetSplitFilter, DatasetSummary, SplitKey } from "@/types/api";

/**
 * Images/texts tab: upload panel, bulk select/move/remove bar, item
 * grid, and pagination. Mutations, query, and selection state stay
 * owned by DatasetPage; this component owns the move/delete
 * confirmation handlers since they only apply within this tab.
 */
export function DatasetImagesTab({
  dataset,
  itemIcon: ItemIcon,
  isNlp,
  itemsQuery,
  split,
  setSplit,
  classFilter,
  setClassFilter,
  uploadFiles,
  setUploadFiles,
  uploadClassId,
  setUploadClassId,
  uploadErrors,
  uploadMutation,
  items,
  itemPage,
  imagesPerPage,
  setImagesPerPage,
  imagePage,
  setImagePage,
  selectedItemId,
  selectedItemSplit,
  selectedItemIds,
  setSelectedItemIds,
  onSelectItem,
  moveTarget,
  setMoveTarget,
  moveItemsMutation,
  deleteItemsMutation,
  confirm
}: {
  dataset: DatasetSummary;
  itemIcon: LucideIcon;
  isNlp: boolean;
  itemsQuery: ReturnType<typeof useDatasetItemsQuery>;
  split: DatasetSplitFilter;
  setSplit: (value: DatasetSplitFilter) => void;
  classFilter: string;
  setClassFilter: (value: string) => void;
  uploadFiles: File[];
  setUploadFiles: (files: File[]) => void;
  uploadClassId: number;
  setUploadClassId: (value: number) => void;
  uploadErrors: Array<Record<string, string>>;
  uploadMutation: ReturnType<typeof useUploadDatasetMutation>;
  items: DatasetItemSummary[];
  itemPage: DatasetItemPage;
  imagesPerPage: number;
  setImagesPerPage: (value: number) => void;
  imagePage: number;
  setImagePage: (value: number) => void;
  selectedItemId: string;
  selectedItemSplit: SplitKey | "";
  selectedItemIds: string[];
  setSelectedItemIds: Dispatch<SetStateAction<string[]>>;
  onSelectItem: (item: DatasetItemSummary, openAnnotate?: boolean) => void;
  moveTarget: SplitKey;
  setMoveTarget: (value: SplitKey) => void;
  moveItemsMutation: ReturnType<typeof useMoveDatasetItemsMutation>;
  deleteItemsMutation: ReturnType<typeof useDeleteDatasetItemsMutation>;
  confirm: ReturnType<typeof useConfirmationDialog>["confirm"];
}) {
  function confirmMoveSelectedImages() {
    if (selectedItemIds.length === 0) return;
    confirm({
      title: `Move selected ${isNlp ? "texts" : "images"}?`,
      message: `This will move ${selectedItemIds.length} selected item${selectedItemIds.length === 1 ? "" : "s"} from ${split} to ${moveTarget}.`,
      confirmLabel: "Move items",
      tone: "warning",
      onConfirm: () => moveItemsMutation.mutate({ datasetId: dataset.id, target: moveTarget })
    });
  }

  function confirmDeleteSelectedImages() {
    if (selectedItemIds.length === 0) return;
    confirm({
      title: `Remove selected ${isNlp ? "texts" : "images"}?`,
      message: `This will delete ${selectedItemIds.length} selected item${selectedItemIds.length === 1 ? "" : "s"} from "${dataset.name}". This action cannot be undone.`,
      confirmLabel: "Remove items",
      onConfirm: () => deleteItemsMutation.mutate({ datasetId: dataset.id, ids: selectedItemIds })
    });
  }

  return (
    <section className="panel dataset-work-panel">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <PanelTitle icon={<ItemIcon size={18} />} title={isNlp ? "Texts" : "Images"} />
        <button className="icon-button" onClick={() => itemsQuery.refetch()} title="Refresh">
          <RefreshCw size={16} />
        </button>
      </div>
      <DatasetSummaryBar dataset={dataset} split={split} setSplit={setSplit} />
      <LabelFilterChips dataset={dataset} value={classFilter} onChange={setClassFilter} />
      <DatasetUploadPanel
        dataset={dataset}
        split={split}
        files={uploadFiles}
        setFiles={setUploadFiles}
        uploadClassId={uploadClassId}
        setUploadClassId={setUploadClassId}
        errors={uploadErrors}
        uploadMutation={uploadMutation}
      />
      <div className="bulk-bar">
        <label className="check-row">
          <input
            type="checkbox"
            checked={items.length > 0 && selectedItemIds.length === items.length}
            onChange={(event) => setSelectedItemIds(event.target.checked ? items.map((item) => item.id) : [])}
          />
          <span>{selectedItemIds.length ? `${selectedItemIds.length} selected` : `Select ${isNlp ? "texts" : "images"}`}</span>
        </label>
        {dataset.editable && split !== "all" && (
          <>
            <select value={moveTarget} onChange={(event) => setMoveTarget(event.target.value as SplitKey)} disabled={selectedItemIds.length === 0}>
              {SPLITS.filter((splitName) => splitName !== split).map((splitName) => (
                <option value={splitName} key={splitName}>{splitName}</option>
              ))}
            </select>
            <button
              className="secondary-button"
              onClick={confirmMoveSelectedImages}
              disabled={selectedItemIds.length === 0 || moveItemsMutation.isPending || moveTarget === split}
            >
              <Copy size={16} /> Move
            </button>
            <button
              className="danger-button"
              onClick={confirmDeleteSelectedImages}
              disabled={selectedItemIds.length === 0 || deleteItemsMutation.isPending}
            >
              <Trash2 size={16} /> Remove
            </button>
          </>
        )}
      </div>
      {itemsQuery.isLoading && <CardGridSkeleton count={6} />}
      <div className={isNlp ? "text-item-list" : "image-grid"}>
        {items.map((item) => (
          <DatasetThumb
            item={item}
            key={`${item.split}-${item.id}`}
            active={item.id === selectedItemId && item.split === selectedItemSplit}
            onClick={() => onSelectItem(item, true)}
            selected={selectedItemIds.includes(item.id)}
            onSelected={(checked) =>
              setSelectedItemIds((ids) => checked ? [...new Set([...ids, item.id])] : ids.filter((id) => id !== item.id))
            }
          />
        ))}
        {items.length === 0 && <EmptyState label={isNlp ? "No texts" : "No images"} />}
      </div>
      <DatasetPagination
        total={itemPage.total}
        offset={itemPage.offset}
        limit={imagesPerPage}
        onLimitChange={setImagesPerPage}
        onPageChange={setImagePage}
      />
      <MutationError mutations={[uploadMutation, deleteItemsMutation, moveItemsMutation]} />
    </section>
  );
}
