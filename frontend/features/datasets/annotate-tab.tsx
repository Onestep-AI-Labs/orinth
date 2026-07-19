"use client";

import type { Dispatch, SetStateAction } from "react";
import { BarChart3, CheckCircle2, ImageIcon, type LucideIcon, RefreshCw } from "lucide-react";
import { AnnotationEditor as DatasetAnnotationEditor } from "@/features/datasets/annotation-editor";
import {
  DatasetPagination,
  DatasetSummaryBar,
  DatasetThumb,
  LabelFilterChips,
  LabelManager
} from "@/features/datasets/dataset-components";
import type {
  useBulkLabelDatasetItemsMutation,
  useDatasetCatalogQuery,
  useDatasetEdaQuery,
  useDatasetItemDetailQuery,
  useDatasetItemsQuery
} from "@/features/datasets/hooks";
import { CardGridSkeleton, EmptyState, MutationError, PanelTitle } from "@/features/platform/ui";
import type { useConfirmationDialog } from "@/features/platform/ui";
import type { DatasetItemPage, DatasetItemSummary, DatasetSplitFilter, DatasetSummary, SplitKey } from "@/types/api";

/**
 * Annotate tab: item grid + bulk label editing on the left, label
 * manager and the (out-of-scope) AnnotationEditor on the right.
 * Mutations/queries stay owned by DatasetPage; only the bulk-label
 * confirmation handler (specific to this tab) moves in here.
 */
export function DatasetAnnotateTab({
  dataset,
  itemIcon: ItemIcon,
  isNlp,
  itemsQuery,
  split,
  setSplit,
  classFilter,
  setClassFilter,
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
  selectedItemSplits,
  setSelectedItemSplits,
  onSelectItem,
  bulkClassId,
  setBulkClassId,
  bulkLabelMutation,
  catalogQuery,
  detailQuery,
  edaQuery,
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
  selectedItemSplits: Record<string, SplitKey>;
  setSelectedItemSplits: Dispatch<SetStateAction<Record<string, SplitKey>>>;
  onSelectItem: (item: DatasetItemSummary, openAnnotate?: boolean) => void;
  bulkClassId: number;
  setBulkClassId: (value: number) => void;
  bulkLabelMutation: ReturnType<typeof useBulkLabelDatasetItemsMutation>;
  catalogQuery: ReturnType<typeof useDatasetCatalogQuery>;
  detailQuery: ReturnType<typeof useDatasetItemDetailQuery>;
  edaQuery: ReturnType<typeof useDatasetEdaQuery>;
  confirm: ReturnType<typeof useConfirmationDialog>["confirm"];
}) {
  const selectedItems = selectedItemIds
    .map((id) => ({ id, split: selectedItemSplits[id] }))
    .filter((item): item is { id: string; split: SplitKey } => Boolean(item.split));

  function toggleItemSelected(item: DatasetItemSummary, checked: boolean) {
    setSelectedItemIds((ids) => (checked ? [...new Set([...ids, item.id])] : ids.filter((id) => id !== item.id)));
    setSelectedItemSplits((splits) => {
      if (checked) return { ...splits, [item.id]: item.split };
      const { [item.id]: _removed, ...rest } = splits;
      return rest;
    });
  }

  function toggleAllSelected(checked: boolean) {
    setSelectedItemIds(checked ? items.map((item) => item.id) : []);
    setSelectedItemSplits(checked ? Object.fromEntries(items.map((item) => [item.id, item.split])) : {});
  }

  function confirmBulkLabelImages() {
    if (selectedItems.length === 0) return;
    const label = dataset.labels[bulkClassId] ?? "selected label";
    confirm({
      title: "Edit selected labels?",
      message: `This will set ${selectedItems.length} selected item${selectedItems.length === 1 ? "" : "s"} to "${label}".`,
      confirmLabel: "Edit labels",
      tone: "warning",
      onConfirm: () => bulkLabelMutation.mutate({ datasetId: dataset.id, items: selectedItems, classId: bulkClassId })
    });
  }

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_440px]">
      <section className="panel dataset-work-panel">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <PanelTitle icon={<ItemIcon size={18} />} title={isNlp ? "Annotate Texts" : "Annotate Images"} />
          <button className="icon-button" onClick={() => itemsQuery.refetch()} title="Refresh">
            <RefreshCw size={16} />
          </button>
        </div>
        <DatasetSummaryBar dataset={dataset} split={split} setSplit={setSplit} />
        <LabelFilterChips dataset={dataset} value={classFilter} onChange={setClassFilter} />
        <div className="bulk-bar">
          <label className="check-row">
            <input
              type="checkbox"
              checked={items.length > 0 && selectedItemIds.length === items.length}
              onChange={(event) => toggleAllSelected(event.target.checked)}
            />
            <span>{selectedItemIds.length ? `${selectedItemIds.length} selected` : `Select ${isNlp ? "texts" : "images"}`}</span>
          </label>
          {dataset.editable && (dataset.task_type === "classification" || dataset.task_type === "text_classification") && (
            <>
              <select value={bulkClassId} onChange={(event) => setBulkClassId(Number(event.target.value))} disabled={selectedItems.length === 0}>
                {dataset.labels.map((label, index) => (
                  <option value={index} key={label}>{label}</option>
                ))}
              </select>
              <button
                className="secondary-button"
                onClick={confirmBulkLabelImages}
                disabled={selectedItems.length === 0 || bulkLabelMutation.isPending}
              >
                <CheckCircle2 size={16} /> Edit label
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
              onClick={() => onSelectItem(item)}
              selected={selectedItemIds.includes(item.id)}
              onSelected={(checked) => toggleItemSelected(item, checked)}
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
        <MutationError mutations={[bulkLabelMutation]} />
      </section>

      <section className="panel">
        <PanelTitle icon={<BarChart3 size={18} />} title="Labels" />
        <LabelManager
          dataset={dataset}
          onChanged={async () => {
            await catalogQuery.refetch();
          }}
        />
        <div className="divider" />
        <PanelTitle icon={<ImageIcon size={18} />} title="Annotation" />
        {detailQuery.isLoading ? (
          <CardGridSkeleton count={1} />
        ) : detailQuery.data ? (
          <DatasetAnnotationEditor
            dataset={dataset}
            item={detailQuery.data}
            onSaved={async () => {
              await Promise.all([
                detailQuery.refetch(),
                itemsQuery.refetch(),
                catalogQuery.refetch(),
                edaQuery.refetch()
              ]);
            }}
          />
        ) : (
          <EmptyState label={isNlp ? "No text selected" : "No image selected"} />
        )}
      </section>
    </div>
  );
}
