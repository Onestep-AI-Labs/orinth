"use client";

import type { Dispatch, SetStateAction } from "react";
import { BarChart3, CheckCircle2, FileJson, ImageIcon, type LucideIcon, RefreshCw } from "lucide-react";
import { AnnotationEditor as DatasetAnnotationEditor } from "@/features/datasets/annotation-editor";
import { DatasetRecordEditor, recordPreviewHeadings } from "@/features/datasets/record-editor";
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
  useDatasetItemsQuery,
  useSaveDatasetRecordMutation
} from "@/features/datasets/hooks";
import { CardGridSkeleton, EmptyState, MutationError, PanelTitle, Select } from "@/features/platform/ui";
import type { useConfirmationDialog } from "@/features/platform/ui";
import type { DatasetItemPage, DatasetItemSummary, DatasetSplitFilter, DatasetSummary, SplitKey } from "@/types/api";

/**
 * Annotate view: browse on the left, edit the selected item on the right.
 *
 * The right column is what the dataset's own kind makes of "annotate". Vision
 * and NLP get the label manager over the `AnnotationEditor`. An `llm_finetune`
 * dataset gets the record editor instead — its records carry no regions and no
 * class list, so a label manager there would manage nothing and the vision
 * editor would draw a canvas over an image that does not exist. Same shape,
 * honest contents; the alternative was to keep hiding the view from LLM
 * datasets, which left the one modality whose items are pure text without the
 * screen for editing text.
 *
 * Mutations/queries stay owned by DatasetPage; only the bulk-label
 * confirmation handler (specific to this view) lives here.
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
  isLlm = false,
  saveRecordMutation,
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
  isLlm?: boolean;
  saveRecordMutation?: ReturnType<typeof useSaveDatasetRecordMutation>;
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

  const [promptHeading, outputHeading] = recordPreviewHeadings(items, dataset.format === "chat_jsonl");

  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_440px]">
      <section className="panel dataset-work-panel">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <PanelTitle
            icon={<ItemIcon size={18} />}
            title={isLlm ? "Annotate Records" : isNlp ? "Annotate Texts" : "Annotate Images"}
          />
          <button className="icon-button" onClick={() => itemsQuery.refetch()} title="Refresh">
            <RefreshCw size={16} />
          </button>
        </div>
        <DatasetSummaryBar dataset={dataset} split={split} setSplit={setSplit} />
        {/* A record has no class, so the chip row would filter by a set of one
            ("All"), and the bulk bar would select for an action that does not
            apply to records. */}
        {!isLlm && <LabelFilterChips dataset={dataset} value={classFilter} onChange={setClassFilter} />}
        {!isLlm && (
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
                <Select value={bulkClassId} onChange={(event) => setBulkClassId(Number(event.target.value))} disabled={selectedItems.length === 0}>
                  {dataset.labels.map((label, index) => (
                    <option value={index} key={label}>{label}</option>
                  ))}
                </Select>
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
        )}
        {itemsQuery.isLoading && <CardGridSkeleton count={6} />}
        {/* Two columns of prose need naming, and the fields they were read from
            are the honest names — the same heads the Records table carries. */}
        {isLlm && items.length > 0 && (
          <div className="record-list-head" aria-hidden="true">
            <span>{promptHeading}</span>
            <span>{outputHeading}</span>
            <span className="record-list-head-meta">Tokens</span>
          </div>
        )}
        <div className={isLlm ? "record-item-list" : isNlp ? "text-item-list" : "image-grid"}>
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
          {items.length === 0 && (
            <EmptyState label={isLlm ? "No records" : isNlp ? "No texts" : "No images"} />
          )}
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

      {isLlm ? (
        <section className="panel annotate-record-panel">
          {detailQuery.isLoading ? (
            <>
              <PanelTitle icon={<FileJson size={18} />} title="Record" />
              <CardGridSkeleton count={1} />
            </>
          ) : detailQuery.data ? (
            <DatasetRecordEditor
              dataset={dataset}
              mode="edit"
              record={detailQuery.data.record ?? null}
              loading={detailQuery.isFetching}
              pending={saveRecordMutation?.isPending ?? false}
              header={<PanelTitle icon={<FileJson size={18} />} title="Record" />}
              onSave={(record) => {
                if (!detailQuery.data || !saveRecordMutation) return;
                saveRecordMutation.mutate({
                  datasetId: dataset.id,
                  split: detailQuery.data.split,
                  itemId: detailQuery.data.id,
                  record
                });
              }}
            />
          ) : (
            <>
              <PanelTitle icon={<FileJson size={18} />} title="Record" />
              <EmptyState label="No record selected" />
            </>
          )}
          <MutationError mutations={saveRecordMutation ? [saveRecordMutation] : []} />
        </section>
      ) : (
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
      )}
    </div>
  );
}
