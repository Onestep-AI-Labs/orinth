"use client";

import { useRef, useState } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, FileJson, Plus, Trash2, Upload, X } from "lucide-react";
import { SPLIT_FILTERS } from "@/features/platform/constants";
import { DatasetRecordEditor, recordPreviewHeadings } from "@/features/datasets/record-editor";
import type {
  useCreateDatasetRecordMutation,
  useDeleteDatasetItemsMutation,
  useSaveDatasetRecordMutation,
  useUploadDatasetRecordsMutation
} from "@/features/datasets/hooks";
import { Badge, EmptyState, InlineSpinner } from "@/features/platform/ui";
import type {
  DatasetItemDetail,
  DatasetItemPage,
  DatasetItemSummary,
  DatasetSplitFilter,
  DatasetSummary,
  SplitKey
} from "@/types/api";

/**
 * Records tab for `llm_finetune` datasets (phase 10): a paginated table with an
 * edit drawer that is role-aware for chat records and three labeled fields for
 * instruction records. Replaces the Images/Annotate tabs, which have no meaning
 * for record data.
 */
export function DatasetRecordsTab({
  dataset,
  items,
  itemsQuery,
  itemPage,
  split,
  setSplit,
  imagesPerPage,
  imagePage,
  setImagePage,
  detailQuery,
  onSelectItem,
  selectedItemId,
  createRecordMutation,
  saveRecordMutation,
  uploadRecordsMutation,
  deleteItemsMutation,
  confirm
}: {
  dataset: DatasetSummary;
  items: DatasetItemSummary[];
  itemsQuery: UseQueryResult<DatasetItemPage>;
  itemPage: DatasetItemPage;
  split: DatasetSplitFilter;
  setSplit: (value: DatasetSplitFilter) => void;
  imagesPerPage: number;
  imagePage: number;
  setImagePage: (value: number) => void;
  detailQuery: UseQueryResult<DatasetItemDetail>;
  onSelectItem: (item: DatasetItemSummary) => void;
  selectedItemId: string;
  createRecordMutation: ReturnType<typeof useCreateDatasetRecordMutation>;
  saveRecordMutation: ReturnType<typeof useSaveDatasetRecordMutation>;
  uploadRecordsMutation: ReturnType<typeof useUploadDatasetRecordsMutation>;
  deleteItemsMutation: ReturnType<typeof useDeleteDatasetItemsMutation>;
  confirm: (options: { title: string; message: string; confirmLabel: string; onConfirm: () => void }) => void;
}) {
  const isChat = dataset.format === "chat_jsonl";
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [mode, setMode] = useState<"create" | "edit">("edit");

  const lastPage = Math.max(0, Math.ceil(itemPage.total / imagesPerPage) - 1);
  const [firstHeading, secondHeading] = recordPreviewHeadings(items, isChat);

  function openEdit(item: DatasetItemSummary) {
    setMode("edit");
    setDrawerOpen(true);
    onSelectItem(item);
  }

  function openCreate() {
    setMode("create");
    setDrawerOpen(true);
  }

  function saveDraft(record: Record<string, unknown>, draftSplit: SplitKey) {
    if (mode === "create") {
      createRecordMutation.mutate(
        { datasetId: dataset.id, split: draftSplit, record },
        { onSuccess: () => setDrawerOpen(false) }
      );
    } else if (selectedItemId && detailQuery.data) {
      saveRecordMutation.mutate(
        { datasetId: dataset.id, split: detailQuery.data.split, itemId: selectedItemId, record },
        { onSuccess: () => setDrawerOpen(false) }
      );
    }
  }

  function deleteRecord(item: DatasetItemSummary) {
    confirm({
      title: "Delete record?",
      message: "This removes the record and rewrites the split's data.jsonl. This cannot be undone.",
      confirmLabel: "Delete record",
      onConfirm: () => deleteItemsMutation.mutate({ datasetId: dataset.id, items: [{ id: item.id, split: item.split }] })
    });
  }

  function uploadFile(file: File | undefined) {
    if (!file) return;
    const form = new FormData();
    form.append("split", split === "all" ? "unassigned" : split);
    form.append("file", file);
    uploadRecordsMutation.mutate({ datasetId: dataset.id, form });
  }

  const pending = createRecordMutation.isPending || saveRecordMutation.isPending;

  return (
    <section className="panel records-panel">
      <div className="records-toolbar">
        <div className="segmented-control">
          {SPLIT_FILTERS.map((filter) => (
            <button
              key={filter}
              type="button"
              className={split === filter ? "segmented-active" : ""}
              onClick={() => setSplit(filter)}
            >
              {filter}
            </button>
          ))}
        </div>
        <div className="records-toolbar-actions">
          {(itemsQuery.isFetching || uploadRecordsMutation.isPending) && (
            <InlineSpinner label={uploadRecordsMutation.isPending ? "Uploading" : "Refreshing"} />
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".jsonl,.json,.csv"
            hidden
            onChange={(event) => {
              uploadFile(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
          <button
            className="secondary-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadRecordsMutation.isPending}
            title="Upload a .jsonl, .json, or .csv file of records"
          >
            <Upload size={16} /> Upload file
          </button>
          <button className="primary-button" onClick={openCreate}>
            <Plus size={16} /> New record
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <EmptyState
          label="No records yet."
          icon={<FileJson size={28} />}
          centered
          description="Add a record by hand, upload a .jsonl/.csv file, or import from the HuggingFace Hub."
        />
      ) : (
        <div className="table-wrap">
          <table className="records-table">
            <thead>
              <tr>
                <th>{firstHeading}</th>
                <th>{secondHeading}</th>
                <th className="records-col-num">Tokens</th>
                <th className="records-col-split">Split</th>
                <th className="records-col-actions" aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={`${item.split}-${item.id}`} onClick={() => openEdit(item)} className="records-row">
                  <td className="records-cell-excerpt">{item.text_preview || "—"}</td>
                  <td className="records-cell-excerpt">{item.output_preview || "—"}</td>
                  <td className="records-col-num">{item.token_estimate}</td>
                  <td className="records-col-split">
                    <Badge tone="neutral">{item.split}</Badge>
                  </td>
                  <td className="records-col-actions">
                    <button
                      className="icon-button"
                      title="Delete record"
                      onClick={(event) => {
                        event.stopPropagation();
                        deleteRecord(item);
                      }}
                    >
                      <Trash2 size={15} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {itemPage.total > imagesPerPage && (
        <div className="records-pager">
          <button className="secondary-button" onClick={() => setImagePage(Math.max(0, imagePage - 1))} disabled={imagePage <= 0}>
            <ChevronLeft size={15} /> Prev
          </button>
          <span>
            Page {imagePage + 1} of {lastPage + 1} — {itemPage.total} records
          </span>
          <button className="secondary-button" onClick={() => setImagePage(Math.min(lastPage, imagePage + 1))} disabled={imagePage >= lastPage}>
            Next <ChevronRight size={15} />
          </button>
        </div>
      )}

      {drawerOpen && (
        <div className="record-drawer" role="dialog" aria-label="Edit record">
          <div className="record-drawer-header">
            <strong>{mode === "create" ? "New record" : "Edit record"}</strong>
            <button className="icon-button" onClick={() => setDrawerOpen(false)} title="Close">
              <X size={16} />
            </button>
          </div>
          <DatasetRecordEditor
            dataset={dataset}
            mode={mode}
            record={detailQuery.data?.record ?? null}
            loading={detailQuery.isLoading}
            defaultSplit={split === "all" ? "unassigned" : (split as SplitKey)}
            pending={pending}
            onSave={saveDraft}
            onCancel={() => setDrawerOpen(false)}
          />
        </div>
      )}
    </section>
  );
}
