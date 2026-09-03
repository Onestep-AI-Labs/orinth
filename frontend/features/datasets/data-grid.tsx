"use client";

/* eslint-disable @next/next/no-img-element */

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Table2 } from "lucide-react";
import { api } from "@/lib/api";
import { toast } from "@/features/platform/toast";
import { SPLIT_FILTERS, TRAINING_SPLITS } from "@/features/platform/constants";
import {
  Badge,
  EmptyState,
  InlineSpinner,
  PanelTitle,
  Select,
  TableSkeleton
} from "@/features/platform/ui";
import type {
  DatasetSplitFilter,
  DatasetSummary,
  DatasetTableColumn,
  DatasetTableRow,
  SplitKey
} from "@/types/api";

/**
 * The dataset as a spreadsheet: every modality, one table.
 *
 * The studio had three browsers — a thumbnail wall, a text preview list, and a
 * records table — which meant the answer to "what is actually in this data"
 * depended on which task you happened to have. Kaggle and the Hugging Face
 * viewer both answer it the same way for everything, and there is no reason a
 * folder of images cannot be a row per file with the thumbnail in the first
 * column.
 *
 * Only the cells with a real write route behind them are editable: the label
 * (`PATCH …/label`) and the split (`POST …/items/move`). A cell that looks
 * editable and silently discards the edit is worse than a read-only one, which
 * is why `editable` is decided on the server beside the column definition rather
 * than guessed here.
 *
 * Editing is optimistic in neither direction: the mutation invalidates and the
 * row re-reads. At a page of fifty that is one cheap request, and it means the
 * grid never shows a value the backend rejected.
 */

const PAGE_SIZE = 50;

function CellValue({
  column,
  row,
  labels,
  onLabel,
  onSplit,
  disabled
}: {
  column: DatasetTableColumn;
  row: DatasetTableRow;
  labels: string[];
  onLabel: (row: DatasetTableRow, label: string) => void;
  onSplit: (row: DatasetTableRow, split: SplitKey) => void;
  disabled: boolean;
}) {
  const value = row.cells[column.key];

  if (column.kind === "image") {
    const url = typeof value === "string" ? value : "";
    if (!url) return <span className="grid-cell-empty">—</span>;
    return <img className="grid-thumb" src={url} alt="" loading="lazy" />;
  }

  if (column.kind === "split") {
    return (
      <Select
        className="grid-cell-select"
        value={row.split}
        disabled={disabled}
        onChange={(event) => onSplit(row, event.target.value as SplitKey)}
        aria-label="Split"
      >
        {(["unassigned", ...TRAINING_SPLITS] as SplitKey[]).map((split) => (
          <option key={split} value={split}>
            {split}
          </option>
        ))}
      </Select>
    );
  }

  if (column.kind === "label" && column.editable) {
    // Labels are a closed set, so this is a picker, not a text field — a typo
    // in a class name creates a class, and doing that by accident in a grid is
    // far too easy.
    return (
      <Select
        className="grid-cell-select"
        value={typeof value === "string" ? value : ""}
        disabled={disabled || labels.length === 0}
        onChange={(event) => onLabel(row, event.target.value)}
        aria-label="Label"
      >
        <option value="">unlabeled</option>
        {labels.map((label) => (
          <option key={label} value={label}>
            {label}
          </option>
        ))}
      </Select>
    );
  }

  if (column.kind === "number") {
    return <span className="grid-cell-number">{typeof value === "number" ? value : 0}</span>;
  }

  const text = value == null || value === "" ? "" : String(value);
  if (!text) return <span className="grid-cell-empty">—</span>;
  // `title` carries the untruncated value: a cell is one line so the table stays
  // scannable, and the full text is one hover away rather than one click.
  return (
    <span className="grid-cell-text" title={text}>
      {text}
    </span>
  );
}

export function DatasetDataGrid({ dataset }: { dataset: DatasetSummary }) {
  const client = useQueryClient();
  const [split, setSplit] = useState<DatasetSplitFilter>("all");
  const [page, setPage] = useState(0);

  const tableQuery = useQuery({
    queryKey: ["dataset-table", dataset.id, split, page],
    queryFn: () =>
      api.datasetTable(dataset.id, { split, limit: PAGE_SIZE, offset: page * PAGE_SIZE })
  });

  const refresh = () =>
    Promise.all([
      client.invalidateQueries({ queryKey: ["dataset-table", dataset.id] }),
      client.invalidateQueries({ queryKey: ["dataset-items"] }),
      client.invalidateQueries({ queryKey: ["dataset-catalog"] })
    ]);

  const labelMutation = useMutation({
    mutationFn: ({ row, label }: { row: DatasetTableRow; label: string }) =>
      api.updateDatasetItemLabel(dataset.id, row.split, row.id, {
        class_id: dataset.labels.indexOf(label),
        class_name: label
      }),
    onSuccess: refresh,
    onError: (error: Error) => toast.error(error.message)
  });

  const splitMutation = useMutation({
    mutationFn: ({ row, target }: { row: DatasetTableRow; target: SplitKey }) =>
      api.moveDatasetItems(dataset.id, {
        ids: [row.id],
        source_split: row.split,
        target_split: target
      }),
    onSuccess: refresh,
    onError: (error: Error) => toast.error(error.message)
  });

  const pending = labelMutation.isPending || splitMutation.isPending;
  const table = tableQuery.data;
  const total = table?.total ?? 0;
  const lastPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);
  const columns = useMemo(() => table?.columns ?? [], [table]);

  return (
    <section className="panel">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <PanelTitle icon={<Table2 size={18} />} title="Data" />
        <div className="flex flex-wrap items-center gap-2">
          {(tableQuery.isFetching || pending) && <InlineSpinner label="Saving" />}
          <Badge tone="neutral">{total} rows</Badge>
          <Select
            value={split}
            onChange={(event) => {
              setSplit(event.target.value as DatasetSplitFilter);
              setPage(0);
            }}
            aria-label="Split filter"
          >
            {SPLIT_FILTERS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </Select>
        </div>
      </div>

      {tableQuery.isLoading ? (
        <TableSkeleton rows={8} />
      ) : total === 0 ? (
        <EmptyState
          icon={<Table2 size={26} />}
          label="Nothing in this split"
          description="Upload data, or run Orinth to distribute what is already here into train, valid and test."
        />
      ) : (
        // Wide tables scroll inside their own container; the page body never
        // scrolls sideways (DESIGN.md §8).
        <div className="grid-scroll">
          <table className="data-grid">
            <thead>
              <tr>
                {columns.map((column) => (
                  <th key={column.key} scope="col">
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(table?.rows ?? []).map((row) => (
                <tr key={`${row.split}/${row.id}`}>
                  {columns.map((column) => (
                    <td key={column.key} className={`grid-cell-${column.kind}`}>
                      <CellValue
                        column={column}
                        row={row}
                        labels={dataset.labels}
                        disabled={pending || !dataset.editable}
                        onLabel={(target, label) =>
                          labelMutation.mutate({ row: target, label })
                        }
                        onSplit={(target, next) =>
                          splitMutation.mutate({ row: target, target: next })
                        }
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {lastPage > 0 && (
        <div className="grid-pager">
          <button
            className="secondary-button"
            type="button"
            onClick={() => setPage((value) => Math.max(0, value - 1))}
            disabled={page === 0}
          >
            <ChevronLeft size={16} /> Previous
          </button>
          <span className="form-caption">
            {page * PAGE_SIZE + 1}–{Math.min(total, (page + 1) * PAGE_SIZE)} of {total}
          </span>
          <button
            className="secondary-button"
            type="button"
            onClick={() => setPage((value) => Math.min(lastPage, value + 1))}
            disabled={page >= lastPage}
          >
            Next <ChevronRight size={16} />
          </button>
        </div>
      )}
    </section>
  );
}
