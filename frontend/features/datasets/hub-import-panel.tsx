"use client";

import { useEffect, useMemo, useState } from "react";
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { CheckCircle2, Cloud, Download, FolderOpen, Search } from "lucide-react";
import { api } from "@/lib/api";
import { useImportHubDatasetMutation } from "@/features/datasets/hooks";
import { formatDatasetFormat } from "@/features/platform/utils";
import { Badge, EmptyState, Field, InlineSpinner, MutationError } from "@/features/platform/ui";
import type { DatasetFormat, DatasetHubSearchResult, DatasetSummary } from "@/types/api";

type Role = { key: "instruction" | "input" | "output" | "messages" | "conversations"; label: string };

const INSTRUCTION_ROLES: Role[] = [
  { key: "instruction", label: "Instruction" },
  { key: "input", label: "Input" },
  { key: "output", label: "Output" }
];
const CHAT_ROLES: Role[] = [
  { key: "messages", label: "Messages" },
  { key: "conversations", label: "Conversations" }
];

/**
 * HuggingFace Hub discovery + import (phase 10). Adopts the Studio Hub anatomy —
 * search plus result list, a preview table whose column headers carry per-column
 * role pickers, a detected-format banner when heuristics succeed, and a footer
 * import gated on mapping completeness — rendered entirely in DESIGN.md tokens.
 */
export function HubImportPanel({
  projectId,
  openDataset,
  catalogQuery,
  onImported
}: {
  projectId: string;
  openDataset: (datasetId: string) => void;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  onImported: () => void;
}) {
  // Studio Hub anatomy: Discover (search + import) vs Imported (what already
  // landed locally — the HF-origin datasets in the catalog).
  const [tab, setTab] = useState<"discover" | "imported">("discover");
  const [searchInput, setSearchInput] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [selected, setSelected] = useState<DatasetHubSearchResult | null>(null);
  const [format, setFormat] = useState<DatasetFormat>("instruction_jsonl");
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [name, setName] = useState("");
  const [maxRows, setMaxRows] = useState(1000);

  const searchQuery = useQuery({
    queryKey: ["hub-search", submitted],
    queryFn: () => api.searchDatasetHub({ query: submitted || undefined, limit: 24 })
  });

  const previewQuery = useQuery({
    queryKey: ["hub-preview", selected?.hub_id],
    queryFn: () => api.previewDatasetHub({ hub_id: selected?.hub_id ?? "", limit: 8 }),
    enabled: Boolean(selected?.hub_id)
  });

  const importMutation = useImportHubDatasetMutation({ openDataset, catalogQuery, onImported });

  // Pre-fill format and column mapping from server-side detection so a
  // recognised dataset (alpaca / sharegpt / messages / qa) imports without
  // manual work; QA columns normalise into the instruction shape.
  useEffect(() => {
    const preview = previewQuery.data;
    if (!preview || preview.error) return;
    const detected = preview.detected_format;
    const detectedMapping = preview.detected_mapping ?? {};
    if (detected === "messages" || detected === "sharegpt") {
      setFormat("chat_jsonl");
      setMapping({
        messages: detectedMapping.messages ?? "",
        conversations: detectedMapping.conversations ?? ""
      });
    } else if (detected === "alpaca" || detected === "qa") {
      setFormat("instruction_jsonl");
      setMapping({
        instruction: detectedMapping.instruction ?? detectedMapping.question ?? "",
        input: detectedMapping.input ?? detectedMapping.context ?? "",
        output: detectedMapping.output ?? detectedMapping.answer ?? ""
      });
    } else {
      setMapping({});
    }
    setName(selected ? selected.hub_id.split("/").pop() ?? selected.hub_id : "");
  }, [previewQuery.data, selected]);

  const roles = format === "chat_jsonl" ? CHAT_ROLES : INSTRUCTION_ROLES;
  const columns = previewQuery.data?.columns ?? [];

  const roleForColumn = (column: string): string =>
    roles.find((role) => mapping[role.key] === column)?.key ?? "";

  function assignColumn(column: string, roleKey: string) {
    setMapping((current) => {
      const next: Record<string, string> = {};
      // A role maps to exactly one column: clear any prior owner of this role,
      // and clear this column from whatever role it previously held.
      for (const [key, value] of Object.entries(current)) {
        if (value !== column) next[key] = value;
      }
      if (roleKey) next[roleKey] = column;
      return next;
    });
  }

  const mappingComplete =
    format === "chat_jsonl"
      ? Boolean(mapping.messages || mapping.conversations)
      : Boolean(mapping.instruction && mapping.output);

  const detectionMatched = useMemo(() => {
    const detected = previewQuery.data?.detected_format;
    return Boolean(detected) && mappingComplete;
  }, [mappingComplete, previewQuery.data?.detected_format]);

  function submitSearch() {
    setSubmitted(searchInput.trim());
    setSelected(null);
  }

  function runImport() {
    if (!selected || !mappingComplete) return;
    importMutation.mutate({
      project_id: projectId,
      hub_id: selected.hub_id,
      config: previewQuery.data?.config ?? null,
      split: previewQuery.data?.split ?? "train",
      task_type: "llm_finetune",
      format,
      name: name.trim() || null,
      max_rows: maxRows,
      mapping: {
        instruction: mapping.instruction || null,
        input: mapping.input || null,
        output: mapping.output || null,
        messages: mapping.messages || null,
        conversations: mapping.conversations || null,
        question: null,
        context: null,
        answer: null
      }
    });
  }

  const imported = (catalogQuery.data ?? []).filter((dataset) => dataset.origin === "imported_hf");

  return (
    <div className="hub-panel">
      <div className="segmented-control mb-3">
        <button type="button" className={tab === "discover" ? "segmented-active" : ""} onClick={() => setTab("discover")}>
          Discover
        </button>
        <button type="button" className={tab === "imported" ? "segmented-active" : ""} onClick={() => setTab("imported")}>
          Imported{imported.length > 0 ? <span className="segmented-count">{imported.length}</span> : null}
        </button>
      </div>

      {tab === "imported" ? (
        imported.length === 0 ? (
          <EmptyState
            label="Nothing imported yet."
            icon={<Cloud size={26} />}
            description="Datasets you import from the Hub appear here and in the catalog."
          />
        ) : (
          <div className="hub-results">
            {imported.map((dataset) => (
              <button
                key={dataset.id}
                type="button"
                className="hub-result"
                onClick={() => openDataset(dataset.id)}
              >
                <div className="hub-result-title">
                  <strong>{dataset.name}</strong>
                  <FolderOpen size={14} />
                </div>
                <div className="hub-result-meta">
                  <span>{formatDatasetFormat(dataset.format)}</span>
                  {dataset.origin_ref && <span title={dataset.origin_ref}>{dataset.origin_ref.split("@")[0]}</span>}
                </div>
              </button>
            ))}
          </div>
        )
      ) : (
        <>
        <div className="hub-search-row">
        <div className="hub-search-input">
          <Search size={15} />
          <input
            value={searchInput}
            onChange={(event) => setSearchInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") submitSearch();
            }}
            placeholder="Search HuggingFace datasets — e.g. alpaca, dolly, oasst"
          />
        </div>
        <button className="secondary-button" onClick={submitSearch} disabled={searchQuery.isFetching}>
          {searchQuery.isFetching ? <InlineSpinner label="Searching" /> : "Search"}
        </button>
      </div>

      {searchQuery.data?.error && <p className="panel-error-note">Hub search failed: {searchQuery.data.error}</p>}

      <div className="hub-layout">
        <div className="hub-results" role="list">
          {searchQuery.isLoading ? (
            <InlineSpinner label="Loading datasets" />
          ) : (searchQuery.data?.results ?? []).length === 0 ? (
            <EmptyState label="No datasets found." icon={<Cloud size={26} />} description="Try a different search term." />
          ) : (
            (searchQuery.data?.results ?? []).map((result) => (
              <button
                key={result.hub_id}
                type="button"
                role="listitem"
                className={`hub-result ${selected?.hub_id === result.hub_id ? "hub-result-active" : ""}`}
                onClick={() => setSelected(result)}
              >
                <div className="hub-result-title">
                  <strong>{result.hub_id}</strong>
                  {result.gated && <Badge tone="warn">Gated</Badge>}
                </div>
                <div className="hub-result-meta">
                  <span>{result.downloads.toLocaleString()} downloads</span>
                  <span>{result.likes.toLocaleString()} likes</span>
                </div>
              </button>
            ))
          )}
        </div>

        <div className="hub-detail">
          {!selected ? (
            <EmptyState
              label="Select a dataset to preview"
              icon={<Cloud size={26} />}
              description="Preview sample rows, map columns to record fields, then import."
            />
          ) : previewQuery.isLoading ? (
            <InlineSpinner label="Loading preview" />
          ) : previewQuery.data?.error ? (
            <p className="panel-error-note">Preview unavailable: {previewQuery.data.error}. You can still import blind.</p>
          ) : (
            <>
              <div className="hub-format-row">
                <div className="segmented-control">
                  <button
                    type="button"
                    className={format === "instruction_jsonl" ? "segmented-active" : ""}
                    onClick={() => setFormat("instruction_jsonl")}
                  >
                    Instruction
                  </button>
                  <button
                    type="button"
                    className={format === "chat_jsonl" ? "segmented-active" : ""}
                    onClick={() => setFormat("chat_jsonl")}
                  >
                    Chat
                  </button>
                </div>
                {detectionMatched ? (
                  <span className="hub-banner hub-banner-success">
                    <CheckCircle2 size={14} /> Detected {previewQuery.data?.detected_format} format — no manual mapping needed
                  </span>
                ) : (
                  <span className="hub-banner hub-banner-info">Assign each column to a record field below</span>
                )}
              </div>

              <div className="table-wrap">
                <table className="hub-preview-table">
                  <thead>
                    <tr>
                      {columns.map((column) => (
                        <th key={column}>
                          <span className="hub-col-name" title={column}>{column}</span>
                          <select
                            className="hub-col-select"
                            value={roleForColumn(column)}
                            onChange={(event) => assignColumn(column, event.target.value)}
                          >
                            <option value="">— unused —</option>
                            {roles.map((role) => (
                              <option key={role.key} value={role.key}>{role.label}</option>
                            ))}
                          </select>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(previewQuery.data?.rows ?? []).map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {columns.map((column) => (
                          <td key={column} title={cellText(row[column])}>{cellText(row[column])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="hub-import-footer">
                <Field label="Dataset name">
                  <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Name this dataset" />
                </Field>
                <Field label="Rows">
                  <select value={maxRows} onChange={(event) => setMaxRows(Number(event.target.value))}>
                    {[500, 1000, 2500, 5000].map((value) => (
                      <option key={value} value={value}>{value.toLocaleString()}</option>
                    ))}
                  </select>
                </Field>
                <button
                  className="primary-button"
                  onClick={runImport}
                  disabled={!mappingComplete || importMutation.isPending}
                >
                  {importMutation.isPending ? (
                    <InlineSpinner label="Importing" />
                  ) : (
                    <>
                      <Download size={16} /> Import {maxRows.toLocaleString()} rows
                    </>
                  )}
                </button>
              </div>
              <MutationError mutations={[importMutation]} />
            </>
          )}
        </div>
      </div>
        </>
      )}
    </div>
  );
}

function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 120 ? `${text.slice(0, 120)}…` : text;
}
