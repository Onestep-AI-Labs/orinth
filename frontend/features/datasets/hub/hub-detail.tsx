"use client";

import { useEffect, useMemo, useState } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import {
  ArrowLeft,
  CheckCircle2,
  Download,
  ExternalLink,
  Info,
  Table2
} from "lucide-react";
import { useHubPreviewQuery } from "@/features/datasets/hub/hub-hooks";
import { useImportHubDatasetMutation } from "@/features/datasets/hooks";
import {
  Badge,
  Field,
  InlineSpinner,
  MutationError,
  Select,
  TableSkeleton
} from "@/features/platform/ui";
import type {
  DatasetFormat,
  DatasetHubSearchResult,
  DatasetSummary
} from "@/types/api";

/**
 * One Hub dataset: what it is, what is in it, and how it comes in.
 *
 * Reads top to bottom as the three questions actually asked, in order — is this
 * the right dataset (header, badges, description), what do the rows look like
 * (the viewer), and how do its columns become training records (the mapping and
 * import footer). The old panel put a column-mapping `<select>` inside every
 * table header before the user had decided whether they wanted the dataset at
 * all, which made "look at this data" and "configure an import" the same
 * gesture.
 *
 * The viewer is a plain read-only table for that reason. Mapping moved below it
 * into a row of named fields, where it is a short form rather than a property
 * grid embedded in a data grid.
 */

type Role = { key: string; label: string; hint: string; required: boolean };

const INSTRUCTION_ROLES: Role[] = [
  {
    key: "instruction",
    label: "Instruction",
    hint: "The prompt the model is asked to answer. Usually the question or task column.",
    required: true
  },
  {
    key: "input",
    label: "Input",
    hint: "Optional extra context the instruction refers to — a passage, a document, a table.",
    required: false
  },
  {
    key: "output",
    label: "Output",
    hint: "The answer the model should learn to produce. This is what it is scored against.",
    required: true
  }
];

const CHAT_ROLES: Role[] = [
  {
    key: "messages",
    label: "Messages",
    hint: "A column holding a list of {role, content} turns — the OpenAI chat shape.",
    required: true
  },
  {
    key: "conversations",
    label: "Conversations",
    hint: "A column holding ShareGPT-style turns, using `from` and `value` keys.",
    required: false
  }
];

const FORMAT_HINTS: Record<DatasetFormat | string, string> = {
  instruction_jsonl:
    "One prompt and one answer per record. The right shape for question answering, summarization, and most task datasets.",
  chat_jsonl:
    "A list of conversation turns per record. Use this when the dataset holds multi-turn dialogue."
};

function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > 160 ? `${text.slice(0, 160)}…` : text;
}

export function HubDatasetDetail({
  result,
  hints,
  projectId,
  openDataset,
  catalogQuery,
  onImported,
  onBack
}: {
  result: DatasetHubSearchResult;
  hints: Map<string, string>;
  projectId: string;
  openDataset: (datasetId: string) => void;
  catalogQuery: UseQueryResult<DatasetSummary[]>;
  onImported: () => void;
  onBack: () => void;
}) {
  const [config, setConfig] = useState<string | undefined>(undefined);
  const [split, setSplit] = useState<string | undefined>(undefined);
  const [format, setFormat] = useState<DatasetFormat>("instruction_jsonl");
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [name, setName] = useState("");
  const [maxRows, setMaxRows] = useState(1000);

  const previewQuery = useHubPreviewQuery(result.hub_id, config, split);
  const preview = previewQuery.data;
  const importMutation = useImportHubDatasetMutation({ openDataset, catalogQuery, onImported });

  // Server-side detection pre-fills format and mapping, so a recognised dataset
  // imports without the user touching the form at all. QA columns normalise into
  // the instruction shape, which is why `question`/`answer` land on
  // `instruction`/`output` rather than needing roles of their own.
  useEffect(() => {
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
  }, [preview]);

  useEffect(() => {
    setName(result.hub_id.split("/").pop() ?? result.hub_id);
  }, [result.hub_id]);

  const roles = format === "chat_jsonl" ? CHAT_ROLES : INSTRUCTION_ROLES;
  const columns = preview?.columns ?? [];
  const columnTypes = preview?.column_types ?? {};

  const mappingComplete =
    format === "chat_jsonl"
      ? Boolean(mapping.messages || mapping.conversations)
      : Boolean(mapping.instruction && mapping.output);

  const autoDetected = useMemo(
    () => Boolean(preview?.detected_format) && mappingComplete,
    [mappingComplete, preview?.detected_format]
  );

  function assignRole(roleKey: string, column: string) {
    setMapping((current) => {
      const next: Record<string, string> = {};
      // One column per role and one role per column: clear whatever else held
      // this column, or the same text would be written into two record fields.
      for (const [key, value] of Object.entries(current)) {
        if (key !== roleKey && value !== column) next[key] = value;
      }
      if (column) next[roleKey] = column;
      return next;
    });
  }

  function runImport() {
    if (!mappingComplete) return;
    importMutation.mutate({
      project_id: projectId,
      hub_id: result.hub_id,
      config: preview?.config ?? null,
      split: preview?.split ?? "train",
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

  const rowsAvailable = preview?.num_rows ?? null;
  const willImport = rowsAvailable === null ? maxRows : Math.min(maxRows, rowsAvailable);

  return (
    <div className="hub-detail-view">
      <div className="hub-detail-head">
        <button type="button" className="hub-back" onClick={onBack}>
          <ArrowLeft size={15} /> All datasets
        </button>
        <a
          className="hub-external"
          href={`https://huggingface.co/datasets/${result.hub_id}`}
          target="_blank"
          rel="noreferrer"
          title="Open the full dataset card on huggingface.co"
        >
          View on HuggingFace <ExternalLink size={13} />
        </a>
      </div>

      <div className="hub-detail-title">
        <h3>{result.pretty_name || result.hub_id}</h3>
        <p className="form-caption">{result.hub_id}</p>
      </div>

      <div className="hub-card-badges">
        {result.modalities.map((value) => (
          <Badge key={`m-${value}`} tone="neutral" title={hints.get(`modality:${value}`) ?? value}>
            {value}
          </Badge>
        ))}
        {result.formats.map((value) => (
          <Badge key={`f-${value}`} tone="neutral" title={hints.get(`format:${value}`) ?? value}>
            {value}
          </Badge>
        ))}
        {result.size_category && (
          <Badge tone="neutral" title={hints.get(`size:${result.size_category}`) ?? "Approximate row count, as declared by the dataset author."}>
            {result.size_category}
          </Badge>
        )}
        {result.task_categories.map((value) => (
          <Badge key={`t-${value}`} tone="info" title={hints.get(`task:${value}`) ?? value}>
            {value.replace(/-/g, " ")}
          </Badge>
        ))}
        {result.license && (
          <Badge tone="neutral" title="The licence the author published this dataset under. Check it allows your use.">
            {result.license}
          </Badge>
        )}
        {result.languages.map((value) => (
          <Badge key={`l-${value}`} tone="neutral" title="Language declared by the dataset author.">
            {value}
          </Badge>
        ))}
      </div>

      {result.summary && <p className="hub-detail-summary">{result.summary}</p>}

      {!result.importable && (
        <p className="hub-banner hub-banner-info">
          <Info size={14} /> Orinth imports Hub datasets as text records. This dataset&apos;s
          modality is {result.modalities.join(", ") || "not text"}, so you can look at it here but
          importing will not produce usable records.
        </p>
      )}

      {/* --- viewer ------------------------------------------------------ */}
      <section className="hub-viewer">
        <div className="hub-viewer-head">
          <span className="section-microlabel">
            <Table2 size={13} /> Dataset viewer
          </span>
          <div className="hub-viewer-controls">
            {previewQuery.isFetching && <InlineSpinner label="Loading rows" />}
            {rowsAvailable !== null && (
              <Badge tone="neutral" title="Rows in the selected split, as reported by the Hub.">
                {rowsAvailable.toLocaleString()} rows
              </Badge>
            )}
            {(preview?.configs?.length ?? 0) > 1 && (
              <Select
                value={preview?.config ?? ""}
                onChange={(event) => {
                  setConfig(event.target.value);
                  setSplit(undefined);
                }}
                aria-label="Configuration"
                title="A dataset can ship several variants — languages, subsets, or task versions. Each has its own columns."
              >
                {(preview?.configs ?? []).map((entry) => (
                  <option key={entry} value={entry}>
                    {entry}
                  </option>
                ))}
              </Select>
            )}
            {(preview?.splits?.length ?? 0) > 1 && (
              <Select
                value={preview?.split ?? ""}
                onChange={(event) => setSplit(event.target.value)}
                aria-label="Split"
                title="Which portion of the dataset to preview and import — usually train."
              >
                {(preview?.splits ?? []).map((entry) => (
                  <option key={entry} value={entry}>
                    {entry}
                  </option>
                ))}
              </Select>
            )}
          </div>
        </div>

        {previewQuery.isLoading ? (
          <TableSkeleton rows={6} />
        ) : preview?.error ? (
          <p className="panel-error-note">
            No preview: {preview.error} You can still import — the columns just have to be named by
            hand below.
          </p>
        ) : (
          <div className="grid-scroll">
            <table className="hub-viewer-table">
              <thead>
                <tr>
                  {columns.map((column) => (
                    <th key={column} scope="col">
                      <span className="hub-col-name" title={column}>
                        {column}
                      </span>
                      {/* The type is what tells a user why a cell previews as
                          JSON — a `struct` answer column is the single most
                          common surprise on the Hub. */}
                      <span className="hub-col-type">{columnTypes[column] ?? "—"}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(preview?.rows ?? []).map((row, index) => (
                  <tr key={index}>
                    {columns.map((column) => (
                      <td key={column} title={cellText(row[column])}>
                        {cellText(row[column])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* --- import ------------------------------------------------------ */}
      <section className="hub-import">
        <div className="hub-viewer-head">
          <span className="section-microlabel">Import as</span>
          {autoDetected ? (
            <span className="hub-banner hub-banner-success">
              <CheckCircle2 size={14} /> Recognised as {preview?.detected_format} — columns are
              already matched, just press Import.
            </span>
          ) : (
            <span className="hub-banner hub-banner-info">
              <Info size={14} /> Orinth could not recognise this shape. Point each field at a column
              below.
            </span>
          )}
        </div>

        <div className="segmented-control">
          {(["instruction_jsonl", "chat_jsonl"] as DatasetFormat[]).map((option) => (
            <button
              key={option}
              type="button"
              className={format === option ? "segmented-active" : ""}
              onClick={() => setFormat(option)}
              title={FORMAT_HINTS[option]}
            >
              {option === "instruction_jsonl" ? "Instruction" : "Chat"}
            </button>
          ))}
        </div>

        <div className="hub-mapping">
          {roles.map((role) => (
            <Field
              key={role.key}
              label={role.label}
              hint={role.required ? undefined : "optional"}
            >
              <Select
                value={mapping[role.key] ?? ""}
                onChange={(event) => assignRole(role.key, event.target.value)}
                title={role.hint}
              >
                <option value="">— none —</option>
                {columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                    {columnTypes[column] ? ` (${columnTypes[column]})` : ""}
                  </option>
                ))}
              </Select>
              <span className="form-caption">{role.hint}</span>
            </Field>
          ))}
        </div>

        <div className="hub-import-footer">
          <Field label="Dataset name">
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Name this dataset"
            />
          </Field>
          <Field
            label="Rows to import"
            hint={rowsAvailable !== null && rowsAvailable > maxRows ? "sampled" : undefined}
          >
            <Select
              value={maxRows}
              onChange={(event) => setMaxRows(Number(event.target.value))}
              title="Import reads from the start of the split. Large datasets are sampled rather than downloaded in full."
            >
              {[500, 1000, 2500, 5000].map((value) => (
                <option key={value} value={value}>
                  {value.toLocaleString()}
                </option>
              ))}
            </Select>
          </Field>
          <button
            className="primary-button"
            onClick={runImport}
            disabled={!mappingComplete || importMutation.isPending}
            title={
              mappingComplete
                ? "Download the rows and create a dataset in this project."
                : "Choose a column for each required field first."
            }
          >
            {importMutation.isPending ? (
              <InlineSpinner label="Importing" />
            ) : (
              <>
                <Download size={16} /> Import {willImport.toLocaleString()} rows
              </>
            )}
          </button>
        </div>
        <MutationError mutations={[importMutation]} />
      </section>
    </div>
  );
}
