"use client";

import { useState } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import { Cloud, Database, Download, FolderOpen, Heart, Search, Table2, X } from "lucide-react";
import {
  EMPTY_FILTERS,
  useHubFacetsQuery,
  useHubSearchQuery,
  type HubFilters
} from "@/features/datasets/hub/hub-hooks";
import { HubDatasetDetail } from "@/features/datasets/hub/hub-detail";
import { useIngestHubDatasetMutation } from "@/features/datasets/prep/prep-hooks";
import { Badge, EmptyState, InlineSpinner, Select } from "@/features/platform/ui";
import type {
  DatasetHubFacetOption,
  DatasetHubSearchResult,
  DatasetSummary
} from "@/types/api";

/**
 * Browsing the Hub, shaped like the Hub.
 *
 * The old panel was a search box over a flat list of ids with a download count
 * beside each. That is a lookup tool: it works when you already know the
 * dataset's name and is useless for finding one. huggingface.co solves the same
 * problem with faceted browse — modality, format, size, task down the left, a
 * sort control, and cards carrying enough metadata to choose between two
 * similarly-named results — so this is that, in Orinth's tokens.
 *
 * Filtering is server-side (`list_datasets(filter=...)`), not a client-side
 * pass over one page of results: a filter that can only see the first thirty
 * rows cannot find a match on page four, and the old panel over-fetched three
 * pages precisely because it was trying.
 *
 * Every filter term carries the sentence explaining it, served from
 * `services/hub_facets.py`. "webdataset" and `10K<n<100K` are not
 * self-describing, and a chip that only names a term is a quiz.
 */

type FacetKey = "modality" | "format" | "size" | "task_category";

function formatCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

/** "3 days ago" — the Hub's own relative form, which is what makes a card scannable. */
function relativeDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const days = Math.floor((Date.now() - then) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months} month${months === 1 ? "" : "s"} ago`;
  const years = Math.floor(months / 12);
  return `${years} year${years === 1 ? "" : "s"} ago`;
}

function FacetGroup({
  label,
  options,
  selected,
  onToggle
}: {
  label: string;
  options: DatasetHubFacetOption[];
  selected: string[];
  onToggle: (value: string) => void;
}) {
  if (options.length === 0) return null;
  return (
    <section className="hub-facet">
      <p className="section-microlabel">{label}</p>
      <div className="hub-facet-chips">
        {options.map((option) => {
          const active = selected.includes(option.value);
          return (
            <button
              key={option.value}
              type="button"
              // The hint is the chip's own tooltip, so the explanation is one
              // hover away from the control it explains rather than parked in a
              // legend somewhere else on the page.
              title={
                option.importable
                  ? option.hint
                  : `${option.hint} Browse only — Orinth cannot import this.`
              }
              aria-pressed={active}
              className={`hub-chip ${active ? "hub-chip-active" : ""} ${
                option.importable ? "" : "hub-chip-muted"
              }`}
              onClick={() => onToggle(option.value)}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </section>
  );
}

//: Rows an inline import takes, smallest first. Import reads from the start of
//: the split, so these are "how much of it do you want" rather than a sample
//: size — and the list stops where `DatasetHubIngestRequest.max_rows` does.
const ROW_CHOICES = [1_000, 5_000, 25_000, 100_000];

function ResultCard({
  result,
  hints,
  onOpen,
  onImport,
  importing
}: {
  result: DatasetHubSearchResult;
  hints: Map<string, string>;
  onOpen: () => void;
  onImport: (maxRows: number) => void;
  importing: boolean;
}) {
  const [maxRows, setMaxRows] = useState(ROW_CHOICES[1]);
  const meta: string[] = [];
  const updated = relativeDate(result.updated_at);
  if (updated) meta.push(`Updated ${updated}`);
  if (result.size_category) meta.push(result.size_category);

  return (
    // An `<article>` holding a button, not a button wrapping everything. The
    // card carries its own import control now, and a button inside a button is
    // invalid markup that browsers resolve by dropping one of them.
    <article className="hub-card">
      <button type="button" className="hub-card-open" onClick={onOpen}>
        <div className="hub-card-head">
          <Database size={14} aria-hidden="true" />
          <strong title={result.hub_id}>{result.hub_id}</strong>
        </div>

        <div className="hub-card-badges">
          {/* `has_viewer` is the difference between a dataset you can inspect
              before importing and one you must take on trust. Saying so on the
              card beats letting the user click through to find out. */}
          {result.has_viewer && (
            <Badge tone="neutral" title="The Hub can render sample rows, so a preview is available.">
              <Table2 size={11} /> Viewer
            </Badge>
          )}
          {result.gated && (
            <Badge tone="warn" title="Accept this dataset's licence on huggingface.co and add a token in Settings before importing.">
              Gated
            </Badge>
          )}
          {!result.importable && (
            <Badge tone="info" title="Orinth has no task for this dataset's modality yet — audio, video and 3D can be browsed here but not imported.">
              Browse only
            </Badge>
          )}
          {result.modalities.slice(0, 2).map((value) => (
            <Badge key={value} tone="neutral" title={hints.get(`modality:${value}`) ?? value}>
              {value}
            </Badge>
          ))}
          {result.formats.slice(0, 2).map((value) => (
            <Badge key={value} tone="neutral" title={hints.get(`format:${value}`) ?? value}>
              {value}
            </Badge>
          ))}
        </div>

        {result.summary && <p className="hub-card-summary">{result.summary}</p>}

        <div className="hub-card-meta">
          {meta.map((entry) => (
            <span key={entry}>{entry}</span>
          ))}
          <span title={`${result.downloads.toLocaleString()} downloads`}>
            <Download size={12} /> {formatCount(result.downloads)}
          </span>
          <span title={`${result.likes.toLocaleString()} likes`}>
            <Heart size={12} /> {formatCount(result.likes)}
          </span>
        </div>
      </button>

      {/* Import from the row, not from a detail screen two clicks away. Most
          imports need no column mapping at all — the agent works the shape out
          from the data — so the only decision left is how many rows, and it
          belongs beside the button that acts on it. */}
      <div className="hub-card-import">
        <Select
          value={maxRows}
          onChange={(event) => setMaxRows(Number(event.target.value))}
          aria-label={`Rows to import from ${result.hub_id}`}
          title="Import reads from the start of the split."
          disabled={!result.importable || importing}
        >
          {ROW_CHOICES.map((value) => (
            <option key={value} value={value}>
              {value.toLocaleString()} rows
            </option>
          ))}
        </Select>
        <button
          type="button"
          className="secondary-button button-sm"
          onClick={() => onImport(maxRows)}
          disabled={!result.importable || importing}
          title={
            result.importable
              ? "Download this dataset as it is. Orinth works out what it is and prepares it."
              : "Orinth has no task for this dataset's modality yet."
          }
        >
          {importing ? (
            <InlineSpinner label="Importing" />
          ) : (
            <>
              <Download size={14} /> Import
            </>
          )}
        </button>
      </div>
    </article>
  );
}

export function HubBrowser({
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
  const [tab, setTab] = useState<"discover" | "imported">("discover");
  const [filters, setFilters] = useState<HubFilters>(EMPTY_FILTERS);
  // The committed search term, separate from the input: searching on every
  // keystroke would be one Hub request per character and a fast route to a 429.
  const [searchInput, setSearchInput] = useState("");
  const [selected, setSelected] = useState<DatasetHubSearchResult | null>(null);

  const facetsQuery = useHubFacetsQuery();
  const searchQuery = useHubSearchQuery(filters);
  const facets = facetsQuery.data;
  //: Which card's Import was pressed. One at a time: the run is queued on the
  //: prep executor and a second click before the first lands would create a
  //: duplicate draft rather than a second dataset the user wanted.
  const [importingId, setImportingId] = useState("");
  const ingestMutation = useIngestHubDatasetMutation((datasetId) => {
    setImportingId("");
    onImported();
    openDataset(datasetId);
  });

  function importAsIs(result: DatasetHubSearchResult, maxRows: number) {
    setImportingId(result.hub_id);
    ingestMutation.mutate(
      {
        project_id: projectId,
        hub_id: result.hub_id,
        config: null,
        split: null,
        name: null,
        max_rows: maxRows
      },
      { onError: () => setImportingId("") }
    );
  }

  // One lookup for every `prefix:value` hint, so a badge on a card and the chip
  // in the rail show the same sentence.
  const hints = new Map<string, string>();
  for (const option of facets?.modalities ?? []) hints.set(`modality:${option.value}`, option.hint);
  for (const option of facets?.formats ?? []) hints.set(`format:${option.value}`, option.hint);
  for (const option of facets?.sizes ?? []) hints.set(`size:${option.value}`, option.hint);
  for (const option of facets?.tasks ?? []) hints.set(`task:${option.value}`, option.hint);

  function toggle(key: FacetKey, value: string) {
    setFilters((current) => {
      const active = current[key].includes(value);
      return {
        ...current,
        [key]: active ? current[key].filter((entry) => entry !== value) : [...current[key], value]
      };
    });
    setSelected(null);
  }

  const activeCount =
    filters.modality.length +
    filters.format.length +
    filters.size.length +
    filters.task_category.length;

  const results = searchQuery.data?.results ?? [];
  const imported = (catalogQuery.data ?? []).filter((dataset) => dataset.origin === "imported_hf");

  if (selected) {
    return (
      <HubDatasetDetail
        result={selected}
        hints={hints}
        projectId={projectId}
        openDataset={openDataset}
        catalogQuery={catalogQuery}
        onImported={onImported}
        onBack={() => setSelected(null)}
      />
    );
  }

  return (
    <div className="hub-panel">
      <div className="segmented-control mb-3">
        <button
          type="button"
          className={tab === "discover" ? "segmented-active" : ""}
          onClick={() => setTab("discover")}
        >
          Discover
        </button>
        <button
          type="button"
          className={tab === "imported" ? "segmented-active" : ""}
          onClick={() => setTab("imported")}
        >
          Imported
          {imported.length > 0 ? <span className="segmented-count">{imported.length}</span> : null}
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
          <div className="hub-grid">
            {imported.map((dataset) => (
              <button
                key={dataset.id}
                type="button"
                className="hub-card"
                onClick={() => openDataset(dataset.id)}
              >
                <div className="hub-card-head">
                  <FolderOpen size={14} aria-hidden="true" />
                  <strong>{dataset.name}</strong>
                </div>
                <div className="hub-card-meta">
                  {dataset.origin_ref && (
                    <span title={dataset.origin_ref}>{dataset.origin_ref.split("@")[0]}</span>
                  )}
                </div>
              </button>
            ))}
          </div>
        )
      ) : (
        <div className="hub-browse">
          <aside className="hub-rail">
            <div className="hub-rail-head">
              <p className="section-microlabel">Filters</p>
              {activeCount > 0 && (
                <button
                  type="button"
                  className="hub-clear"
                  onClick={() => {
                    setFilters((current) => ({ ...EMPTY_FILTERS, query: current.query, sort: current.sort }));
                    setSelected(null);
                  }}
                >
                  <X size={12} /> Clear {activeCount}
                </button>
              )}
            </div>
            {facetsQuery.isLoading ? (
              <InlineSpinner label="Loading filters" />
            ) : (
              <>
                <FacetGroup
                  label="Modality"
                  options={facets?.modalities ?? []}
                  selected={filters.modality}
                  onToggle={(value) => toggle("modality", value)}
                />
                <FacetGroup
                  label="Format"
                  options={facets?.formats ?? []}
                  selected={filters.format}
                  onToggle={(value) => toggle("format", value)}
                />
                <FacetGroup
                  label="Size"
                  options={facets?.sizes ?? []}
                  selected={filters.size}
                  onToggle={(value) => toggle("size", value)}
                />
                <FacetGroup
                  label="Task"
                  options={facets?.tasks ?? []}
                  selected={filters.task_category}
                  onToggle={(value) => toggle("task_category", value)}
                />
              </>
            )}
          </aside>

          <div className="hub-main">
            <div className="hub-search-row">
              <div className="hub-search-input">
                <Search size={15} />
                <input
                  value={searchInput}
                  onChange={(event) => setSearchInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key !== "Enter") return;
                    setFilters((current) => ({ ...current, query: searchInput.trim() }));
                    setSelected(null);
                  }}
                  placeholder="Filter by name — e.g. squad, alpaca, dolly"
                  aria-label="Search HuggingFace datasets"
                />
              </div>
              <Select
                value={filters.sort}
                onChange={(event) =>
                  setFilters((current) => ({ ...current, sort: event.target.value }))
                }
                aria-label="Sort"
                title={
                  (facets?.sorts ?? []).find((option) => option.value === filters.sort)?.hint ?? ""
                }
              >
                {(facets?.sorts ?? []).map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </Select>
            </div>

            <div className="hub-result-count">
              {searchQuery.isFetching ? (
                <InlineSpinner label="Searching the Hub" />
              ) : (
                <span className="form-caption">
                  {results.length} dataset{results.length === 1 ? "" : "s"}
                  {filters.query ? ` matching “${filters.query}”` : ""}
                </span>
              )}
            </div>

            {searchQuery.data?.error && (
              <p className="panel-error-note">Hub search failed: {searchQuery.data.error}</p>
            )}

            {results.length === 0 && !searchQuery.isFetching ? (
              <EmptyState
                label="No datasets match those filters."
                icon={<Cloud size={26} />}
                description={
                  activeCount > 0
                    ? "Clear a filter, or search by name instead."
                    : "Try a search term, or pick a modality on the left."
                }
              />
            ) : (
              <div className="hub-grid">
                {results.map((result) => (
                  <ResultCard
                    key={result.hub_id}
                    result={result}
                    hints={hints}
                    onOpen={() => setSelected(result)}
                    onImport={(maxRows) => importAsIs(result, maxRows)}
                    importing={importingId === result.hub_id}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
