# Spec: Phase 10 Dataset Hub and LLM Data Formats

## Status

Implemented. The backend (task type, formats, record storage + `data.jsonl`
regeneration, ShareGPT normalization, Hub search/preview/import, LLM EDA, origin
provenance) and the frontend (sectioned catalog, hub import panel, Records tab
with role-aware edit drawer, LLM-aware EDA) are in place with backend tests. Two
scope adjustments from the draft are recorded under Deferred.

**Revised (as-is import).** Hub import produced exactly one thing —
`llm_finetune` records from a hand-mapped pair of columns — which made every
image, classification and summarization dataset on the Hub "browse only". A
second path, `POST /api/datasets/import/hub/as-is`, now downloads a split in the
shape the Hub served it. Import moved onto the result card with a row-count
selector beside it, because with no mapping to fill in there is nothing left to
open a detail screen for. See "As-is import" below.

**Revised again (downloading is not preparing).** The as-is path originally
handed straight off to the phase-21 agent, and that was wrong in a way only
using it showed: a Hub dataset is under no obligation to be shaped like
something Orinth trains, so a download that worked perfectly reported itself as
a failure whenever the agent could not name a task. Import now ends at
*downloaded* — the dataset is in the workspace, browsable, readable from a
notebook — and preparing is the button the Overview tab already had. Files
dropped into the workspace still auto-prepare; there the user has already said
what they have by handing over its structure.

**Revised (phase 21 follow-up).** The import panel was a search box over a flat
list of ids — a lookup tool that works only when you already know the dataset's
name. It is now a faceted browser shaped like huggingface.co: `hub/facets`
serves the filter vocabulary with a one-line explanation per term, `hub/search`
pushes filtering to the Hub instead of over-fetching and dropping locally, and
selecting a dataset opens a detail view with a read-only dataset viewer above
the import form. Two real defects surfaced while building it and are fixed:
`rajpurkar/squad` detected as no known shape (its column is `answers`, not
`answer`) and, once mapped by hand, imported **zero** rows while reporting
success, because its answer is a struct of parallel arrays that `_validate_record`
rejected per row. See "HuggingFace Hub browse" below.

## Goal

Reorganize the dataset catalog into a dynamic, sectioned hub that distinguishes user-created datasets from datasets imported from the HuggingFace Hub, and introduce the platform's LLM data vocabulary — a new `llm_finetune` task type with `instruction_jsonl` and `chat_jsonl` formats, a records-based viewer/editor, and LLM-aware EDA. This spec is the foundation for data recipes (phase 11) and LLM fine-tuning (phase 14): both produce or consume the formats defined here.

## Scope

In:

- Sectioned dataset catalog: Project datasets / Imported from HuggingFace / Shared samples.
- HuggingFace Hub search, preview, and import. Two import paths: the mapped one
  (columns → `llm_finetune` records) and the as-is one (download unchanged, let
  the prep agent decide the task).
- New task type `llm_finetune` and dataset formats `instruction_jsonl` and `chat_jsonl`.
- Records tab (viewer + editor) and LLM EDA for `llm_finetune` datasets.
- Manifest `origin` / `origin_ref` provenance fields.

Out:

- Document-to-dataset generation (phase 11).
- Any training behavior for `llm_finetune` datasets (phase 14).
- DB schema changes — datasets remain filesystem-only; no Alembic migration in this phase.

## Interfaces

- `GET /api/datasets/hub/facets` — the browse vocabulary, with a hint per term
- `GET /api/datasets/hub/search?query=&task=&limit=&modality=&format=&size=&task_category=&sort=`
- `GET /api/datasets/hub/preview?hub_id=&config=&split=&limit=`
- `POST /api/datasets/import/hub` — mapped import, synchronous, returns the dataset
- `POST /api/datasets/import/hub/as-is` — as-is import, 202, returns the draft while
  the download and prep run continue on the prep executor
- Existing `GET /api/datasets` — summaries gain an `origin` field.
- Existing dataset item routes (`GET/POST /api/datasets/{id}/items`, item detail, delete, move) work unchanged for `llm_finetune` records.
- New item text route reuse: `GET /api/datasets/{id}/items/{split}/{item_id}/text` returns the raw record JSON for `llm_finetune` items.
- Schemas: `DatasetHubSearchResult`, `DatasetHubPreview`, `DatasetHubImportRequest`, `DatasetHubIngestRequest`, `DatasetHubFacets`, `DatasetHubFacetOption` in `backend/app/schemas.py`; `TaskType` gains `llm_finetune`; `DatasetFormat` gains `instruction_jsonl` and `chat_jsonl`.
- Frontend surfaces: `frontend/features/datasets/catalog-view.tsx` (sections), `frontend/features/datasets/hub/` (`hub-browser.tsx`, `hub-detail.tsx`, `hub-hooks.ts`, replacing `hub-import-panel.tsx`), new `frontend/features/datasets/records-tab.tsx`, `detail-tabs.tsx` (tab switch by task type), `frontend/lib/api/datasets.ts` (hub search/preview/import calls).
- Storage/DB changes: dataset manifests gain `origin` (`created | imported_hf | recipe`) and `origin_ref`; `llm_finetune` items stored as `<split>/records/<item_id>.json` plus a regenerated `<split>/data.jsonl`. No DB change.

## Behavior

### Catalog sections

- The catalog groups datasets into three sections: Project datasets (`origin: created` or `recipe`), Imported from HuggingFace (`origin: imported_hf`), and Shared samples (existing `shared` flag). Sections render only when non-empty, so existing single-section workspaces look unchanged.
- Existing manifests without an `origin` field are treated as `created`. Backfilling manifests on disk is not required; the default keeps old datasets in the right section without a migration step.
- Task-type filter chips filter all sections at once. Shared-sample task filtering already exists and extends to `llm_finetune`.
- `origin` is a manifest field, not an id or name convention, for the same reason `shared` is (phase 4): naming conventions break silently on rename.

### LLM task type and formats

- One task type `llm_finetune` covers instruction-, chat-, and QA-shaped SFT data. The format distinguishes record shape; the task stays singular because all shapes go through the same chat-template path at training time (phase 14).
- `instruction_jsonl` records are `{"instruction": str, "input": str?, "output": str}`. QA-shaped data maps into it (question → instruction, context → input, answer → output) rather than getting a third format.
- `chat_jsonl` records are `{"messages": [{"role": "system" | "user" | "assistant", "content": str}, ...]}` with at least one user and one assistant message.
- ShareGPT-style `conversations: [{from, value}]` columns are detected on import and normalized to `messages` (`human` → `user`, `gpt` → `assistant`). This is a native re-implementation of a common community convention; no code is copied from the AGPL-licensed Unsloth Studio.
- Phase 4's rule "storage format is derived from the task, never chosen" gains an explicit exception: the create flow for `llm_finetune` shows an instruction | chat schema selector on the task step, because the task alone does not determine record shape. All other task types keep the derived-format caption.
- Projects must declare `llm_finetune` in their `task_types` to create or import datasets of that type; the create panel and hub import block otherwise, pointing at project settings.

### Record storage

- Each record is a JSON file at `<split>/records/<item_id>.json`, so the existing item CRUD, move, and delete routes work without special-casing. Records live alongside `images/`/`texts/` conventions but never mix with them in one dataset.
- On every record write (create, edit, delete, move), the service regenerates `<split>/data.jsonl` — one record per line — as the canonical training input. This mirrors the YOLO precedent of writing `data.yaml`/`labels/*.txt` for training compatibility: editors work on items, trainers read one flat file.
- `llm_finetune` datasets use the existing `unassigned` inbox plus train/valid/test splits and the existing split-processing flow; no label list is required at creation (record content carries the supervision).
- Records can be populated three ways: one at a time in the Records tab, by uploading a `.jsonl`/`.json`/`.csv` file (`POST /api/datasets/{id}/records/upload` — each row validated against the dataset's shape, malformed rows skipped and counted), or by importing from the HuggingFace Hub. All three land through the same `_validate_record` path and regenerate `data.jsonl`.

### HuggingFace Hub search, preview, import

- `GET /api/datasets/hub/search` proxies `huggingface_hub.HfApi.list_datasets` (already a transitive dependency of transformers). Hub errors surface as a panel message and never break the local catalog.
- `GET /api/datasets/hub/preview` fetches sample rows through the hosted datasets-server rows API over `httpx`. When the hosted server does not cover a dataset, the backend falls back to `datasets` streaming — only when the `llm` extra (phase 14) is installed; otherwise the preview reports the limitation and import remains possible blind.
- `POST /api/datasets/import/hub` takes `hub_id`, `config`, split mapping, target `task_type`, target format, column mapping, `max_rows`, `name`, and `project_id`. Import is synchronous and row-capped (max 5000; the UI defaults to 1000 with a 500/1000/2500/5000 selector). The datasets-server rows API caps a page at 100, so pages are fetched through a small worker pool and `data.jsonl` is written straight from the in-memory rows — a 1000-row import lands in ~7s, well under the dev proxy timeout, so no job table is warranted. Oversized requests are rejected with the cap stated.
- Column mapping covers the supported shapes: instruction/input/output columns, a `messages` or `conversations` column, or QA question/context/answer columns. Unmapped required fields fail validation before any download starts.
- Import writes into a temp directory and atomically moves into `storage/datasets/` on success, so an interrupted import never leaves a half-dataset in the catalog.
- Imported datasets are editable (source `editable`), carry `origin: imported_hf` and `origin_ref: <hub_id>@<revision>`, and land in the `unassigned` inbox for the standard split flow.
- Gated hub datasets use the configured HF token (`HUGGINGFACE_HUB_TOKEN`/`HF_TOKEN`); a 401/403 returns an actionable error naming the hub id and telling the user to accept the license on huggingface.co.
- Duplicate import of the same hub id gets a name suffix instead of failing; provenance stays distinguishable via `origin_ref`.

### HuggingFace Hub browse

The search endpoint returns the structured tags already split out — modality,
format, task categories, size category, languages, licence — plus `pretty_name`,
a flattened one-line `summary` from the dataset card, `has_viewer`, and
`importable`. The card renders those, because the question a user is actually
answering is "which of these two similarly-named datasets do I want", and an id
with a download count cannot answer it.

**Filtering is server-side.** `list_datasets(filter=["modality:text", ...])`
matches the same `prefix:value` tags huggingface.co's own facets use.
The previous implementation asked for `limit * 3` results and dropped the
non-text ones in Python, which is wrong in both directions: it wastes two thirds
of every request, and a filter that only ever sees the first page cannot find a
match on page four. `size_categories` is the one exception, passed as its own
argument because the Hub silently matches nothing when it arrives via `filter=`.

**Every term carries its explanation.** `services/hub_facets.py` holds the
vocabulary — nine modalities, eight formats, five size buckets, seven task
categories, four sorts — each with a `label`, a `hint`, and an `importable` flag.
The UI renders `hint` as the tooltip on the filter chip *and* on the badge
showing the same term on a result card, so there is one string per term and one
place to correct it. `importable` is the part that is ours rather than the Hub's:
it now means "Orinth has a task that could hold this", which covers text,
tabular, image and time-series and excludes audio, video, 3D, geospatial,
documents, and the shard formats the preview server cannot read row-wise. The
chip says so before the click rather than the import failing after it.

**Two defects the browse work exposed**, both fixed with regression tests:

- `detect_record_format` required a literal `answer` column, so `rajpurkar/squad`
  — the Hub's most-downloaded QA dataset, whose column is `answers` — detected as
  nothing and dropped into manual mapping. It now matches through the same
  `QUESTION_COLUMNS` / `ANSWER_COLUMNS` alias tuples the rest of `detect.py` uses,
  and the returned mapping names the dataset's real columns rather than the
  canonical roles.
- SQuAD's answer is `{"text": ["Denver Broncos"], "answer_start": [177]}`, a
  struct of parallel arrays. `_record_from_row` passed that dict straight to
  `_validate_record`, which raised per row — so the import reported success with
  5,000 skipped and nothing written. `_as_text` now flattens a one-element list
  and takes a struct's first string-valued member (`text` first, being the
  near-universal payload name). `None` passes through untouched, because an
  absent value is a skipped row where `"None"` would be a corrupt one that
  trains. Verified live: 500 rows imported, 0 skipped, readiness `ready`.

The preview additionally returns `column_types` (the datasets-server feature spec
reduced to one word) and `num_rows` for the selected split. The type is what tells
a user why a cell previews as JSON, and a `struct` answer column is the single
most common surprise on the Hub. `num_rows` is `None` rather than `0` when the
Hub will not say — an unsupported dataset reporting nothing must not render as an
empty one.

### As-is import

`POST /api/datasets/import/hub/as-is` takes `hub_id`, optional `config`/`split`,
`name`, `project_id`, and `max_rows` (1..100,000, default 5,000). It creates a
draft dataset, marks it `origin: imported_hf` immediately so it groups correctly
while it downloads, and queues the download on the **prep executor**. It answers
202 with the draft; the client polls `GET /datasets/{id}/prep/status`, which is
the same readout an uploaded folder gets.

When the download finishes the dataset is a **draft** and nothing else has
happened to it. `POST /datasets/{id}/prep` is a separate request, made by the
"Prepare with Orinth" button the Overview tab already renders for a draft.

The download writes into the draft's `_staging/`, in one of two shapes decided by
the split's *features* rather than by its Hub tags:

- **A media column** (`_type` of `Image`, `Audio`, or `Video`) makes it a folder
  tree: one directory per class, named from the `ClassLabel` column's vocabulary,
  one file per row inside it. That is exactly what `_detect_image_folders` reads,
  so a CIFAR-10 import lands as `classification` with ten labels and no further
  input. Nothing else is written beside the images — an earlier version also
  dropped the other columns into a `metadata.jsonl` sidecar, and that one file
  made `_detect_record_files` (which runs first) classify a folder of pictures as
  a table of rows.
- **Anything else** becomes one JSONL file of the rows as served, with
  `ClassLabel` integers resolved to their names first. A folder called `3` is not
  a class, and neither is a label column full of `3`.

**The sample spans the split, in blocks.** `stanfordnlp/imdb` stores its train
split sorted by label — rows 0–12,499 are `neg` and 12,500–24,999 are `pos` — so
taking rows off the head of a perfectly balanced dataset yields exactly one
class, detection refuses to call a single-valued column a taxonomy, and the
import looks like it failed on a dataset Orinth trains happily. Sorting by class,
by source file, or by date all behave this way.

Striding row by row fixes the distribution and breaks everything else: the rows
API caps a page at 100, so individual rows across a 25,000-row split is one
request per two or three of them — 250 requests for 600 rows, which
datasets-server answers with a 502 (observed). The sample is therefore taken as
`_SAMPLE_BLOCKS` (8) contiguous blocks spread evenly across the split, or more
blocks when the request needs more pages anyway. That costs the same number of
requests as reading off the head and still spans the data. A block that fails is
skipped, not fatal.

Measured on the real dataset: 600 rows of `stanfordnlp/imdb` arrive 300 `neg` /
300 `pos` (previously 600 `neg`), and pressing Prepare yields
`text_classification` with labels `[neg, pos]`, split, trainable.

**A detection gap this exposed.** `_detect_record_files` recognized only the
record shapes — alpaca, messages, ShareGPT, QA, `chosen`/`rejected` — and
returned "no recognizable structure" for anything else, so a `.jsonl` of `text`
and `label` (the commonest classification shape on the Hub) failed while the
identical data as a `.csv` succeeded, because only `_detect_tables` reached
`_classify_rows`. Unrecognized record files now fall through to the same
classifier, guarded on the upload containing no images so one stray sidecar
cannot outvote ten thousand pictures.

**Import lives on the card.** The mapped import needs a column mapping and
therefore a form; the as-is import needs a row count. So the result card carries
a rows selector and an Import button, and the detail screen leads with the same
pair. The card is an `<article>` with the body as its own button — a button
inside a button is invalid markup that browsers resolve by dropping one.

**The mapping form is a disclosure.** It was the face of every Hub dataset,
which is nonsense in front of an image classifier — a form asking which column
is the "instruction" told the user Orinth only wanted LLM data. It is a
`<details>` now, closed unless the preview server already recognised the rows as
alpaca/chat/QA, in which case it opens itself and says so. Native disclosure
rather than a state flag: keyboard and screen-reader behaviour come free.

### Records tab and LLM EDA

- For `llm_finetune` datasets the detail workspace replaces the Images and Annotate tabs with a Records tab: a paginated table (instruction/first-user-message excerpt, output/last-assistant excerpt, token estimate, split) with an edit drawer. The drawer is role-aware for chat records: messages render as an ordered list with per-message role select and content textarea; instruction records render as three labeled fields. The drawer also has a **Raw JSON** mode so a record can be edited as whatever structure the source actually carries, and the structured editor preserves any extra fields it does not surface (shown as field chips) rather than dropping them — imported rows keep unmapped columns intact. Splitting an LLM dataset uses a light path that never parses record bodies (records carry no label), so distributing a large imported dataset into train/valid/test stays fast.
- Record edits validate shape server-side (roles from the allowed set, non-empty output/assistant content) and rewrite `data.jsonl` for the affected split.
- The EDA tab for `llm_finetune` reports: record counts per split, token/character length distributions (estimated with a whitespace-and-punctuation heuristic so EDA never loads a tokenizer), role counts for chat data, and warnings (empty outputs, records over a configurable length threshold, duplicate records). Charts consume the `--label` ramp through `labelColor()`/`labelFill()` per the design contract.

### Design and Studio UI adoption

- Unsloth Studio (`unsloth/studio/frontend`, AGPL-3.0) is the UX reference for this phase: screen **anatomy and flows** are adopted; its code is never copied and its visual styling (gradients, shine borders, multi-hue cards) is not — everything renders in `frontend/DESIGN.md` tokens and `@/features/platform/ui` primitives.
- Hub browse adopts the Studio Hub page anatomy (`features/hub/hub-page.tsx`): a Discover | Imported tab pair — Discover is search plus result list, Imported lists what already landed locally (our HF section of the catalog). Tab and query state live in the URL search params so hub views are deep-linkable, mirroring Studio's `?tab=&kind=` pattern.
- The import flow adopts the Studio dataset-preview dialog anatomy (`features/studio/sections/dataset-preview-dialog*.tsx`): a preview rows table whose **column headers carry per-column role pickers** (instruction/input/output or message roles), a detected-format banner ("Detected alpaca format — no manual mapping needed") when heuristics succeed, an explicit manual-mapping state when they don't, and a footer Import button gated on mapping completeness. Detection heuristics (alpaca columns, `messages`/`conversations` shapes) are re-implemented natively in the backend per the Behavior section.
- No new tokens, no shell change. Catalog sections are hairline-separated groups with uppercase micro-label headers, following existing panel structure.
- The records table is a semantic `<table>` inside an `overflow-x: auto` wrapper; the edit drawer and the import dialog use `--shadow-overlay` (a sanctioned elevation).
- Hub search results use existing card/list primitives from `@/features/platform/ui`; the gated flag renders as a `warning`-tone badge (`-tint` background, `-strong` text); the detected-format banner uses the `success` pair, the manual-mapping state the `info` pair.

## Edge Cases

- Hub unreachable or rate-limited: search/preview return a surfaced error; the local catalog renders normally.
- Preview unsupported by the hosted datasets-server and `llm` extra absent: preview panel states the limitation; import stays available.
- Import interrupted mid-download: temp dir is discarded; no partial dataset appears.
- Non-UTF8 or oversized single records: skipped with a per-record warning count in the import response, not a run failure.
- Empty or invalid column mapping: rejected with field-level messages before download.
- Project without `llm_finetune` in `task_types`: create and import blocked with guidance to update project settings.
- Record with zero assistant/output content: rejected on write with a field message.

## Acceptance Criteria

- Catalog shows sections only when non-empty; existing datasets appear under Project datasets without manifest edits.
- Users can search the HF Hub, preview rows, map columns, and import a text dataset capped at the row limit; the imported dataset appears in the Imported from HuggingFace section and its items are browsable and editable.
- Users can import any dataset Orinth has a task for straight from the result card, choosing only a row count, and watch the download on a progress bar. The dataset then reads *downloaded*, is browsable, and offers "Prepare with Orinth" — which turns an image split into `classification` with its classes, and a `text`/`label` split into `text_classification`.
- A split sorted by its label imports balanced: `stanfordnlp/imdb` at 600 rows arrives 300/300, not 600/0.
- Users can create an `llm_finetune` dataset choosing instruction or chat schema, add records via the Records tab, edit them role-aware, and see `data.jsonl` regenerate per split.
- ShareGPT-style `conversations` import normalizes to `messages`.
- Gated dataset import without an accepted license shows the actionable license message.
- LLM EDA renders length distributions and warnings for a populated dataset.
- Backend tests cover: format validation for both record shapes, ShareGPT normalization, import column mapping, atomic import, `data.jsonl` regeneration, origin defaulting for legacy manifests.

## Deferred

- Multimodal and audio/video import. The as-is path handles image and row data;
  audio and video stay browse-only until the platform has a task for them.
- Bounding boxes and masks on an imported image dataset. The images and their
  class folders come across; an `object-detection` split's boxes do not, so it
  imports as classification and the regions have to be drawn in Orinth.
- Tokenizer-accurate token counts in EDA (requires the phase-14 extras).
- Hub dataset revision pinning/update flows beyond recording `origin_ref`.
- AI-assisted column mapping (Studio's "AI assist" button on the mapping dialog): needs the OpenRouter settings that phase 11 introduces; add it to the import dialog once those exist.
- URL-deep-linkable hub browse state. The hub panel has the Studio **Discover | Imported** tab pair (Imported lists the HF-origin datasets already in the catalog), but keeps tab/search/selection/mapping in local component state rather than `?tab=&kind=` search params. Revisit if hub views need to be shareable.
- `datasets`-streaming preview fallback. Preview and import go through the hosted datasets-server rows API only; when a dataset is not covered there the preview reports the limitation and import proceeds blind, as specified. Wire the streaming fallback when the phase-14 `llm` extra lands.

## Validation

Commands to run:

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```
