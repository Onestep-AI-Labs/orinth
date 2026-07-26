# Spec: Phase 11 Data Recipes

## Status

Implemented. Two items landed as deferred rather than in-scope (see Deferred):
CSV/JSONL sources flow through the standard extract → chunk → generate pipeline
(their rows are stringified to text) rather than a separate no-LLM direct
column→field mapping; and per-chunk regeneration is not exposed as an endpoint —
the Generate step re-runs the whole recipe, and record provenance (`source_id`,
`chunk_index`) is stored so a future per-chunk endpoint can be added without a
data change.

## Goal

Let users turn their own documents into training datasets: upload PDF/DOCX/TXT/MD/CSV/JSONL sources, extract and chunk the text, generate instruction/chat/QA records — LLM-assisted through OpenRouter when a key is configured, deterministic rule-based otherwise — then review, edit, and commit the records as an `llm_finetune` dataset in the hub (phase 10 formats).

## Scope

In:

- Recipe workspace lifecycle: create, list, inspect, delete.
- Source upload and text extraction for PDF, DOCX, TXT, MD, CSV, JSONL.
- Chunking with configurable size/overlap.
- LLM-assisted record generation via OpenRouter (user-provided API key + model in Settings) with a rule-based fallback per chunk and for whole runs.
- A recipe template gallery: preset recipe definitions (document QA, instruction-from-answer, conversation) that prefill output format, chunking, and generation settings.
- Record review/edit/add/delete and commit-to-dataset.
- OpenRouter settings fields and settings page section.

Out:

- OCR for scanned or image-only PDFs (unsupported; per-source warning only).
- Unsloth Studio's node-graph Recipe Studio (React Flow editor, samplers, expression columns, MCP tool providers): the adopted surface is its template gallery and a linear stepper, not the graph. The graph is a product of its own and its `data_designer` backend is AGPL-3.0.
- Fine-tuning behavior (phase 14).
- Recipe history queries beyond the filesystem listing — no DB table, no Alembic migration in this phase.

## Interfaces

- `POST /api/recipes`
- `GET /api/recipes?project_id=`
- `GET /api/recipes/{id}`
- `DELETE /api/recipes/{id}`
- `POST /api/recipes/{id}/sources` (multipart, multiple files)
- `DELETE /api/recipes/{id}/sources/{source_id}`
- `POST /api/recipes/{id}/generate`
- `POST /api/recipes/{id}/cancel`
- `GET /api/recipes/{id}/records?page=&page_size=`
- `PATCH /api/recipes/{id}/records/{index}`
- `POST /api/recipes/{id}/records`
- `POST /api/recipes/{id}/records/delete`
- `POST /api/recipes/{id}/commit`
- `GET /api/recipes/openrouter/models`
- Existing `GET/PATCH` settings routes extended with `openrouter_api_key` (write-only) and `openrouter_model`.
- Schemas: `RecipeCreate`, `RecipeRead`, `RecipeSourceRead`, `RecipeGenerateRequest`, `RecipeRecord`, `RecipeCommitRequest` in `backend/app/schemas.py`.
- Frontend surfaces: routes `frontend/app/(platform)/datasets/recipes/page.tsx` and `datasets/recipes/[recipeId]/page.tsx` (thin entrypoints); feature code under `frontend/features/recipes/`; catalog entry action "New from documents" in `frontend/features/datasets/catalog-view.tsx`; OpenRouter section in the settings feature; API module `frontend/lib/api/recipes.ts` merged into `api`.
- Storage/DB changes: `storage/recipes/<recipe_id>/` containing `manifest.json`, `sources/`, `chunks.json`, `records.jsonl`. No DB change.

## Behavior

### Service and persistence

- A new service package `backend/app/services/recipes/` owns the lifecycle; the router lives at `backend/app/api/routers/recipes.py`; the singleton is registered in `backend/app/container.py` following the existing service pattern.
- Persistence is filesystem-only, mirroring datasets: recipe state lives in `manifest.json` (`id`, `project_id`, `name`, `output_format` (`instruction_jsonl | chat_jsonl`), `status`, `sources`, `generation` settings, `warnings`, timestamps). Recipes are working areas, not queryable history, so a DB table and migration are deliberately avoided.
- `status` is one of `draft | extracting | generating | ready | failed`. Generation runs on the shared thread executor and the frontend polls recipe status with the standard TanStack refetch interval — no websockets, per platform convention.
- Backend startup sweeps any manifest left in `extracting`/`generating` to `failed`, for the same reason phase 3 fails orphaned training jobs: executor work does not survive a restart.

### Sources and extraction

- Accepted uploads: `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.jsonl`, capped at 20 MB per file. Parsing uses `pypdf` (PDF) and `python-docx` (DOCX) — small pure-Python core dependencies — and the stdlib for the rest; heavy ML libraries are never imported in this path.
- Extraction runs at upload time per source and stores plain text plus per-source stats (pages, characters, extraction warnings) in the manifest. A PDF with no extractable text (scanned or encrypted) records a per-source warning and is excluded from generation; it does not fail the recipe.
- CSV and JSONL sources are extracted to readable text (rows stringified as `key: value` lines) and flow through the standard chunk → generate pipeline. Direct column→field mapping with no LLM round trip is deferred; the HuggingFace Hub import path (phase 10) already covers structured column mapping for those who need it.

### Chunking and generation

- `POST /api/recipes/{id}/generate` accepts `mode` (`auto | llm | rules`), an optional OpenRouter model override, `chunk_size` (default 3000 characters), `chunk_overlap` (default 200), and `records_per_chunk` (default 3). Chunking splits on paragraph boundaries where possible and stores chunk provenance (source id, offset) in `chunks.json`.
- In `auto` mode the run is LLM-assisted when an OpenRouter key and model are configured, rule-based otherwise. `llm` mode without a configured key is rejected with a pointer to Settings.
- The LLM path calls OpenRouter chat completions over `httpx` with a per-chunk prompt demanding strict JSON matching the output format. A malformed response gets one repair retry; a second failure falls back to rules for that chunk and records a warning. A hard API failure (401, repeated 429, timeout) flips the remainder of the run to rule-based and records the reason — a half-generated recipe with provenance beats a dead run.
- Every record carries `generator: "llm" | "rules"` and its chunk provenance, so the review table can badge origin and per-chunk regeneration knows what to replace.
- The rule-based fallback is deterministic per format: QA detects heading/question patterns and pairs them with following text; instruction frames the section title as the instruction and the chunk as the output; chat produces a single-turn templated exchange. The spec states plainly: rule output is structurally valid scaffolding meant for human review, not a quality substitute for LLM generation.
- Generation is cancelable between chunks via `POST /api/recipes/{id}/cancel`; already-produced records are kept.

### OpenRouter settings

- Settings gain `openrouter_api_key` and `openrouter_model`. The key is write-only: reads expose only `openrouter_api_key_configured: bool`, following the existing `save_huggingface_token` masking pattern. The key is injected server-side at call time and is never echoed to the client, logged, or written into recipe manifests.
- `GET /api/recipes/openrouter/models` returns a live model list from OpenRouter's `/models` endpoint when a key exists, and a small curated fallback list otherwise, so the model select never renders empty.

### Review and commit

- `GET /api/recipes/{id}/records` pages through `records.jsonl`; `PATCH` edits a record with the same shape validation as phase-10 record writes; add and bulk-delete round out review. Edited records keep their provenance fields.
- `POST /api/recipes/{id}/commit` creates a dataset through `DatasetService` with the recipe's `output_format`, task `llm_finetune`, `origin: recipe`, and `origin_ref: <recipe_id>`; records land in the `unassigned` inbox per the phase-4 convention so the standard split flow applies. Commit with zero records is blocked.
- Commit does not delete the recipe; users can iterate and commit again (each commit creates a new dataset).

### Frontend

- The recipes landing page adopts the Unsloth Studio data-recipes anatomy (`features/data-recipes/pages/data-recipes-page.tsx`): a template gallery on top — one card per preset with a short description and concept badges (e.g., "Documents", "LLM-assisted", "QA pairs") plus a "Start blank" card — and the list of existing recipes below with status and actions.
- Presets are declarative JSON-like definitions on the frontend (`frontend/features/recipes/presets.ts`), mirroring Studio's `learning-recipes/*.json` idea at our scale: each sets `output_format`, chunking defaults, `records_per_chunk`, and the generation prompt flavor. Initial presets: Document QA (`instruction_jsonl`, QA prompting), Instruction from Answer (`instruction_jsonl`), Conversation (`chat_jsonl`). Choosing a preset creates a recipe prefilled with those settings; everything remains editable afterward.
- The recipe workspace is a stepper: Sources → Generate → Review → Commit. Generate shows whether OpenRouter is configured with a link to `/settings` and re-runs the whole recipe (`Regenerate`); Review is a paginated table with an edit drawer, provenance badges, and a warnings panel; Commit asks for the dataset name. Per-chunk regenerate is deferred (records still carry `source_id`/`chunk_index` provenance so it can be added later).
- Generated content is research data under the platform's medical-context rule: no UI copy may frame generated records as clinical guidance.

### Design and Studio UI adoption

- Unsloth Studio is UX reference only: the gallery/stepper anatomy is adopted; its code is never copied, and its gradient card surfaces and shine borders are not — template cards are flat, hairline-bordered cards with token-only color, per `frontend/DESIGN.md`.
- No new tokens, no shell change. The stepper reuses existing form/panel primitives; provenance badges use the `info` tone pair; concept badges on template cards are neutral badges; warnings use the `warning` `-tint`/`-strong` pair; the edit drawer uses `--shadow-overlay`.

## Edge Cases

- Scanned/encrypted PDF: per-source warning, excluded from generation, recipe continues.
- Empty extraction across all sources: generate is rejected with a clear message.
- OpenRouter 401: run flips to rules with a "key rejected" warning; 429/timeout: retry once, then per-chunk fallback.
- Source over the size cap or with a disallowed extension: rejected at upload with a field message.
- Cancel between chunks: status returns to `ready` with partial records.
- Restart mid-generation: manifest swept to `failed`; records produced so far remain reviewable.
- Commit into a project lacking `llm_finetune` in `task_types`: blocked with guidance (same rule as phase 10).

## Acceptance Criteria

- Users can create a recipe from a template card or blank, upload mixed sources, and see per-source extraction stats and warnings; a template-created recipe arrives with its preset format and generation settings applied.
- A run with no OpenRouter key produces rule-based records; configuring a key and rerunning produces LLM records with `llm` badges; a forced API failure mid-run yields mixed provenance plus a surfaced warning.
- Records can be edited, added, deleted, and regenerated (whole-recipe re-run from the Generate step); commit produces a phase-10 `llm_finetune` dataset whose records appear in the dataset's Records tab with `origin: recipe`.
- The OpenRouter key is never present in any API response, log line, or manifest on disk.
- Backend tests cover: extraction per file type (fixtures), chunking boundaries/overlap, rule-based generation per format, LLM-path JSON repair and fallback (mocked OpenRouter), cancel semantics, commit integration with `DatasetService`, settings masking.

## Deferred

- OCR for scanned documents.
- Multi-turn conversation synthesis beyond single-turn templated chat in the rule-based path.
- Recipe sharing/duplication across projects.
- Direct CSV/JSONL column→field mapping with no LLM round trip (rows currently stringify to text and flow through generation; Hub import covers structured column mapping).
- Per-chunk regenerate endpoint (whole-recipe regenerate ships; provenance is stored to enable a targeted endpoint later).

## Validation

Commands to run:

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```
