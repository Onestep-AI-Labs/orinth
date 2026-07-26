# Spec: Phase 15 LLM Serving, Chat Inference, and Export

## Status

Implemented

## Goal

Complete the LLM lifecycle after fine-tuning (phase 14): export a model as a LoRA adapter zip, merged safetensors, or quantized GGUF; serve GGUF models through a managed llama.cpp server subprocess; and chat with the served model in a streaming chat surface that is deliberately differentiated from the form-based vision/NLP inference page. Users can therefore serve, inference, and download/export their fine-tuned LLMs.

## Scope

In:

- Export jobs: `adapter_zip`, `merged_16bit`, `gguf_q4_k_m`, `gguf_q5_k_m`, `gguf_q8_0`, `gguf_f16`, with download links.
- `ServingService` managing a single llama.cpp server subprocess (start/stop/status, port allocation, readiness, idle timeout, orphan reaping).
- SSE chat proxy endpoint and a new streaming chat surface at `/inference/chat`.
- Export and serving panels on the model detail page (phase 12).

Out:

- Serving safetensors directly via `transformers.generate` — llama.cpp-only serving MVP; non-GGUF models route through export first.
- Multiple concurrent serving sessions.
- Persisted chat history (transcripts are in-memory per browser session).
- DB changes — serving state is in-memory plus a state file; no Alembic migration in this phase (a ServingSession table would buy a migration with no queryable-history need behind it).

## Interfaces

- `POST /api/models/{id}/export`
- `GET /api/models/{id}/exports`
- `GET /api/models/{id}/exports/{export_id}/download`
- `POST /api/serving/start`
- `POST /api/serving/stop`
- `GET /api/serving/status`
- `POST /api/serving/chat` (SSE stream)
- New router `backend/app/api/routers/serving.py`; export endpoints join the phase-12 `models.py` router; `ServingService` singleton in `backend/app/container.py`.
- Schemas: `ModelExportRequest` (`format`), `ModelExportStatus`, `ServingStartRequest` (`model_id`, `context_length?`, `n_gpu_layers?`), `ServingStatus`, `ChatRequest` (`messages`, `system?`, `temperature?`, `top_p?`, `max_tokens?`) in `backend/app/schemas.py`.
- Frontend surfaces: new route `frontend/app/(platform)/inference/chat/page.tsx` with feature `frontend/features/inference/chat/chat-page.tsx`; export/serving panels in `frontend/features/models/llm-panels.tsx` rendered by `model-detail-page.tsx`; cross-link "Chat with a served LLM" on the inference landing; export methods added to `frontend/lib/api/models.ts` and a new `frontend/lib/api/serving.ts` (the SSE chat stream is a `ReadableStream` generator, not `jsonFetch`).
- Storage/DB changes: export artifacts under `storage/trained_models/<model_id>/exports/<export_id>/` (or `storage/uploaded_models/...` for uploads); converter tooling under `storage/tools/llama.cpp/<tag>/`; serving state file `storage/serving/state.json`. No DB change.

## Behavior

### Export

- `POST /api/models/{id}/export` accepts `llm_adapter` and `llm_hf` models (and `llm_gguf` for re-quantization) and runs on the shared evaluation-style executor as a job with status/progress surfaced through `GET /api/models/{id}/exports`; exports and training share nothing at runtime, so they do not compete for the training executor slot.
- Formats: `adapter_zip` zips the PEFT adapter directory; `merged_16bit` merges the adapter into its base via PEFT `merge_and_unload` executed in a subprocess (memory isolation — a 4B merge peaks well above the API process budget) and saves safetensors; `gguf_*` runs merge (if needed) → HF-to-GGUF f16 conversion → llama.cpp quantization as one pipeline.
- GGUF tooling: `llama-cpp-python` (already in the phase-14 `llm` extra) provides prebuilt wheels including Metal and the quantize API for `q4_k_m`/`q5_k_m`/`q8_0`; HF→GGUF conversion uses a version-pinned, checksum-verified `convert_hf_to_gguf.py` fetched on first use into `storage/tools/llama.cpp/<tag>/` together with the pip `gguf` package. Rationale: pip-only install, no user-side compilation, no AGPL code. **Pinned tag: `b9000`** (`app/ml/llm/gguf_tools.py`, sha256 `3ff05b62f65c16c1864ee3439b692aa6f184f5e18356f94ae2ebe1c7427644d4`). This is the last llama.cpp release whose converter is a single self-contained script — later tags split it into a repo-local `conversion/` package that cannot be fetched as one verifiable file. b9000 registers the spec-required Qwen3.5 (`Qwen3_5ForConditionalGeneration`/`…MoeForCausalLM`), Gemma 4 (`Gemma4ForConditionalGeneration`), and Ministral 3 (`Ministral3ForCausalLM`) architectures, plus the phase-14 catalog families (Qwen2.5, SmolLM2, Gemma 3). The script runs as a subprocess with `NO_LOCAL_GGUF=1` so it uses the pip `gguf` package rather than a colocated `gguf-py/`.
- SentencePiece vocab preservation (Gemma/Llama GGUF fix): the b9000 converter's Gemma path uses the SentencePiece vocab only when `tokenizer.model` is present in the model dir, else it falls back to the gpt2 BPE path, which asserts (`max(tokenizer.vocab.values()) < vocab_size`) and aborts export. A fast tokenizer's `save_pretrained` writes `tokenizer.json` but not `tokenizer.model`, so a merged (or full-fine-tuned) SentencePiece model loses it. `gguf_tools.ensure_sentencepiece_vocab` restores it from the adapter dir or the base (local dir or Hub) after both the training save and the export merge, so GGUF export of Gemma succeeds; it is a no-op for BPE-only models (Qwen, SmolLM2).
- Disk preflight before merge/conversion (merged 16-bit models run 2×–4× the base download); failure names required-vs-available sizes.
- Extras preflight: before launching, the service checks the optional deps a format needs (`peft` for any merge; `gguf` + `llama-cpp-python` for GGUF) and fails with an `uv sync --extra llm` hint, so a missing dependency surfaces as an actionable error rather than a mid-job subprocess traceback.
- Export artifacts write to a temp dir with an atomic move; an interrupted export leaves no partial artifact and reports failed with the stderr tail.
- Completed `gguf_*` exports register the artifact as an `llm_gguf` model (source `trained`, `base_model_id` preserved) so it appears in the model catalog and is directly servable; `adapter_zip`/`merged_16bit` are download-only artifacts listed on the model detail page.
- The models page LLM cards explain the artifact rather than showing a stale gate: `llm_adapter` reads "LoRA adapter — the tuned delta over its base; export to GGUF to serve", with Export / Serve actions routing to the detail page; a completed LLM training run's detail page surfaces a "Save / Export" action linking to the registered adapter. LLM models are label-free (the registry no longer falls back to the medical `DEFAULT_LABELS` for `llm_*` families, which had made an adapter card render "granuloma / kista").
- A completed LLM training run's detail page links "Export" to the registered adapter's model detail page with the export panel focused — the same arrive-from-a-run handoff Unsloth Studio does with its `/export?run=` search param, expressed here as a route link because the export panel lives on the model detail page rather than a standalone export route.
- Base-model licenses ride along: exports of Gemma-based models copy the license notice into the artifact directory.

### Serving from a custom path

- Besides a registered `llm_gguf` model, `POST /api/serving/start` accepts a `model_path` to a `.gguf` file anywhere on the local machine (exactly one of `model_id`/`model_path`). The session id for a path is `path:<resolved>` so a repeat start of the same file is recognised.
- `GET /api/serving/scan?path=…` lists servable GGUF files at a path: a single file, or a directory walked (bounded to 200 hits) for `*.gguf`. Registered GGUFs are tagged with their catalog id so the UI prefers them; non-GGUF Hugging Face folders (config.json + safetensors) are reported separately as export candidates. This is a local single-user convenience (localhost-only serving, per the platform's stated auth model) — arbitrary-path serving is not exposed to a network deployment.
- `GET /api/serving/browse?path=…` returns one directory level (subfolders + GGUF files) — retained as a fallback path lister.
- Native OS picker: `POST /api/serving/pick {kind: folder|file}` opens the platform's own file dialog with **stdlib only** (macOS `osascript`, Linux `zenity` via `subprocess`; no third-party dependency) and returns the chosen absolute path — the LM-Studio-style picker. When no native dialog is available it returns `unavailable: true` and the UI falls back to manual path entry. Only the localhost single-user machine can do this, consistent with the platform's auth model.
- Persisted models directory: `GET/PUT /api/serving/config` stores the last-used models directory in `storage/serving/config.json`; a saved dir that no longer exists is dropped on read. The launcher restores it on load and **auto-scans** it, so a returning user sees their GGUF files without re-picking.
- Hugging Face catalog: `GET /api/serving/hf/search?query=&format=gguf|mlx` lists downloadable repos (downloads/likes), `GET /api/serving/hf/files?repo_id=&format=` lists a repo's GGUF/MLX files with sizes and quant tags, and `POST /api/serving/hf/download {repo_id, filename}` downloads one file into `storage/serving/models/` (a gated 403 returns an actionable message). GGUF downloads are served immediately; MLX is download-only (the llama.cpp runtime cannot serve MLX).
- Frontend: the inference LLM launcher (`LlmServeView`) is a single Serve panel (no Chat panel) with three source tabs — **Trained models** (registered `llm_gguf`, shown with file sizes), **Local folder** (OS-native Browse folder / Browse .gguf buttons + a path field that auto-scans and is persisted), and **Hugging Face** (search GGUF/MLX, expand a repo to its files, download, then serve). Starting a server from any tab **auto-redirects to `/inference/chat`**, since `POST /serving/start` returns only once the server is ready; the old "Open chat" launcher panel is gone (a small Open-chat link remains only while a session is already running, beside Stop).
- Serving requires the `llama-cpp-python[server]` extra (pinned in the `llm` optional-dependency group): the base bindings lack `sse-starlette`/`starlette-context`, so `python -m llama_cpp.server` otherwise crashes at import. Install with `uv sync --extra llm`.

### Serving

- `ServingService` manages exactly one active session: `POST /api/serving/start` with a new model stops the previous session first (stated memory constraint on a single-user machine), launches `python -m llama_cpp.server` as a subprocess with the model path, context length, and `n_gpu_layers` (auto: all layers on Metal/CUDA, 0 on plain CPU), and returns once ready.
- Port allocation walks a configurable range (default 8600–8699) and binds the first free port; the server listens on localhost only.
- Readiness is polled against the server's `/v1/models` with a timeout; a crash before ready surfaces the stderr tail in the start error.
- Only `llm_gguf` models are servable. `llm_adapter` and `llm_hf` detail pages show an "Export to GGUF first" call to action instead of a start button — one serving runtime keeps the memory and failure model simple.
- Idle timeout (configurable, default 15 minutes) stops the server after no chat activity; activity timestamps update per streamed token so a long generation never counts as idle.
- Orphan reaping: the subprocess pid and port persist to `storage/serving/state.json`; app startup and shutdown both check the file and terminate any recorded process that still runs. In-memory state is authoritative during normal operation — the file exists solely so a crashed API process cannot leak a llama.cpp server.
- `GET /api/serving/status` reports `stopped | starting | running | stopping`, the served model, port, uptime, and last-activity time; the frontend polls it with the standard refetch interval.

### Chat streaming

- `POST /api/serving/chat` proxies the llama.cpp server's streamed chat completion and re-emits it as Server-Sent Events; the frontend consumes the stream via fetch `ReadableStream`. SSE-over-fetch is chosen over websockets deliberately: the platform has a no-websockets/polling convention, and chat needs only one-directional streaming, which SSE provides without a new transport.
- The proxy exists so the browser only ever talks to the API origin — the llama.cpp port stays an internal implementation detail. Because the config `/api/*` rewrite buffers streaming responses (tokens would arrive all at once), the chat request goes through a dedicated Next route handler `frontend/app/api/serving/chat/route.ts` that returns the upstream `ReadableStream` directly and forwards the client abort signal; it sits at the exact path so it wins over the catch-all rewrite.
- The stream carries token deltas, a final usage event (prompt/completion tokens, tokens/s), and an error event when the upstream dies mid-stream; client-side abort cancels the upstream request so generation stops server-side too.
- Requests pass `messages`, optional system prompt, `temperature`, `top_p`, `max_tokens`; the served model's chat template is applied by llama.cpp from GGUF metadata.

### Web search (chat "Search" mode)

- `POST /api/serving/web-search {query, max_results}` runs a keyless DuckDuckGo search via the stdlib-friendly `ddgs` package (added to the `llm` extra) and returns title/url/snippet hits; failures come back as an `error` string, never a 500, so chat still answers ungrounded.
- Chat flow: a composer **Search** chip toggles the mode. When on, `send()` first fetches results, renders them as a numbered **Sources** citation block on the assistant turn (title link + host), and grounds the model by appending the results to the system prompt with an instruction to cite `[n]`. A **Thinking** chip (same chip row, mirroring the reference's Search/Code chips) toggles whether the model's `<think>` reasoning is shown.

### Shared model picker

- `features/inference/model-source-picker.tsx` is one reusable tabbed picker — **Trained** (registered `llm_gguf`, with sizes) · **Local** (a path field, persisted + auto-scanned into a *selectable* GGUF list, plus OS-native folder/.gguf dialogs) · **Hugging Face** — used by both the chat rail ("Select model" / "Switch model") and the inference serve view, so model selection is one consistent surface. Every source resolves to a `ServingStartRequest`. Options stay selectable while a model runs (starting a new one stops the old), fixing the earlier all-disabled state.
- Hugging Face tab: `GET /serving/hf/recommended` returns curated, downloadable `bartowski/*-GGUF` chat models with a Q4_K_M size estimate, ranked by whether they fit detected memory (unified memory on Apple/CUDA, RAM on CPU, with context headroom) — shown as a "Recommended for your hardware" list with a fits/tight badge before any search. `list_models` no longer passes the removed `direction` kwarg (huggingface_hub 1.x); `sort="downloads"` is descending by default.

### LLM held-out testing

- The testing page (phase 2 evaluation) gains an `llm_finetune` path so a fine-tuned adapter or merged HF checkpoint can be scored on a held-out split. `EvaluationService.list_datasets` now surfaces `instruction_jsonl`/`chat_jsonl` datasets (their `test` split), and `_evaluate_llm` resolves the base + adapter (shared `app/ml/llm/model_paths.py`, the same resolver export uses), then spawns `python -m app.training.runners.llm_eval` for memory isolation.
- Metrics are the standard held-out LLM set — perplexity, mean cross-entropy loss, and next-token accuracy over the assistant tokens only (reusing the SFT completion-masking pipeline so the scored tokens match what training optimized). Per-record perplexity/loss populate the per-item table.
- GGUF models are excluded from batch testing (tested interactively via serving/chat instead) — the testing model picker omits `llm_gguf`, and `_evaluate_llm` rejects it with a clear message. The frontend adds an `llm` evaluation display kind (perplexity / token-accuracy / loss columns; per-record table).

### Training download progress

- LLM training streams Hugging Face base-model download progress to the live log. The training subprocess reader now splits output on `\r` as well as `\n` (`iter_process_lines`), so tqdm/hf progress bars — which redraw in place with `\r` and no newline — surface during the multi-minute download instead of appearing only at completion. The runner also logs an explicit "Fetching base model…" line up front.

### Chat UI

- New route `frontend/app/(platform)/inference/chat` — a separate surface, not a tab on the existing form-based inference page, because the interaction model (streaming conversation vs submit-and-render-overlay) shares nothing with it. The inference landing cross-links both ways, and the cross-link is task-conditional: it renders only when the `llm_finetune` task is selected and at least one servable GGUF model exists, never on vision/NLP tasks.
- The inference landing itself branches on the selected task: for `llm_finetune` it replaces the submit-and-render form with a Serve/Chat launcher (`LlmServeView`) — pick a servable GGUF model, start/stop the managed server, then "Open chat" once running — since form-based inference does not apply to a streaming LLM. Only `llm_gguf` models are offered; when a project has none, the launcher points to the models page to export one. The actual conversation still happens only on `/inference/chat`.
- Layout: left controls rail (served-model status with start/stop, model picker over servable `llm_gguf` entries, temperature/top_p/max_tokens, system prompt textarea) and a transcript column with a composer pinned at the bottom. Stop-generation is available while streaming; the footer shows tokens/s and time-to-first-token for the last response.
- Transcript rows are hairline-separated with uppercase micro-label role headers — no chat bubbles, no shadows, per the design contract; the assistant's streaming caret animates 120–200ms ease-out and honors `prefers-reduced-motion`; code blocks render in the mono stack at 13px; the send button is the `--ink` primary with accent reserved for focus rings.
- Serving errors and mid-stream failures surface in the transcript and the toast layer — never silently swallowed.
- Medical context: the chat surface carries the platform's research-instrument framing. No diagnosis language, and no medical-assistant system-prompt preset ships with the UI.

### Design and Studio UI adoption

- Unsloth Studio is the UX reference for both surfaces (anatomy only; AGPL code never copied; styling re-expressed in `frontend/DESIGN.md` tokens):
  - The export panel adopts Studio's Export page anatomy (`features/export/export-page.tsx`): a format choice group with a quantization select for GGUF, a live log tail streaming while an export job runs, and a per-export artifact list with status, size, and download actions.
  - The chat surface adopts Studio's Chat page anatomy (`features/chat/chat-page.tsx`): transcript column with pinned composer and stop-generation, plus a settings rail carrying what Studio keeps in its chat settings sheet — served-model status with start/stop, sampler controls, system prompt. Studio's thread sidebar is not adopted because persisted transcripts are out of scope; if chat history lands later, that sidebar is the reference anatomy.
- No new tokens, no shell change; this is the heaviest UI of the suite and stays within existing vocabulary: hairline structure, `--shadow-overlay` only for the model picker popover, status pairs for serving state badges (`running` = success pair, `starting` = info, errors = danger).

## Edge Cases

- Port range exhausted or collision at bind: next free port; all taken → actionable error naming the range setting.
- Server crash mid-stream: SSE error event terminates the transcript entry; status flips to `stopped`; stderr tail retrievable from status.
- Model too large for available memory: llama.cpp exits during load; start fails with the stderr tail and a smaller-quant suggestion.
- Concurrent start requests: service lock serializes; the loser gets the winner's session in the response.
- Idle timeout firing during an active stream: prevented by per-token activity updates.
- Export requested while the same model is being served: allowed — exports read, serving reads; no write conflict.
- Converter tag lacking support for a model architecture: conversion fails with the upstream error and the spec-recorded tag named, rather than a silent bad GGUF.
- App shutdown with a live server: lifespan hook terminates the subprocess; state file cleaned.

## Acceptance Criteria

- A phase-14 `llm_adapter` model can be exported to `adapter_zip`, `merged_16bit`, and `gguf_q4_k_m`; the GGUF export appears as a servable `llm_gguf` model and all artifacts download.
- Starting serving on a GGUF model reaches `running` with a health-checked llama.cpp subprocess; starting a second model stops the first; stop and idle timeout both reap the subprocess; killing the API process and restarting reaps the orphan.
- The chat page streams tokens live, supports stop-generation, shows tokens/s, and surfaces mid-stream failures.
- Non-GGUF LLM models show the "Export to GGUF first" CTA instead of a start button.
- Backend tests cover: export pipeline orchestration (subprocess-mocked), format validation, GGUF registration, port allocation, single-session semantics, state-file orphan reaping, SSE proxy framing (upstream-mocked), idle-timeout activity accounting.
- Manual check: full loop — fine-tune (phase 14) → export GGUF → serve → chat → download — on an Apple Silicon machine with Metal.

## Deferred

- Serving safetensors via transformers/vLLM runtimes.
- Multi-session serving and remote GPU serving.
- Persisted chat transcripts and shareable conversations.
- OpenAI-compatible passthrough endpoint for external clients.

## Validation

Commands to run:

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```

Manual smoke (requires `llm` extras and a GGUF export): start serving from the model detail page, chat at `/inference/chat`, confirm streaming, stop-generation, idle timeout, and artifact downloads.
