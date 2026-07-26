# Spec: Phase 14 LLM Fine-Tuning

## Status

Implemented

## Goal

Fine-tune small (1B–4B) instruction LLMs on `llm_finetune` datasets (phase 10) inside the existing training-job platform, with a dual backend: Unsloth 4-bit QLoRA when CUDA is available, plain transformers + PEFT LoRA on MPS/CPU so training works on Apple Silicon. Initial catalog: Qwen3.5 2B/4B, Gemma 4 E2B/E4B, Ministral 3 3B, plus any locally registered `llm_hf` model (phase 12). Completed runs auto-register a LoRA adapter model that phase 15 exports and serves.

## Scope

In:

- `uv` optional dependency groups for LLM training and the transformers upgrade.
- LLM environment detection endpoint.
- LLM model catalog (five hub options + dynamic local bases) and base-asset preparation.
- Subprocess runner `llm_sft` with Unsloth/PEFT backend selection, TRL `SFTTrainer` loop, phase-3 progress contract.
- Chat-template data pipeline over phase-10 `data.jsonl` files.
- Auto-registration of the resulting adapter; LLM advanced parameters via the phase-13 mechanism.

Out:

- Serving, chat inference, export (phase 15).
- RLHF/DPO/GRPO, full fine-tuning, vision fine-tuning.
- Resume-from-checkpoint.
- DB changes — jobs reuse `TrainingJob`; no Alembic migration in this phase.

## Interfaces

- `GET /api/training/llm/environment`
- `GET /api/training/llm/model-info?model_ref=<option id | hub id>` — accurate base-model detail from the Hugging Face Hub API (real size, file count, gating, downloads/likes); resolves catalog option ids and local base ids, and reports local bases' on-disk size.
- Existing `POST /api/training/jobs` with `task_type: "llm_finetune"` and family `llm_sft`.
- Existing `GET /api/training/model-options?task_type=llm_finetune`.
- Existing `POST /api/training/model-assets/prepare` reused for base-model downloads.
- Schemas: `LlmEnvironment` (`device: cuda | mps | cpu`, `unsloth_available`, `peft_available`, `bitsandbytes_available`, `recommended_backend`, `notes`) and `LlmModelInfo` (`model_ref`, `exists`, `gated`, `size_bytes`, `file_count`, `downloads`, `likes`, `library`, `error`) in `backend/app/schemas.py`.
- New modules: catalog `backend/app/ml/llm/catalog.py`; environment probe `backend/app/ml/llm/environment.py`; runner `backend/app/training/runners/llm_sft.py`, dispatched from `_command_for_job` in `backend/app/services/training/service.py`.
- Frontend surfaces: training form gains an environment banner and gated-option badges for `llm_finetune`; LLM advanced parameters render through the phase-13 accordion; detail page adds sample generations.
- Storage/DB changes: run artifacts under `storage/training_runs/<job_id>/` (existing); adapter output registered into `storage/trained_models/` + `storage/model_registry.json` (existing paths). No DB change.

## Behavior

### Dependencies (load-bearing)

- `backend/pyproject.toml` gains two `uv` optional groups:
  - `llm`: `peft`, `trl`, `accelerate`, `datasets`, `sentencepiece`, `gguf`, `llama-cpp-python` — installable on every platform, including macOS.
  - `llm-cuda`: `unsloth`, `bitsandbytes` — separated because these are only reliably installable on CUDA platforms; keeping them out of `llm` is what lets a Mac install the feature at all.
- The core `transformers` pin is raised to `>=5.5.0`, the first release with Gemma 4 support (verified at implementation; the lockfile resolves 5.13.1). Acceptance criterion: the existing BERT/BART runner tests pass after the upgrade, since those runners share the dependency.
- The backend degrades gracefully without the extras: LLM catalog options still list (so the UI can explain the feature) but are gated `runnable: false` with an "install the `llm` extras" note; `POST /api/training/jobs` for `llm_sft` without importable `peft`/`trl` is rejected with the same install hint. All LLM imports stay inside function bodies per the lazy-import rule.
- The vendored `unsloth/` checkout at the repo root is reference material only. The `unsloth` pip package (Apache-2.0) is the dependency; Unsloth Studio code (`unsloth/studio/`, AGPL-3.0) must never be copied — its patterns (format detection, chat-template application, subprocess worker) are re-implemented natively here. Add `unsloth/` to the root `.gitignore` so the checkout cannot be committed by accident.

### Environment detection

- `GET /api/training/llm/environment` probes, with lazy imports inside the handler: torch device (cuda > mps > cpu), importability of `unsloth`, `peft`, `bitsandbytes`, and returns `recommended_backend` (`unsloth` iff CUDA + unsloth importable, else `peft`) plus human-readable notes (e.g., "4-bit QLoRA unavailable on MPS"). The training form renders this as an info banner so users know what a run will actually do before submitting.

### Catalog

- `backend/app/ml/llm/catalog.py` declares family `llm_sft`, task `llm_finetune`, source `huggingface`, with options (model ids verified downloadable at implementation — the draft's `Qwen3.5`/`Gemma 4`/`Ministral 3` names were aspirational and 404 on the Hub, so real current small instruct models are used instead):
  - `qwen2_5_1_5b` — `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, ungated — the default)
  - `smollm2_1_7b` — `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Apache-2.0, ungated — no-login fallback)
  - `qwen2_5_3b` — `Qwen/Qwen2.5-3B-Instruct` (Apache-2.0)
  - `gemma3_1b` — `google/gemma-3-1b-it` (`gated_license: true`)
  - `gemma3_4b` — `google/gemma-3-4b-it` (`gated_license: true`)
- Each option carries memory guidance in its description (approximate CUDA VRAM for QLoRA and unified-memory needs for MPS LoRA) so users can self-select; 4B-class models on MPS are flagged as needing a high-memory machine.
- A dynamic option per registered `llm_hf` model (phase 12 uploads and prior merges) lists under a "Local base models" group, satisfying "fine-tune using local model or from HuggingFace" — the runner receives its filesystem path instead of a hub id.
- A `llm_hf_custom` option ("Custom Hugging Face model…") lets the user fine-tune **any** hub checkpoint by typing its id, not only the five catalog bases. Its ref rides the job's `base_model` field; job creation rejects an empty id, and `POST /model-assets/prepare` accepts a `model_ref` so the same disk-preflight / gated-403 flow covers custom downloads.
- Only the Hugging Face Transformers family (`llm_hf`, i.e. safetensors/PyTorch + config.json + tokenizer) is a trainable base — `TRAINABLE_BASE_FAMILIES` / `TRAINABLE_FORMAT_NOTE` in the catalog. GGUF and TFLite are quantized inference/export formats that expose no trainable modules and are surfaced to the user as non-trainable (GGUF is a phase-15 export target, not a training input).
- Base downloads reuse `POST /api/training/model-assets/prepare`: HF token aliases exported as today, cache under `storage/model_assets/`, plus a disk-space preflight — bases run 2–8 GB each, and failing before download beats failing at 90%.
- Gemma options require an accepted license on huggingface.co; a 403 during preparation returns an actionable message naming the model page, same pattern as phase-10 gated datasets.

### Runner

- `backend/app/training/runners/llm_sft.py` is a subprocess entrypoint dispatched from `_command_for_job`, launched with `PYTHONUNBUFFERED=1` — restating the phase-3 rationale: without it the child block-buffers stdout and live logs arrive in one burst at exit.
- Download visibility: the runner pre-fetches the base via `snapshot_download` and reports progress itself rather than relying on the library's bars. The framing line prints first, before any network call (a blocking metadata lookup previously left the log silent for the whole download — the reported "stuck" symptom). Xet and the built-in bars are disabled so blobs stream to a growing `.incomplete` file, and a background thread polls the model's cache folder, printing throttled `downloading base model: 47% (2.1 GB/4.6 GB)` lines (raw bytes when the size lookup is slow). Total size is fetched in that same background thread so it never blocks. The prefetch populates the cache `from_pretrained` reads (a cache hit follows); local paths skip it; any failure is non-fatal (the load re-raises the authoritative 404/gated error).
- Verbose progress: the runner emits numbered phase lines — `[1/4] Dataset ready`, `[2/4] Downloading base model`, `[3/4] Training …`, `[4/4] Saving …` — and the service turns per-step metrics into a readable header (`Training — step 12/40 · loss 1.83 · val loss 1.90`) instead of echoing a raw log line. `results.csv` is still rewritten per logging step, so the loss/eval curves update live during the run.
- Device selection mirrors `runners/nlp/huggingface.py`: cuda > mps > cpu with `ONESTEP_TRAIN_DEVICE` override.
- Backend selection: Unsloth `FastLanguageModel` with `load_in_4bit` QLoRA when the device is CUDA and `unsloth` imports; otherwise transformers `AutoModelForCausalLM` + PEFT `LoraConfig`/`get_peft_model` with fp16/bf16 where supported and gradient checkpointing on MPS. Both paths hand the model to TRL's `SFTTrainer`, so the training loop, metrics, and artifacts are backend-independent.
- Progress follows the phase-3 contract: the runner rewrites `results.csv` at every logging step (not per epoch — LLM runs are step-denominated), mapping `processed/total` to `current_step/max_steps` (or steps-per-epoch × epochs when `max_steps` is unset). A runner that writes results only at completion leaves the progress bar pinned for hours on LLM-scale runs.
- Token accuracy: when TRL's `SFTTrainer` reports `mean_token_accuracy` / `eval_mean_token_accuracy` (the fraction of predicted completion tokens that match the reference), the runner writes them as the `accuracy` / `val_accuracy` columns in `results.csv` and `val_accuracy` in `metrics.json`. This drives the detail page's accuracy curve (rendered to the right of the loss curve). Guarded — older TRL/transformers versions that omit these keys simply produce no accuracy column and no accuracy panel.
- Dataset pipeline: the runner reads the split `data.jsonl` files (phase 10). `chat_jsonl` records go through `tokenizer.apply_chat_template`; `instruction_jsonl` records are templated into messages first (instruction+input as the user turn, output as the assistant turn) and then through the same path, so both formats share one code path. Records exceeding `max_seq_length` are truncated and counted; the count lands in the log tail. Tokenization goes through the `template_token_ids` helper, which requests `return_dict=True` and reads `input_ids` — necessary because transformers 5.x returns a `BatchEncoding` (not a bare list) from `apply_chat_template(tokenize=True)`, and `list()`-ing that yields the dict keys, which otherwise collapsed every example to two tokens and dropped them all as fully-masked.
- Train-on-completions masking (loss on assistant tokens only) is the default; a `train_on_full_sequence` advanced flag disables it. Implementation note: records are pre-tokenized in the runner (prompt-prefix label masking, template-agnostic) and handed to `SFTTrainer` with dataset preparation skipped, which is what makes truncation countable and masking independent of any template's generation markers. The `packing` flag is accepted but currently logged and trained unpacked — packing conflicts with prompt masking in the pre-tokenized pipeline.
- Split resolution: the runner reads the `train` split; if it is empty but the `unassigned` inbox has records (the shape a recipe commit or fresh upload produces), the inbox becomes the training pool and a small deterministic validation holdout (~15%) is carved from it — but only when that leaves ≥10 training records, otherwise the whole pool trains and eval is skipped. This means a freshly committed dataset trains without a manual Splits step; a real `train` split, when present, is always preferred and an existing `valid` split is never disturbed.
- Minimum dataset size: fewer than 10 usable train records (after the resolution above) fails fast with a clear error — an SFT run on a handful of rows silently overfits into garbage and wastes an hour doing it.
- Artifacts under `storage/training_runs/<job_id>/`: `adapter/` (PEFT format) + tokenizer files, `results.csv`, `metrics.json` (final train/valid loss, perplexity when valid exists), `sample_generations.json` (fixed prompts drawn from the valid split, generated by the just-trained adapter) so the detail page can show qualitative output next to curves.
- Completion auto-registers a model: family `llm_adapter`, `source: "trained"`, `base_model_id` pointing at the catalog option or local base, artifacts copied to `storage/trained_models/<slug-id>/` — the same auto-registration flow YOLO/Keras use, and the input phase 15 exports and serves.
- Cancel uses the existing subprocess terminate flow; backend restart marks queued/running jobs failed (existing phase-3 rule, unchanged).
- Concurrency: the training executor is single-worker by configuration, which serializes LLM runs and is the platform's protection against two jobs contending for one GPU or one machine's unified memory. The spec depends on that setting staying 1 for LLM workloads.

### Fine-tuning methods

- `finetune_method` (catalog `Method` group; surfaced as a prominent basic select, not buried in the accordion): `lora` (default) | `qlora` | `full` | `continued_pretrain`.
  - `lora` — PEFT LoRA adapter, no quantization. Registered as `llm_adapter`.
  - `qlora` — LoRA over a 4-bit (nf4, bitsandbytes) base; the only 4-bit method. On non-CUDA it downgrades to plain LoRA with a log note. Registered as `llm_adapter`.
  - `full` — full fine-tune: no PEFT, every weight updated, saved as a standalone HF model to `model/` and registered as `llm_hf` (phase 15 serves it directly, no merge). Memory-heavy; warns on non-CUDA.
  - `continued_pretrain` — LoRA with completion masking off (loss on the whole sequence), for causal-LM continuation rather than instruction following. Registered as `llm_adapter`.
- The runner branches on method: adapter methods save `adapter/`; `full` saves `model/`. `find_best_model` and `_register_training_model` pick the produced directory and register the matching family.

### Advanced parameters (via the phase-13 mechanism)

- Groups: Method — `finetune_method` (see above); LoRA — `lora_r` (default 16), `lora_alpha` (32), `lora_dropout` (0.05), `target_modules` (multiselect, default all attention+MLP projections); Quantization — `load_in_4bit` (default true, only active for QLoRA on CUDA; downgraded with a log note elsewhere); Sequence — `max_seq_length` (2048), `packing` (false); Optimization — `learning_rate` (2e-4), `lr_scheduler` (`linear | cosine`), `warmup_ratio` (0.03), `weight_decay` (0.01), `per_device_batch_size` (2), `gradient_accumulation_steps` (4), `epochs` / `max_steps`, `seed` (42); Runtime — `precision` (`bf16 | fp16 | fp32`, device-guarded), `gradient_checkpointing` (true), `train_on_full_sequence` (false).
- The learning-rate soft warning gains an LLM rule: rates above `1e-3` are flagged for LoRA fine-tuning (LoRA tolerates larger rates than full fine-tuning, so the BERT-era `1e-4` threshold would false-positive on every default run).
- The learning-rate soft warning gains an LLM rule: rates above `1e-3` are flagged for LoRA fine-tuning (LoRA tolerates larger rates than full fine-tuning, so the BERT-era `1e-4` threshold would false-positive on every default run).

### Design and Studio UI adoption

- Unsloth Studio's train surface is the UX reference (anatomy only; no AGPL code, styling per `frontend/DESIGN.md`): the LLM training form uses the phase-13 Configure grid (Model | Dataset | Parameters | Run), and creating a job navigates straight to the training detail page — the platform's equivalent of Studio auto-switching to its Current Run tab when training starts.
- The training detail page adopts Studio's live-training-view anatomy (`features/studio/live-training-view.tsx`, `sections/progress-section.tsx`, `charts-section.tsx`) for LLM runs: a progress header (phase, step/total, ETA, tokens seen), a charts panel (train/eval loss over steps), the log tail, and — after completion — a results state that stays on the page with final metrics, sample generations, and next-step links (Export / Serve arrive with phase 15). The existing job table remains the history surface, mirroring Studio's History card grid at our density.
- Environment banner uses the `info` `-tint`/`-strong` pair; gated options show a lock badge; loss/eval curves use the `--label` ramp; sample generations render in the mono stack at 13px. No new tokens, no shell change.
- Device is auto-detected from the environment probe: the LLM advanced defaults for `load_in_4bit` and `precision` are seeded to match the device (off/fp32 on MPS/CPU, on/bf16 on CUDA) so the shown knobs equal what actually runs. The vision training form's Device control is a dropdown (Auto — showing the detected accelerator — plus the detected accelerator and CPU) rather than free text, so an invalid device string can't reach a run. The probe endpoint is reused for both surfaces (fetched for vision and LLM, never NLP).
- Base-model detail is fetched live from the Hugging Face API when an LLM base is selected (catalog/local option id, or the debounced custom hub id): a token-only caption shows real download size, file count, downloads, and a "license required" marker. Gating is driven by this live value (falling back to the catalog's static `gated_license`), so a genuinely gated model is flagged even when its static flag is off. An unknown or gated id surfaces the API's message inline.

## Edge Cases

- Extras not installed: options gated, job creation rejected with install hint.
- Gemma 403 without accepted license: actionable message naming the model page.
- Disk preflight failure: preparation rejected with required-vs-available sizes.
- Empty train split but records in the `unassigned` inbox: the inbox is used as the training pool with a validation holdout (see Split resolution). Only a dataset with fewer than 10 usable records anywhere fails fast, naming both "add more records" and the phase-10 split flow.
- MPS out-of-memory on a 4B model: caught, failed with a message naming memory guidance and suggesting the 2B-class options or smaller `max_seq_length`/batch.
- CUDA present but `bitsandbytes` broken: backend falls back to PEFT fp16 with a log note rather than crashing on import.
- `chat_jsonl` record whose tokenizer lacks a chat template: runner applies a default ChatML-style template and logs it.

## Acceptance Criteria

- With extras installed on a CUDA machine, a Qwen3.5-2B QLoRA job runs via Unsloth; on an MPS machine the same job definition runs via PEFT LoRA — verified by the backend name in the run log.
- Progress, log tail, loss curves, and ETA update live during a run (results.csv per logging step).
- A completed run registers an `llm_adapter` model with `base_model_id` and shows sample generations on the training detail page.
- Existing BERT/BART runner tests pass after the transformers upgrade.
- Without extras, LLM options list as gated and job creation returns the install hint.
- Backend tests cover: environment probe shapes, catalog gating, command generation for `llm_sft`, custom-hub command/prepare/validation, trainable-format note, dataset pipeline templating for both formats (tokenizer-mocked), truncation counting, minimum-row rejection, auto-registration payload.

## Deferred

- Full fine-tuning and continued pretraining.
- DPO/GRPO preference training.
- Resume-from-checkpoint; multi-GPU selection.
- Evaluation harness beyond valid-loss/perplexity/sample generations.

## Validation

Commands to run:

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```

Manual smoke (requires extras): `cd backend && uv sync --extra llm` (plus `--extra llm-cuda` on CUDA), then run a small `llm_sft` job against a sample `llm_finetune` dataset and confirm live progress and adapter registration.
