# Spec: Phase 13 Advanced Training Settings

## Status

Implemented.

Deviations from the draft, all made to avoid double-mapping a value that is
already a basic field or to match the actual runner surface:

- **YOLO advanced excludes `optimizer`, `lr0`, and `patience`.** Those remain
  basic top-level `TrainingJobCreate` fields already threaded to the runner;
  re-declaring them as advanced keys would submit the same value twice. The
  advanced set carries the remaining Optimization/Runtime knobs (`lrf`,
  `momentum`, `weight_decay`, `warmup_epochs`, `close_mosaic`, `cos_lr`,
  `seed`) plus the full augmentation group, each a direct Ultralytics
  `model.train(**kwargs)` argument.
- **HF advanced omits `max_seq_length`.** `max_length` is already a basic
  hyperparameter for HF options, so the advanced set adds `weight_decay`,
  `warmup_ratio`, `lr_scheduler`, `gradient_accumulation_steps`, `seed`, and
  the device-guarded `precision`.
- **Keras augmentation is implemented as new `tf.keras` layers** (RandomFlip/
  Rotation/Zoom/Contrast) in the classification runner — the runner had no
  augmentation pipeline before this phase — plus `lr_schedule`, `dropout`,
  `label_smoothing`, `unfreeze_layers`, `early_stop_patience`,
  `class_weighting`, and `seed`. `optimizer` stays basic.
- **sklearn baselines** map to the existing TF-IDF path in
  `app/training/runners/nlp/baseline.py` (there is no separate "baseline
  runner"); `C` overrides the basic learning-rate-derived value when supplied.
- The Studio Configure-tab reference is UX-only; there is no
  `features/studio/studio-page.tsx` in the repo. The anatomy was re-expressed
  in `frontend/DESIGN.md` tokens as `.training-config` / `.training-config-grid`.
- Shared field rendering lives in `frontend/features/training/advanced-settings.tsx`;
  the runner-side allowlist/precision helper lives in
  `backend/app/training/runners/advanced.py`, with per-family allowlists
  derived from the catalog specs so a field and its runner argument cannot drift.

## Goal

Expose advanced training hyperparameters for every model family — deep learning, classic ML, and (in phase 14) LLM — through a single catalog-driven mechanism: model options declare typed parameter specs, the training form renders them as a grouped "Advanced settings" accordion, and values travel through the existing open-ended `hyperparameters` dict. No API shape change, no migration.

## Scope

In:

- `advanced_parameters` on the training catalog types, serialized through the existing model-options endpoint.
- Concrete advanced parameter sets for the existing families: YOLO, Keras classification, HF BERT/BART, sklearn baselines.
- Training-form accordion, catalog-default hydration, and job-detail rendering of submitted values.
- Runner consumption rules (allowlist + ignore-unknown).

Out:

- LLM parameter values — phase 14 defines them using this mechanism.
- Any new endpoint, `TrainingJobCreate` change, or DB change — values ride the existing `hyperparameters` dict; no Alembic migration in this phase.
- Hard server-side validation of parameter combinations (soft warnings only, matching the existing learning-rate pattern).

## Interfaces

- Existing `GET /api/training/model-options?task_type=...` — each option gains `advanced_parameters`.
- Existing `POST /api/training/jobs` — unchanged; advanced values are keys inside `hyperparameters`.
- Schemas: `AdvancedParameterSpec` in `backend/app/schemas.py` with fields `{key, label, type: int | float | bool | select | multiselect | text, default, min, max, step, options, help, group}`; `TrainingModelDefinition` / `TrainingModelOption` in `backend/app/ml/common/catalog.py` gain `advanced_parameters: list[AdvancedParameterSpec]` (default empty, so untouched catalogs serialize as before).
- Frontend surfaces: `frontend/features/training/training-page.tsx` (accordion), `training-detail-page.tsx` (submitted-values panel), shared field rendering in `frontend/features/training/advanced-settings.tsx`.
- Storage/DB changes: none.

## Behavior

### Mechanism

- Every catalog definition may declare `advanced_parameters`. The list is serialized verbatim through the model-options endpoint; the frontend renders fields generically from `type`/`min`/`max`/`step`/`options` and groups them by `group` — the form has no per-family knowledge, so new families (phase 14's LLM catalog) get their UI for free.
- Fields hydrate from catalog defaults exactly as basic fields already do; a value equal to its default is still submitted, so runs are reproducible from the job record alone rather than from "default at the time".
- Submitted values are plain keys in the existing `hyperparameters` dict. `TrainingJobCreate` is untouched — the dict was left open-ended for exactly this.
- Runners consume advanced keys through an explicit per-runner allowlist and log-and-ignore unknown keys: a stale frontend or an edited request must never crash a run, only produce a log note naming the ignored keys.
- Validation is soft, following the existing HF learning-rate warning: out-of-range or suspicious values render an inline warning but never block submission. Device-incompatible combinations (e.g., `bf16` on MPS) are downgraded inside the runner with a log note rather than rejected at the API — the runner is the only layer that knows the actual device.

### Parameter sets (each key maps to a named runner argument)

- YOLO (`app/ml/vision/yolo/catalog.py` → YOLO runner kwargs): `optimizer`, `lr0`, `lrf`, `momentum`, `weight_decay`, `warmup_epochs`, `patience`, `close_mosaic`, `cos_lr`, `seed`; augmentation group: `mosaic`, `mixup`, `fliplr`, `flipud`, `hsv_h`, `hsv_s`, `hsv_v`, `degrees`, `translate`, `scale`.
- Keras classification (→ Keras runner): `optimizer`, `lr_schedule` (`constant | cosine | step | plateau`), `label_smoothing`, `dropout`, `unfreeze_layers`, `early_stop_patience`, `class_weighting`, `seed`; augmentation group: flip/rotation/zoom/contrast toggles already supported by the runner's augmentation pipeline.
- HF BERT/BART (→ `runners/nlp/huggingface.py`): `weight_decay`, `warmup_ratio`, `lr_scheduler` (`linear | cosine`), `gradient_accumulation_steps`, `max_seq_length`, `precision` (`fp32 | fp16 | bf16`, device-guarded), `seed`.
- sklearn baselines (→ baseline runner): `tfidf_max_features`, `tfidf_ngram_range` (select of common ranges), `C`, `max_iter`, `class_weight`.
- Existing basic fields (epochs, batch size, learning rate, image size, device) stay where they are; this phase adds groups, it does not move fields — moving them would churn every existing test and muscle memory for no behavior gain.

### Frontend

- The training form adopts a Configure anatomy (UX inspired by Unsloth Studio's Configure tab; no such file exists in this repo): a full-width Model header (task · base model · name) above a responsive grid pairing a narrow config rail (Dataset + Run panels stacked) with the wider Parameters panel, replacing the current single stacked column. The rail keeps the short Dataset/Run cards from stranding a lone field beside the tall Parameters panel; all three sections remain present and the grid collapses to one column below 1024px. This is a layout regrouping of existing fields — no field changes ownership between basic and advanced, so existing tests and muscle memory survive.
- The Parameters card hosts a collapsed-by-default "Advanced settings" accordion, grouped by `group` with group micro-headers (Optimization, Augmentation, Regularization, Runtime). Collapsed state signals "defaults are fine", matching the platform's task-first flow.
- The job detail page renders the submitted advanced values as a definition list in a panel, so a finished run documents its own configuration.
- Changing the selected model option rehydrates the accordion from that option's defaults, discarding edits — the same behavior basic fields have today, called out so it is deliberate rather than surprising.

### Design and Studio UI adoption

- Unsloth Studio is UX reference only: the Configure-grid anatomy is adopted; no AGPL code is copied, and Studio's styling is re-expressed in `frontend/DESIGN.md` tokens (flat hairline-bordered section cards, no resting shadows).
- No new tokens, no shell change. The accordion reuses the config-tab accordion pattern from the dataset studio; group headers are uppercase micro-labels; warnings use the `warning` `-tint`/`-strong` pair.

## Edge Cases

- Job created from an older frontend without advanced keys: runners use their defaults; nothing to migrate.
- Unknown or stale key in `hyperparameters`: logged and ignored by the runner allowlist.
- Empty multiselect: treated as "runner default", not "none", and the help text says so.
- Numeric bounds: the form clamps to `min`/`max`; hand-crafted API requests beyond bounds get the soft-warning treatment in the runner log.
- Reruns after a catalog default changes: the job record carries the values it ran with, so history remains truthful.
- Seed set: runners that support determinism pass it everywhere they can; the help text states YOLO/TF determinism is best-effort on GPU.

## Acceptance Criteria

- Model options for all four existing families serialize non-empty `advanced_parameters`, and the training form renders them grouped without family-specific frontend code.
- The training form renders the Model header above a config rail (Dataset + Run) beside the wider Parameters panel at desktop widths, and stacks to one column on narrow viewports.
- Submitting a job with modified advanced values produces runner behavior changes observable in logs/metrics (e.g., YOLO `patience`, HF `warmup_ratio`).
- Unknown keys in `hyperparameters` do not fail a run and appear in the log tail as ignored.
- `bf16` requested on a non-CUDA device downgrades with a log note instead of crashing.
- Job detail shows the submitted advanced values for a completed run.
- Backend tests cover: catalog serialization, per-runner allowlist mapping (command/kwarg generation), ignore-unknown behavior, precision downgrade logic.

## Deferred

- Server-side hard validation profiles.
- Saved parameter presets per project.
- LLM parameter set (phase 14).

## Validation

Commands to run:

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```
