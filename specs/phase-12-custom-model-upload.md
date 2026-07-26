# Spec: Phase 12 Custom Model Upload

## Status

Implemented. Model endpoints were extracted into `backend/app/api/routers/models.py`
(paths unchanged) and joined by `GET /api/models/upload-options` and
`POST /api/models/upload`. Upload validation lives in
`backend/app/services/model_upload.py` (structural only; deep validation stays
deferred to first predictor construction). Uploaded weights land under
`storage/uploaded_models/<model_id>/` via a temp-dir → atomic-rename flow;
registry entries gain `source: "uploaded"`, `base_model_id`, and `format`, and
the `sklearn_pipeline` predictor is `backend/app/ml/sklearn/predictor.py`. LLM
families (`llm_hf`/`llm_adapter`/`llm_gguf`) register but raise a phase-15 gate on
predictor construction. Frontend: `features/models/models-page.tsx` (source
sections + Upload action), `features/models/upload-model-dialog.tsx` (dynamic
form), `features/models/model-detail-page.tsx` + route
`app/(platform)/models/[modelId]`.

Adapter `base_model_id` currently validates against registered `llm_hf` models
only; the phase-14 catalog is consulted lazily via `app.ml.llm.catalog` once it
exists.

## Goal

Let users upload their own model weights for every family the platform knows — deep learning (YOLO `.pt`, Keras `.h5`/`.keras`, U-Net + Inception pairs), classic ML (sklearn pickle/joblib pipelines), and LLM artifacts (HF safetensors directory/zip, LoRA adapter, GGUF) — validated structurally, registered in the model registry, and usable wherever their family is usable. Adds the minimal model detail page that phase 15 extends with serving and export.

## Scope

In:

- A dedicated models router (extracting model endpoints out of `health.py`).
- Upload options descriptor + multipart upload endpoint with per-family structural validation.
- Registry extensions: `source: "uploaded"`, families `llm_hf` / `llm_adapter` / `llm_gguf`, `base_model_id`.
- Models page sections, upload flow, and a minimal model detail route.
- Directory-backed model download as zip.

Out:

- Serving or running uploaded LLM artifacts (phase 15; cards show the gate).
- Fine-tuning from uploaded bases (phase 14 consumes `llm_hf` entries).
- DB schema changes — the registry stays `storage/model_registry.json`; no Alembic migration in this phase.

## Interfaces

- New router `backend/app/api/routers/models.py` absorbing the existing model endpoints from `health.py` (aggregator `routes.py` unchanged in shape):
  - `GET /api/models`
  - `PATCH /api/models/{id}`
  - `DELETE /api/models/{id}`
  - `GET /api/models/{id}/download`
- `GET /api/models/upload-options`
- `POST /api/models/upload` (multipart)
- Schemas: `ModelUploadOption`, `ModelUploadResult` in `backend/app/schemas.py`; `ModelInfo` gains `base_model_id` and `format` passthroughs.
- Frontend surfaces: `frontend/features/models/models-page.tsx` (Upload action, Reference/Trained/Uploaded sections, source badges); new route `frontend/app/(platform)/models/[modelId]/page.tsx` with feature `frontend/features/models/model-detail-page.tsx`; upload dialog `frontend/features/models/upload-model-dialog.tsx`; `frontend/lib/api/models.ts` additions.
- Storage/DB changes: uploaded weights under `storage/uploaded_models/<model_id>/`; registry entries in `storage/model_registry.json` gain `source: "uploaded"`, families `llm_hf`, `llm_adapter`, `llm_gguf`, and optional `base_model_id`. No DB change.

## Behavior

### Router extraction

- Model endpoints move from `backend/app/api/routers/health.py` into a new `backend/app/api/routers/models.py`. Paths do not change, so no frontend or test URL updates are needed; the extraction exists because this phase triples the router's surface and `health.py` was never its home.

### Upload options and form

- `GET /api/models/upload-options` returns a per-family descriptor: accepted extensions, whether a file pair or zip is expected, and which extra fields are required (labels, input size, task type, `base_model_id`). The frontend renders the upload form dynamically from this descriptor, so adding a family later is a backend-only change.
- Supported families and payloads:
  - `yolo` — single `.pt`; requires task type (detection/segmentation) and labels.
  - `keras_classification` — single `.h5`/`.keras`; requires labels and input size.
  - `unet_inception` — file pair (U-Net `.h5` + Inception `.h5`); both required, one missing rejects the upload.
  - `sklearn_pipeline` — single `.pkl`/`.joblib`; requires a declared text task and labels where the task needs them.
  - `llm_hf` — zip of an HF model directory (`config.json` + `*.safetensors` + tokenizer files).
  - `llm_adapter` — zip containing `adapter_config.json` + adapter weights; requires `base_model_id` referencing a registered `llm_hf` model or a phase-14 catalog base id.
  - `llm_gguf` — single `.gguf`.

### Validation

- Validation is two-tier, preserving the platform's lazy-loading rule. Upload time performs synchronous structural validation only: extension check plus magic bytes (`.pt`/`.keras` are zip archives, `GGUF` magic for `.gguf`, member listing for HF/adapter zips, pickle opcode sniff for sklearn files). Deep validation — actually constructing the predictor — stays deferred to first use, exactly like every other registry family. The upload response reports `validated: "structural"` so the UI can say what was and wasn't checked.
- Security: unpickling sklearn files executes arbitrary code by design. This is acceptable only under the platform's existing single-user local-trust posture and is stated in the descriptor help text; `docs/ai/rules.md` already requires auth before any network-exposed deployment, and this feature does not change that line.
- Zip handling: entries are streamed to disk (no full-file memory buffering), zip-slip paths (`..`, absolute) are rejected, and the expanded size is capped. The default upload cap is 10 GB (settings-configurable) to admit multi-GB LLM directories while still bounding disk use.
- All writes go to a temp directory first with an atomic move into `storage/uploaded_models/<model_id>/` on success; failures clean up so no orphan directories accumulate.

### Registration and use

- Successful uploads register a `ModelSpec` with `source: "uploaded"`, the declared family/task/labels, and paths into `storage/uploaded_models/`. Uploaded vision/NLP/sklearn models become available to inference and testing exactly as trained models do — predictor construction is the existing lazy dispatch by family.
- `sklearn_pipeline` uploads are runnable only if the pipeline satisfies the existing baseline predictor contract (`predict`/`predict_proba` on raw text); a mismatch surfaces at first use as a surfaced predictor error, consistent with other lazy families.
- `llm_hf`, `llm_adapter`, and `llm_gguf` entries register now but are not yet runnable; their cards and detail pages show a "Servable after export/serving lands (phase 15)" gate, and `llm_hf` entries additionally become selectable fine-tune bases in phase 14.
- `GET /api/models/{id}/download` is extended to zip directory-backed models on the fly (HF dirs, adapters), keeping single-file behavior for single-file models.
- Uploaded models support the existing rename/delete flows; delete removes the `storage/uploaded_models/<model_id>/` directory.

### Models page and detail route

- The models page groups cards into Reference / Trained / Uploaded sections (rendered only when non-empty) with a source badge per card, and gains a primary "Upload model" action — fulfilling the promise its empty state already makes.
- A minimal detail route `frontend/app/(platform)/models/[modelId]` shows metadata (family, task, labels, source, base model, size on disk), artifact listing, and the rename/download/delete actions. Phase 15 extends this page with export and serving panels; it exists now so those panels have a home.

### Design and Studio UI adoption

- The Uploaded/Trained listings adopt the anatomy of Unsloth Studio's Hub "Downloaded" view (`features/hub/hub-page.tsx`): dense rows showing name, family badge, task, size on disk, and a per-row actions menu — a list, not a marketing card grid, because these are operational inventories. Studio is UX reference only; no AGPL code is copied and its styling is re-expressed in `frontend/DESIGN.md` tokens.
- The model detail page adopts Studio's model-card structure (metadata block, artifact/variant list, action row) translated to the platform's panel vocabulary; phase 15 slots its export and serving panels beneath.
- No new tokens, no shell change. The upload dialog uses `--shadow-overlay`; source badges use status `-tint`/`-strong` pairs; the detail page header follows the existing eyebrow + 24px title pattern; sections are hairline-separated.

## Edge Cases

- Family/file mismatch (e.g., `.gguf` uploaded as `yolo`): rejected by structural validation with the expected-extension message.
- Adapter upload without `base_model_id` or with a dangling reference: rejected with a field message.
- Duplicate display name: allowed (ids are canonical), but the response notes the collision so users can rename.
- Interrupted multipart upload: temp dir discarded, nothing registered.
- Oversized file or zip-bomb expansion beyond the cap: rejected with the cap stated; partial writes cleaned up.
- U-Net pair with one file missing: rejected naming the missing member.
- Upload while the same display name's model is loaded in inference: unaffected — new uploads always mint new ids; predictor caches key on id.

## Acceptance Criteria

- Model endpoints respond at unchanged paths after the router extraction; existing tests pass unmoved.
- Users can upload one model of each supported family through the dynamic form and see it registered under Uploaded with a source badge.
- Structural validation rejects each documented mismatch case with an actionable message.
- An uploaded YOLO/Keras/sklearn model runs in inference/testing without server restart.
- `llm_*` uploads show the phase-15 gate and cannot be selected for inference.
- Directory-backed models download as a zip; single-file models download unchanged.
- Backend tests cover: descriptor contents, per-family structural validation (fixtures with crafted magic bytes), zip-slip rejection, size-cap rejection, atomic cleanup on failure, registry round-trip, download-as-zip.

## Deferred

- Deep validation at upload time (opt-in "verify now" action).
- ONNX and TensorFlow SavedModel families.
- Multi-file sharded safetensors upload without zipping.

## Validation

Commands to run:

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```
