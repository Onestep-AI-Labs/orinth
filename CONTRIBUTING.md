# Contributing

Thanks for helping improve Onestep AI Platform. This project is intended to be open-source friendly while keeping local research assets, generated artifacts, and secrets out of source control.

## Model Adoption

Models are organized by task first, then by model family:

- Vision models live under `backend/app/ml/vision/{model_family}/`.
- NLP models live under `backend/app/ml/nlp/{model_family}/`.
- Shared predictor primitives live under `backend/app/ml/common/`.
- Long-running training code lives under `backend/app/training/runners/`.

When adding a model, add a package that declares its own training option catalog. A model option must state:

- `id`: stable option id, for example `hf_bart_summarizer`.
- `family`: stable registered model family, for example `hf_bart_summarization`.
- `task_types`: the exact supported task types.
- `source`: `local`, `keras_applications`, `ultralytics`, or `huggingface`.
- `runnable`: whether training can run today.
- `needs_download`: whether explicit asset preparation is required.
- `defaults`: safe training defaults such as epochs, batch size, learning rate, max length, model id, or image size.

Wire the package catalog through `backend/app/ml/training_catalog.py`. Do not hardcode new model lists directly in API routers.

## Required Model Pieces

Each runnable model family needs:

- A catalog entry under its model package.
- A subprocess training path that writes artifacts under `storage/training_runs`.
- A predictor that returns the existing normalized inference shapes.
- Registry support in `backend/app/ml/model_registry.py`.
- Promotion support in `backend/app/services/training/service.py`.
- Tests for option discovery, command generation, artifact discovery, and prediction.
- Spec updates when behavior, APIs, data flow, or acceptance criteria change.

## Custom Model Wiring Steps

1. Choose the task type and family id.
   Use the existing task ids: `classification`, `object_detection`, `segmentation`, `text_classification`, `summarization`, or `question_answering`. Pick a stable family id such as `my_text_classifier`; this id is used by the registry, promotion, and frontend filtering.

2. Add a model package catalog.
   Put the package under `backend/app/ml/vision/{family}/` or `backend/app/ml/nlp/{family}/`. Define a `TrainingModelDefinition` with safe defaults, the supported task types, `source`, `runnable`, and `needs_download`.

3. Register the catalog.
   Export the package catalog through `backend/app/ml/training_catalog.py`. Keep routers generic; do not add model-specific lists directly to API routes.

4. Implement or extend a runner.
   Long training work must run through `backend/app/training/runners/`. The runner should read dataset paths and hyperparameters from CLI args and write artifacts under the provided run directory.

5. Write expected artifacts.
   Always write `results.csv`, `metrics.json`, and validation predictions when the task can produce them. Save the best model using the existing conventions listed below so promotion can discover it.

6. Add predictor support.
   Implement a predictor that returns normalized outputs: image models use detections/class scores through `InferenceResult`; NLP models return `predict_text()` dictionaries with `label`, `scores`, `summary`, or `answer`.

7. Wire the registry.
   Add the family to `backend/app/ml/model_registry.py` so trained/promoted artifacts can instantiate the correct predictor lazily.

8. Wire training service promotion.
   Add the family to the runnable/promotion allowlists in `backend/app/services/training/service.py`, define command generation, and copy any sidecar files needed for inference.

9. Keep the frontend task-first.
   Model options must filter by `task_type`; task-specific form controls should send extra values through `hyperparameters` rather than adding unrelated top-level fields.

10. Validate and update specs.
    Add focused backend tests for option discovery, command generation, artifact discovery, and predictor smoke behavior. Update the relevant `specs/` file before considering the model integrated.

Artifact expectations:

- Keras image models save `best_model.keras`.
- Keras NLP classifiers save `best_model.keras`, `last_model.keras`, `tokenizer.json`, `metadata.json`, `metrics.json`, `validation_predictions.json`, and `results.csv`.
- Keras seq2seq summarizers use the same Keras NLP sidecars.
- Hugging Face models save a model directory containing `config.json`, tokenizer files, model weights, `metadata.json`, plus run-level `metrics.json`, `validation_predictions.json`, and `results.csv`.
- Baselines may save `model.pkl` or `model.json` when that is the intended lightweight artifact.

## Security and Assets

Never commit:

- Roboflow keys, Hugging Face tokens, API tokens, or credentials.
- PHI or patient-identifying data.
- Runtime uploads, overlays, databases, logs, training runs, model weights, datasets, or notebooks.

Use ignored `.env` files for local secrets. `HUGGINGFACE_HUB_TOKEN` may be configured for private/gated Hugging Face models, but code must never print the token or commit it to source control.

## Validation

Run the smallest useful checks for your change:

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```

Backend-only model changes usually require `cd backend && uv run pytest`. Frontend checks are required when UI, TypeScript types, or API client behavior changes.
