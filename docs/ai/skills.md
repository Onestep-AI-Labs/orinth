# Project AI Skills

The project-local skill lives at:

- `.agents/skills/orinth`

Use `$orinth` for repository work when the agent runtime discovers local skills.

The sections below document the workflows captured by that skill.

## Repository Orientation

Use when starting any task.

- Read `README.md`, `AGENTS.md`, `docs/ai/workflow.md`, and relevant specs.
- Check `git status --short`.
- Identify touched subsystem before editing.

## Backend API

Use when changing FastAPI routes, schemas, DB models, or services.

- Read `backend/app/api/routes.py`, the relevant domain router under `backend/app/api/routers/`, `backend/app/schemas.py`, and relevant service packages.
- Preserve response compatibility unless the spec says otherwise.
- Validate with `cd backend && uv run pytest`.

## ML Inference

Use when changing YOLO or U-Net + Inception behavior.

- Read `backend/app/ml/model_registry.py`.
- Read predictor files under `backend/app/ml/predictors`.
- Keep outputs normalized to `InferenceResult`.
- Smoke test with a local image when model files are present.

## Evaluation and Metrics

Use when changing phase 2 testing.

- Read `backend/app/services/evaluation/`.
- Read `backend/app/services/metrics.py`.
- Keep notebook-derived metric semantics unless a spec changes them.

## Dataset Studio

Use when changing dataset browsing, imports, uploads, or annotations.

- Read `backend/app/services/datasets/`.
- Keep reference datasets under `datasets/` read-only.
- Store editable datasets and uploaded images under ignored `storage/datasets`.
- Update `specs/phase-4-dataset-studio.md` when dataset APIs or annotation behavior change.

## Training Jobs

Use when changing phase 3 training.

- Read `backend/app/services/training/`.
- Read runners under `backend/app/training/runners`.
- Keep long training in subprocesses.
- Store artifacts under ignored `storage/training_runs`.

## Frontend Workflow UI

Use when changing app screens or controls.

- Read `frontend/app/page.tsx`.
- Read `frontend/components/platform-pages.tsx`, relevant files under `frontend/features/`, `frontend/lib/api/`, and `frontend/types/api.ts`.
- Validate with `pnpm typecheck`, `pnpm lint`, and `pnpm build`.

## Documentation and Specs

Use when updating process, behavior, or implementation plans.

- Update `specs/` before or alongside behavior changes.
- Keep docs concise and operational.
- Cross-link Codex and Claude entrypoints when workflow changes.
