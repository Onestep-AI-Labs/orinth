# AI Workflow

Use this workflow for Codex, Claude Code, or any future coding agent.

## 1. Orient

- Read `AGENTS.md` and `CLAUDE.md`.
- Read the relevant spec under `specs/`.
- Inspect current source before proposing or editing.
- Confirm whether the task touches backend, frontend, ML inference, testing, training, or documentation.

## 2. Plan

- If the request changes behavior or APIs, update or create a spec first.
- Keep specs decision-complete: goal, scope, interfaces, data flow, edge cases, and acceptance checks.
- Prefer existing app patterns over new abstractions.
- Keep current datasets, model weights, and notebooks as read-only references.

## 3. Implement

- Backend changes go under `backend/app`.
- Route changes go in the matching domain router under `backend/app/api/routers/`; keep `backend/app/api/routes.py` as the router aggregator.
- Shared backend service singletons live in `backend/app/container.py`.
- Large domain services may be packages under `backend/app/services/`; preserve compatibility exports from each package `__init__.py`.
- Frontend changes go under `frontend`.
- Frontend route entrypoints stay under `frontend/app`; reusable page and workflow code should live under `frontend/features`.
- Frontend API calls belong in the domain-based `frontend/lib/api/` package while preserving the public `api` export.
- Keep app-wide base styles in `frontend/app/globals.css` and platform UI selectors in `frontend/app/styles/platform.css`.
- Generated runtime output goes under `storage`.
- Use lazy imports for heavy ML libraries.
- Keep inference responses normalized across YOLO and U-Net + Inception.
- Do not add secrets or notebook API keys to source.

## 4. Validate

Run the checks that match the touched area:

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```

For inference changes, also run at least one model smoke test when local model files are present.

## 5. Report

Summarize:

- files changed,
- behavior changed,
- validation commands and results,
- known limitations or deferred work.

Do not hide failures. State them directly and give the next concrete fix.
