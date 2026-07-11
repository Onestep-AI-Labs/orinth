# Onestep AI Platform

Production-oriented local image intelligence workspace for dataset preparation, annotation, training, testing, and model inspection.

The repository keeps existing research assets local-only:

- `datasets/`
- `models/`
- `notebooks/`

The app source lives in `backend/` and `frontend/`.

## Architecture

The backend is a FastAPI app with domain routers and service packages:

- App entrypoint: `backend/app/main.py`
- Shared service container: `backend/app/container.py`
- API router aggregator: `backend/app/api/routes.py`
- Domain routers: `backend/app/api/routers/`
- API serializers: `backend/app/api/serializers.py`
- Schemas: `backend/app/schemas.py`
- DB models: `backend/app/db/models.py`
- ML registry and predictors: `backend/app/ml/`
- Services: `backend/app/services/`
- Training runners: `backend/app/training/runners/`

The frontend is a Next.js app with route entrypoints, feature modules, and a domain-based API client:

- App routes: `frontend/app/`
- App shell: `frontend/components/app-shell.tsx`
- Page export barrel: `frontend/components/platform-pages.tsx`
- Feature modules: `frontend/features/`
- API client package: `frontend/lib/api/`
- API types: `frontend/types/api.ts`
- Global base styles: `frontend/app/globals.css`
- Platform styles: `frontend/app/styles/platform.css`

## Backend

```bash
cd backend
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend API docs will be available at `http://localhost:8000/docs`.

Backend validation:

```bash
cd backend && uv run pytest
```

## Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Frontend will be available at `http://localhost:3000`.

Frontend validation:

```bash
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```

## Phases

Phase 1 includes inference for both available local model families:

- YOLOv11 segmentation: `models/yolo_11_best/weights/best.pt`
- U-Net + Inception: `models/unet_inception/best_unet_model.keras` and `best_classifier_inception.keras`

Phase 2 adds dataset testing jobs and metrics dashboards.

Phase 3 adds subprocess-based training jobs and model promotion.

## AI Workflow

Project AI guidance is shared across Codex and Claude Code:

- Codex entrypoint: `AGENTS.md`
- Claude Code entrypoint: `CLAUDE.md`
- Local skill: `.agents/skills/onestep-ai-platform`
- Workflow and rules: `docs/ai/`
- Planning and execution references: `specs/`


## ToDo
- remove upload dialog from browser after select folder (datasets)-> use custom modal dialog
- move class page from frontend/features/platform/internal.tsx to every module and class also check the app, components, features, etc make them all easy to maintain so dont too long in single files
- add bbox tool, after polygon tool(already there) annotations for object detection task
- add download button in three dots options in model list, make the model downloadable.
- start implement simple NLP task pipeline
- planning simple LLM task pipeline
