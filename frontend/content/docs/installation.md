# Installation & Start

Follow these instructions to set up the Onestep AI Platform locally. The repository contains both the Python backend and the Next.js frontend.

## Prerequisites

- Python 3.11
- `uv` (Fast Python package installer and resolver)
- Node.js (>= 18.x)
- `pnpm` (Fast, disk space efficient package manager)

## Starting the Backend

The backend is built with FastAPI and runs on Python 3.11.

```bash
cd backend
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- The API will be available at `http://localhost:8000`.
- Swagger documentation can be viewed at `http://localhost:8000/docs`.

To run tests:
```bash
cd backend && uv run pytest
```

## Starting the Frontend

The frontend is built with Next.js and React.

```bash
cd frontend
pnpm install
pnpm dev
```

- The web interface will be accessible at `http://localhost:3000`.

To validate the frontend codebase (linting and type checking):
```bash
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```

## Optional: API Keys

Some features read keys you set under **Settings** in the app:

- **Hugging Face token** — required to download gated base models for training and LLM fine-tuning. Also settable via `HUGGINGFACE_HUB_TOKEN` (alias `HF_TOKEN`) in the backend `.env`.
- **OpenRouter key** — enables LLM-assisted record generation in Data Recipes. Without it, recipes fall back to deterministic rule-based records.

Both keys are stored server-side and never returned to the browser.

## Documentation

This guide ships inside the app. Start the frontend and open [`/documentation`](/documentation) — no separate documentation server is needed.
