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

## Running the Documentation Site

The documentation is built with Nextra.

```bash
cd docs-site
npm install
npm run dev
```

- The docsite will be accessible at `http://localhost:3000` (make sure the platform frontend is running on a different port or run this on a different port if running simultaneously, e.g. `npm run dev -- -p 3001`).
