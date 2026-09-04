# AI Agent Instructions

This repository supports both Codex and Claude Code. Treat this file as the project entrypoint for agent work.

## Start Here

If local skills are available, invoke `$orinth` for repository work.

1. Read `README.md`.
2. Read `docs/ai/workflow.md`.
3. Read `docs/ai/rules.md`.
4. Read the relevant spec in `specs/`.
5. For UI work, read `frontend/DESIGN.md`, then use the `/frontend-design` skill. For auditing or
   reworking an existing surface, `hallmark audit` / `hallmark redesign` are also available.
6. Inspect source before changing it.

## Project Shape

- Backend: `backend/` using FastAPI, `uv`, Python 3.11, SQLite, SQLAlchemy, TensorFlow/Keras, and Ultralytics.
- CLI: `backend/app/cli/` — the `orinth` command. An **HTTP client of the backend**; it never
  imports `app.services`, `app.container`, or `app.core.database`. See `specs/phase-23-cli.md`.
- Notebook SDK: `backend/orinth/` — the package a kernel imports. One module per thing the platform
  does (`datasets`, `models`, `train`, `evaluate`, `inference`, `runs`, `projects`, `settings`).
  May import `app`; **`app` must never import it**. See `specs/phase-22-notebooks.md`.
- Frontend: `frontend/` using Next.js, TypeScript, Tailwind, and pnpm.
- Desktop: `desktop/` using Tauri v2 (Rust), packaged as a macOS `.dmg`.
- Runtime artifacts: `storage/`, ignored.
- Reference assets: `datasets/`, `models/`, `notebooks/`, ignored.

## Required Commands

- Backend tests: `cd backend && uv run pytest`
- Backend lint: `cd backend && uv run ruff check .`
- Frontend typecheck: `cd frontend && pnpm typecheck`
- Frontend lint: `cd frontend && pnpm lint`
- Frontend tests: `cd frontend && pnpm test`
- Frontend build: `cd frontend && pnpm build`
- Everything, in order: `make check`

After changing `backend/app/schemas.py`, regenerate the frontend types with
`cd frontend && pnpm generate:api` and commit `frontend/types/generated/api.ts`.

## Hard Rules

- Do not move, rewrite, or commit files under `datasets/`, `models/`, or `notebooks/` (the
  repo-root reference directories — not `storage/notebooks/`, which is the notebooks feature).
- Do not embed Roboflow keys, API tokens, PHI, or secrets in source.
- Keep model loading lazy. Import TensorFlow and Ultralytics inside predictor code paths.
- Keep generated uploads, overlays, DB files, logs, and training artifacts under ignored `storage/`.
- Use `uv` for backend dependency work and `pnpm` for frontend dependency work.
- **The `orinth` CLI stays a client.** `backend/app/cli/` imports `argparse`, `json`, `tomllib`,
  and `httpx` — nothing from `app.services` or `app.container`. `commands/serve.py` is the one
  exception and imports uvicorn *inside its handler*. `tests/test_cli_dispatch.py` enforces this.
- **The dependency arrow between `orinth` and `app` points one way.** The SDK may import the
  backend; the backend importing the SDK would drag the notebook stack into the API process.
  `tests/test_orinth_sdk.py` enforces this.
- **Never import `jupyter_server` in the API process.** The notebook runtime spawns it as a
  subprocess and probes with `importlib.util.find_spec`, which does not execute it.
- Update the relevant `specs/` file when behavior, APIs, data flow, or acceptance criteria change.
- `frontend/DESIGN.md` is the authoritative design contract. Never hardcode a color; never add elevation
  to a resting surface. Update the contract when the system changes.
- Use `hallmark` only via its `audit` / `redesign` verbs or on component-scope briefs. Its default
  greenfield flow builds marketing pages and conflicts with "every screen is the working app".
- `frontend/app/(auth)` holds the single public route: `/`, the sign-in screen. There is no marketing
  landing page. It is bound by `frontend/DESIGN.md` §10: one typeface, weight cap 600, token-only
  color, border-first elevation, and no invented metrics or testimonials.
- The product is **Orinth**, by **Onestep AI Labs**. Name and mark come from
  `frontend/components/brand.tsx`.

## Medical Context

This app is a research/engineering platform. Do not add UI or docs that imply final clinical diagnosis or autonomous medical decision-making.
