# AI Agent Instructions

This repository supports both Codex and Claude Code. Treat this file as the project entrypoint for agent work.

## Start Here

If local skills are available, invoke `$onestep-ai-platform` for repository work.

1. Read `README.md`.
2. Read `docs/ai/workflow.md`.
3. Read `docs/ai/rules.md`.
4. Read the relevant spec in `specs/`.
5. For UI work, read `frontend/DESIGN.md`, then use the `/frontend-design` skill. For auditing or
   reworking an existing surface, `hallmark audit` / `hallmark redesign` are also available.
6. Inspect source before changing it.

## Project Shape

- Backend: `backend/` using FastAPI, `uv`, Python 3.11, SQLite, SQLAlchemy, TensorFlow/Keras, and Ultralytics.
- Frontend: `frontend/` using Next.js, TypeScript, Tailwind, and pnpm.
- Runtime artifacts: `storage/`, ignored.
- Reference assets: `datasets/`, `models/`, `notebooks/`, ignored.

## Required Commands

- Backend tests: `cd backend && uv run pytest`
- Frontend typecheck: `cd frontend && pnpm typecheck`
- Frontend lint: `cd frontend && pnpm lint`
- Frontend build: `cd frontend && pnpm build`

## Hard Rules

- Do not move, rewrite, or commit files under `datasets/`, `models/`, or `notebooks/`.
- Do not embed Roboflow keys, API tokens, PHI, or secrets in source.
- Keep model loading lazy. Import TensorFlow and Ultralytics inside predictor code paths.
- Keep generated uploads, overlays, DB files, logs, and training artifacts under ignored `storage/`.
- Use `uv` for backend dependency work and `pnpm` for frontend dependency work.
- Update the relevant `specs/` file when behavior, APIs, data flow, or acceptance criteria change.
- `frontend/DESIGN.md` is the authoritative design contract. Never hardcode a color; never add elevation
  to a resting surface. Update the contract when the system changes.
- Inside `frontend/app/(platform)`, use `hallmark` only via its `audit` / `redesign` verbs or on
  component-scope briefs. Its default greenfield flow builds marketing pages and conflicts with
  "every platform screen is the working app".
- `frontend/app/(marketing)` — the two public routes `/` and `/signin` — is the one place the greenfield
  flow applies. It is still bound by `frontend/DESIGN.md` §10: one typeface, weight cap 600, token-only
  color, border-first elevation, real screenshots, and no invented metrics or testimonials.

## Medical Context

This app is a research/engineering platform. Do not add UI or docs that imply final clinical diagnosis or autonomous medical decision-making.
