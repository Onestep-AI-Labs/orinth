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

## 3. Design

Applies to any change that adds or alters a UI surface. Skip only for pure backend, data, or docs work.

- Read `frontend/DESIGN.md` first. It is the authoritative contract; everything below is how to apply it.
- Invoke the `/frontend-design` skill for all UI and styling work.
- Invoke `hallmark` only with an explicit verb — `hallmark audit <target>` to score an existing surface, or
  `hallmark redesign <target>` to rework one. Component-scope briefs ("just the button") are also fine.
- **Do not run hallmark's default greenfield flow on a `(platform)` route.** It composes marketing
  pages — heroes, marquees, feature pills, testimonials, footers — which `docs/ai/rules.md` forbids for
  the working app. Its `audit` and `redesign` verbs are the parts that fit those surfaces.
- There is no greenfield surface left in this app. `frontend/app/(auth)` holds the single public route
  (`/`, sign-in) and is bound by `frontend/DESIGN.md` §10 — treat it with `audit` / `redesign`, same as
  a platform route. See `specs/phase-19-orinth-rebrand.md` for why the landing page was removed.
- Design decisions that change tokens, elevation, typography, or shell anatomy belong in a spec before
  implementation, and in `frontend/DESIGN.md` after.
- Reuse tokens and shared primitives. Add a token rather than a literal; add a variant rather than a
  one-off class.

## 4. Implement

- Backend changes go under `backend/app`.
- Route changes go in the matching domain router under `backend/app/api/routers/`; keep `backend/app/api/routes.py` as the router aggregator.
- **Declaration order matters in a router.** FastAPI matches in order, so literal segments
  (`/hub/search`, `/templates`, `/runtime/*`, `/proxy/*`) must be declared *before* a sibling path
  parameter (`/{dataset_id}`, `/{notebook_id}`) or the parameter swallows them.
- CLI changes go under `backend/app/cli/`. Adding a command group means one row in `main.GROUPS`
  (a module *path string*, never an import), one module with `build_parser()` and `run()`, and one
  row in `commands/completion.VERBS`.
- Notebook SDK changes go under `backend/orinth/`. Its public surface is documented by the shipped
  templates in `backend/app/services/notebooks/templates/` — if a new call does not fit in one of
  them, that is a signal the surface is growing past what a user can discover.
  `test_the_sdk_modules_are_each_demonstrated_by_a_template` enforces it: a new public call needs a
  template that opens with it, or the test fails.
- Shared backend service singletons live in `backend/app/container.py`.
- Large domain services may be packages under `backend/app/services/`; preserve compatibility exports from each package `__init__.py`.
- Frontend changes go under `frontend`.
- Frontend route entrypoints stay under `frontend/app`; reusable page and workflow code should live under `frontend/features`.
- Frontend API calls belong in the domain-based `frontend/lib/api/` package while preserving the public `api` export.
- Keep app-wide base styles in `frontend/app/globals.css` and platform UI selectors in `frontend/app/styles/platform.css`.
- After changing `backend/app/schemas.py`, run `cd frontend && pnpm generate:api` and commit the
  regenerated `frontend/types/generated/api.ts`. The generated file is the single source of truth;
  `frontend/types/api.ts` is a thin facade over it.
- Generated runtime output goes under `storage`.
- Use lazy imports for heavy ML libraries.
- Keep inference responses normalized across YOLO, U-Net + Inception, and NLP Hugging Face Transformers.
- Do not add secrets or notebook API keys to source.

## 5. Validate

Run the checks that match the touched area:

```bash
cd backend && uv run ruff check .
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm test
cd frontend && pnpm build
```

Or `make check`, which runs them in that order and stops at the first failure.

For inference changes, also run at least one model smoke test when local model files are present.

**Run the thing you changed against the real workspace before reporting it done.** The unit suite
does not start a server, a kernel, or a subprocess, and every defect found late in phases 21–23 was
found this way and not by a test:

- API or CLI: `cd backend && uv run uvicorn app.main:app --port 8123` in the background, then drive
  it — `ORINTH_BACKEND=http://127.0.0.1:8123 uv run orinth doctor`.
- Notebooks: `POST /api/notebooks/runtime/start`, then create a notebook and read it back through
  `/api/notebooks/proxy/api/contents/<path>`.
- SDK: `uv run python -c "import orinth; print(orinth.datasets.list())"`.

There are real datasets in `storage/datasets/`. Use them, and clean up anything you create.

## 6. Report

Summarize:

- files changed,
- behavior changed,
- validation commands and results,
- known limitations or deferred work.

Do not hide failures. State them directly and give the next concrete fix.
