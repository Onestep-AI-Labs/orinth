---
name: onestep-ai-platform
description: Onestep AI Platform workflow for Codex or Claude Code. Use when working on this repository's FastAPI backend, Next.js frontend, image datasets, YOLO inference, U-Net + Inception inference, testing/evaluation jobs, training jobs, SQLite persistence, project specs, or AI workflow documentation.
---

# Onestep AI Platform

## Core Workflow

1. Read `AGENTS.md`, `docs/ai/workflow.md`, and `docs/ai/rules.md`.
2. Read the relevant spec in `specs/` before changing behavior.
3. Inspect current code before editing.
4. Keep `datasets/`, `models/`, `notebooks/`, and `storage/` as local reference/runtime assets.
5. Update the relevant spec when APIs, schemas, data flow, model behavior, or acceptance criteria change.
6. Run validations for the touched subsystem.

## Project Map

Read `references/project-map.md` when you need subsystem-specific paths, commands, and invariants.

Key defaults:

- Backend uses FastAPI, `uv`, Python 3.11, SQLite, SQLAlchemy, TensorFlow/Keras, and Ultralytics.
- Frontend uses Next.js, TypeScript, Tailwind, and pnpm.
- YOLO model id is `yolo_11_best`.
- U-Net + Inception model id is `unet_inception`.
- Class labels are `granuloma`, `kista`, and image-level `Normal`.

## Implementation Rules

- Use `uv` for backend commands.
- Use `pnpm` for frontend commands.
- Keep heavy ML imports lazy inside predictor paths.
- Keep inference output normalized through `InferenceResult`.
- Do not commit secrets, Roboflow keys, PHI, model weights, datasets, uploads, overlays, or DB files.
- Store generated runtime artifacts under ignored `storage/`.

## Validation

Run the smallest useful set:

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
```

For inference changes, run a local smoke test with at least one available test image and model.
