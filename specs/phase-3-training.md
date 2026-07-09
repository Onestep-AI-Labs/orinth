# Spec: Phase 3 Training

## Status

Partial MVP

## Goal

Provide a local training job platform that runs long training work outside the FastAPI request path.

## Interfaces

- `POST /api/training/jobs`
- `GET /api/training/jobs`
- `GET /api/training/jobs/{id}`
- `POST /api/training/jobs/{id}/promote`

## Behavior

- Training jobs run in subprocesses.
- Logs and artifacts are written under ignored `storage/training_runs`.
- YOLO training uses local dataset paths and notebook-derived hyperparameters.
- Completed YOLO runs can be promoted into the model registry.
- U-Net + Inception training is gated until the long two-stage TensorFlow path is validated.

## Acceptance Criteria

- Training dashboard can create and list jobs.
- YOLO runner writes logs and artifacts.
- Failed jobs persist a clear error.
- Promotion registers a completed YOLO `best.pt` as an inference model.

## Deferred

- Full U-Net + Inception two-stage training runner.
- Training cancellation.
- Live log streaming.
- Hardware/device selection.
