# Dental Segmentation Platform

Production-oriented local platform for dental radiograph segmentation and classification of `granuloma` and `kista`.

The repository keeps existing research assets local-only:

- `datasets/`
- `models/`
- `notebooks/`

The app source lives in `backend/` and `frontend/`.

## Backend

```bash
cd backend
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend API docs will be available at `http://localhost:8000/docs`.

## Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Frontend will be available at `http://localhost:3000`.

## Phases

Phase 1 includes inference for both available local model families:

- YOLOv11 segmentation: `models/yolo_11_best/weights/best.pt`
- U-Net + Inception: `models/unet_inception/best_unet_model.keras` and `best_classifier_inception.keras`

Phase 2 adds dataset testing jobs and metrics dashboards.

Phase 3 adds subprocess-based training jobs and model promotion.
