# Project Map

## Backend

- App entry: `backend/app/main.py`
- Routes: `backend/app/api/routes.py`
- Settings: `backend/app/core/config.py`
- DB models: `backend/app/db/models.py`
- Schemas: `backend/app/schemas.py`
- Model registry: `backend/app/ml/model_registry.py`
- Predictors: `backend/app/ml/predictors/`
- Inference service: `backend/app/services/inference.py`
- Evaluation service: `backend/app/services/evaluation.py`
- Metrics: `backend/app/services/metrics.py`
- Training service: `backend/app/services/training.py`
- Training runners: `backend/app/training/runners/`

Backend commands:

```bash
cd backend && uv run pytest
cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend

- App screen: `frontend/app/page.tsx`
- API client: `frontend/lib/api.ts`
- API types: `frontend/types/api.ts`
- Styling: `frontend/app/globals.css`

Frontend commands:

```bash
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
cd frontend && pnpm dev
```

## Specs

- Inference: `specs/phase-1-inference.md`
- Testing: `specs/phase-2-testing.md`
- Training: `specs/phase-3-training.md`
- New spec template: `specs/SPEC_TEMPLATE.md`

Update specs when implementation changes behavior.

## Local Assets

- YOLO dataset: `datasets/dental dataset_yolov11_format`
- COCO dataset: `datasets/dental dataset_coco_format`
- YOLO weights: `models/yolo_11_best/weights/best.pt`
- U-Net model: `models/unet_inception/best_unet_model.keras`
- Inception classifier: `models/unet_inception/best_classifier_inception.keras`

Do not move or commit these assets.

## API Contracts

Inference:

- `GET /api/models`
- `POST /api/inference`
- `GET /api/inference`
- `GET /api/inference/{id}`

Testing:

- `GET /api/testing/datasets`
- `POST /api/testing/jobs`
- `GET /api/testing/jobs`
- `GET /api/testing/jobs/{id}`

Training:

- `POST /api/training/jobs`
- `GET /api/training/jobs`
- `GET /api/training/jobs/{id}`
- `POST /api/training/jobs/{id}/promote`
