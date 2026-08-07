# Project Map

## Backend

- App entry: `backend/app/main.py`
- Shared app container: `backend/app/container.py`
- Router aggregator: `backend/app/api/routes.py`
- Domain routers: `backend/app/api/routers/`
- Route serializers: `backend/app/api/serializers.py`
- Settings: `backend/app/core/config.py`
- DB models: `backend/app/db/models.py`
- Schemas: `backend/app/schemas.py`
- Model registry: `backend/app/ml/model_registry.py`
- Predictors: `backend/app/ml/predictors/`
- Dataset service package: `backend/app/services/datasets/`
- Inference service: `backend/app/services/inference.py`
- Evaluation service package: `backend/app/services/evaluation/`
- Metrics: `backend/app/services/metrics.py`
- Training service package: `backend/app/services/training/`
- Training runners: `backend/app/training/runners/`

Compatibility imports are preserved for service packages, such as:

```python
from app.services.datasets import DatasetService
from app.services.evaluation import EvaluationService
from app.services.training import TrainingService
```

Backend commands:

```bash
cd backend && uv run pytest
cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend

- App route entrypoints: `frontend/app/`
- App shell: `frontend/components/app-shell.tsx`
- Page export barrel: `frontend/components/platform-pages.tsx`
- Feature modules: `frontend/features/`
- API client package: `frontend/lib/api/`
- API types: `frontend/types/api.ts`
- Base styling: `frontend/app/globals.css`
- Platform styling: `frontend/app/styles/platform.css`

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
