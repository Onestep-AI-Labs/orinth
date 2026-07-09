# Backend

FastAPI backend for model registry, inference, evaluation jobs, and training jobs.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The ML dependencies are loaded lazily when a model is used. API docs are available at `/docs`.
