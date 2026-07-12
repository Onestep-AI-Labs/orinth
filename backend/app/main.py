from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.container import (
    evaluation_service,
    inference_service,
    job_executor,
    storage,
    training_service,
)
from app.core.config import get_settings
from app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    storage.ensure()
    init_db()
    training_service.reconcile_stale_jobs()
    evaluation_service.reconcile_stale_jobs()
    inference_service.reconcile_stale_jobs()
    yield
    job_executor.shutdown(wait=True)


settings = get_settings()
settings.storage_path.mkdir(parents=True, exist_ok=True)
app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=settings.storage_path), name="media")
app.include_router(router, prefix=settings.api_prefix)


@app.get("/health")
def root_health() -> dict[str, str]:
    return {"status": "ok"}
