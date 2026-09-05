from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.container import (
    dataset_prep_service,
    evaluation_executor,
    evaluation_service,
    export_executor,
    export_service,
    inference_executor,
    inference_service,
    notebook_runtime,
    recipe_executor,
    recipe_service,
    serving_service,
    storage,
    training_executor,
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
    recipe_service.reconcile_stale_recipes()
    # A prep state is a string on a manifest with nothing running behind it, so
    # an interrupted run stays `planning` forever — and the studio polls the
    # catalog every four seconds for as long as it does.
    dataset_prep_service.reconcile_stale_preps()
    # Directories a previous process renamed aside for deletion but was killed
    # before it finished unlinking. Nothing reads them; they are only disk.
    storage.purge_trash()
    export_service.reconcile_stale_exports()
    # A crashed API process cannot leak a llama.cpp server: reap whatever the
    # serving state file still records before serving requests.
    serving_service.reap_orphans()
    yield
    serving_service.shutdown()
    # Phase 22: kills the jupyter-server subprocess and every kernel under it,
    # so a restart does not leave a stranded server holding its port.
    notebook_runtime.shutdown()
    training_executor.shutdown(wait=True)
    evaluation_executor.shutdown(wait=True)
    inference_executor.shutdown(wait=True)
    recipe_executor.shutdown(wait=True)
    export_executor.shutdown(wait=True)


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
    """Liveness, plus enough to tell two builds apart.

    `orinth version` and `orinth doctor` exist to answer "is the CLI I am running
    the same build as the server it is talking to", which they cannot do without
    the version here. Additive: phase 18's desktop supervisor polls this for a
    200 and ignores the body.
    """
    from app.cli import __version__

    return {"status": "ok", "app": "orinth", "version": __version__}
