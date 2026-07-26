from fastapi import APIRouter

from app.api.routers import (
    datasets,
    health,
    inference,
    models,
    projects,
    recipes,
    serving,
    settings,
    testing,
    training,
)

router = APIRouter()

router.include_router(health.router)
router.include_router(models.router)
router.include_router(projects.router)
router.include_router(datasets.router)
router.include_router(recipes.router)
router.include_router(inference.router)
router.include_router(serving.router)
router.include_router(testing.router)
router.include_router(training.router)
router.include_router(settings.router)
