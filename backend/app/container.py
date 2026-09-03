from concurrent.futures import ThreadPoolExecutor

from app.core.config import get_settings
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry
from app.services.architectures import ArchitectureService
from app.services.dataset_hub import DatasetHubService
from app.services.datasets import DatasetService
from app.services.datasets.prep.service import DatasetPrepService
from app.services.evaluation import EvaluationService
from app.services.inference import InferenceService
from app.services.llm_export import ExportService
from app.services.model_upload import ModelUploadService
from app.services.projects import ProjectService
from app.services.recipes import RecipeService
from app.services.serving import ServingService
from app.services.settings import SettingsService
from app.services.training import TrainingService

settings = get_settings()
storage = Storage(settings)
registry = ModelRegistry(settings, storage)

project_service = ProjectService()
settings_service = SettingsService(settings)
dataset_service = DatasetService(settings, storage)
dataset_hub_service = DatasetHubService(settings, storage, dataset_service)
# Phase 21. Reuses the OpenRouter key/model the user already configured for
# recipes, so there is one place a model is chosen.
# The prep agent (phase 21) runs off the request path for the same reason
# recipes do: applying a plan to a 15,000-row upload is ~90,000 file operations,
# which is far longer than a proxy will hold a request open.
prep_executor = ThreadPoolExecutor(
    max_workers=settings.prep_executor_workers,
    thread_name_prefix="prep-executor",
)
dataset_prep_service = DatasetPrepService(
    settings, dataset_service, settings_service, prep_executor
)
inference_service = InferenceService(storage, registry)
model_upload_service = ModelUploadService(settings, storage, registry)
evaluation_service = EvaluationService(settings, storage, registry, dataset_service)
training_service = TrainingService(settings, storage, registry, dataset_service)
# Phase 17: graph CRUD and validation only. The model build runs in a
# subprocess, so this service never imports TensorFlow.
architecture_service = ArchitectureService(settings, storage)

# One executor per job domain, dispatching training/evaluation/inference
# jobs off the request-serving path. Kept small by default so long-running
# jobs cannot starve API responsiveness; sized from settings for deployments
# that need more. Separate pools (rather than one shared pool) mean a couple
# of long training jobs can't also starve inference/evaluation of workers.
training_executor = ThreadPoolExecutor(
    max_workers=settings.training_executor_workers,
    thread_name_prefix="training-executor",
)
evaluation_executor = ThreadPoolExecutor(
    max_workers=settings.evaluation_executor_workers,
    thread_name_prefix="evaluation-executor",
)
inference_executor = ThreadPoolExecutor(
    max_workers=settings.inference_executor_workers,
    thread_name_prefix="inference-executor",
)
# Recipe generation (phase 11) runs off the request path; the frontend polls
# recipe status rather than holding the request open.
recipe_executor = ThreadPoolExecutor(
    max_workers=settings.recipe_executor_workers,
    thread_name_prefix="recipe-executor",
)
recipe_service = RecipeService(settings, storage, dataset_service, recipe_executor)
# LLM export jobs (phase 15) follow the same off-request pattern; exports and
# training share nothing at runtime, so they never compete for the training
# executor slot.
export_executor = ThreadPoolExecutor(
    max_workers=settings.export_executor_workers,
    thread_name_prefix="export-executor",
)
export_service = ExportService(settings, storage, registry, export_executor)
serving_service = ServingService(settings, storage, registry)


def refresh_settings() -> None:
    global settings
    get_settings.cache_clear()
    settings = get_settings()
    settings_service.settings = settings
    registry.settings = settings
    model_upload_service.settings = settings
    dataset_service.settings = settings
    dataset_hub_service.settings = settings
    dataset_prep_service.settings = settings
    recipe_service.settings = settings
    evaluation_service.settings = settings
    training_service.settings = settings
    architecture_service.settings = settings
    export_service.settings = settings
    serving_service.settings = settings
