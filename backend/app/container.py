from concurrent.futures import ThreadPoolExecutor

from app.core.config import get_settings
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry
from app.services.datasets import DatasetService
from app.services.evaluation import EvaluationService
from app.services.inference import InferenceService
from app.services.projects import ProjectService
from app.services.settings import SettingsService
from app.services.training import TrainingService

settings = get_settings()
storage = Storage(settings)
registry = ModelRegistry(settings, storage)

project_service = ProjectService()
settings_service = SettingsService(settings)
dataset_service = DatasetService(settings, storage)
inference_service = InferenceService(storage, registry)
evaluation_service = EvaluationService(settings, storage, registry, dataset_service)
training_service = TrainingService(settings, storage, registry, dataset_service)

# Shared executor for dispatching training/evaluation/inference jobs off the
# request-serving path. Kept small by default so long-running jobs cannot
# starve API responsiveness; sized from settings for deployments that need more.
job_executor = ThreadPoolExecutor(
    max_workers=settings.job_executor_workers,
    thread_name_prefix="job-executor",
)


def refresh_settings() -> None:
    global settings
    get_settings.cache_clear()
    settings = get_settings()
    settings_service.settings = settings
    registry.settings = settings
    dataset_service.settings = settings
    evaluation_service.settings = settings
    training_service.settings = settings
