from app.core.config import get_settings
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry
from app.services.datasets import DatasetService
from app.services.evaluation import EvaluationService
from app.services.inference import InferenceService
from app.services.projects import ProjectService
from app.services.training import TrainingService

settings = get_settings()
storage = Storage(settings)
registry = ModelRegistry(settings, storage)

project_service = ProjectService()
dataset_service = DatasetService(settings, storage)
inference_service = InferenceService(storage, registry)
evaluation_service = EvaluationService(settings, storage, registry, dataset_service)
training_service = TrainingService(settings, storage, registry, dataset_service)
