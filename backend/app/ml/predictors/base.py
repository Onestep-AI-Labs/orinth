from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas import Detection, InferenceParameters


class Predictor(ABC):
    @abstractmethod
    def predict(self, image_path: Path, parameters: InferenceParameters) -> list[Detection]:
        raise NotImplementedError

    def classify(self, image_path: Path, parameters: InferenceParameters) -> dict[str, float] | None:
        return None

    def predict_text(self, text: str, parameters: InferenceParameters) -> dict:
        raise NotImplementedError("This model does not support text inference")


def image_level_from_detections(detections: list[Detection]) -> str:
    if not detections:
        return "Normal"
    largest = max(detections, key=lambda item: item.mask_area)
    return largest.class_name
