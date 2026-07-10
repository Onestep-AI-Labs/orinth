from dataclasses import dataclass
from pathlib import Path


@dataclass
class GroundTruthObject:
    class_id: int
    polygon: list[list[float]]
    bbox: tuple[float, float, float, float]


@dataclass
class EvaluationSample:
    image_path: Path
    objects: list[GroundTruthObject]


@dataclass
class ClassificationSample:
    image_path: Path
    class_id: int
