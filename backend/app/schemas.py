from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Box(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Box
    polygon: list[list[float]] = Field(default_factory=list)
    mask_area: float = 0.0


class InferenceParameters(BaseModel):
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class InferenceResult(BaseModel):
    id: str
    model_id: str
    image_level_label: str
    detections: list[Detection]
    overlay_url: str | None = None
    original_url: str | None = None
    parameters: InferenceParameters
    created_at: datetime


class ModelInfo(BaseModel):
    id: str
    name: str
    family: Literal["yolo", "unet_inception"]
    description: str
    available: bool
    paths: dict[str, str]
    promoted: bool = False


class EvaluationDatasetInfo(BaseModel):
    key: str
    name: str
    format: Literal["yolo", "coco"]
    split: str
    available: bool
    path: str


class EvaluationJobCreate(BaseModel):
    model_id: str
    dataset_key: str
    limit: int | None = Field(default=None, ge=1, le=500)


class EvaluationJobRead(BaseModel):
    id: str
    model_id: str
    dataset_key: str
    status: str
    limit: int | None
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    error: str | None
    created_at: datetime
    updated_at: datetime


class TrainingJobCreate(BaseModel):
    model_family: Literal["yolo", "unet_inception"]
    epochs: int = Field(default=50, ge=1, le=1000)
    image_size: int = Field(default=512, ge=128, le=2048)
    batch_size: int = Field(default=16, ge=-1, le=256)


class TrainingJobRead(BaseModel):
    id: str
    model_family: str
    status: str
    parameters: dict[str, Any]
    artifacts: dict[str, Any]
    promoted_model_id: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime
