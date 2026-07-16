from datetime import datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, field_validator

from app.core.defaults import DEFAULT_LABELS, DEFAULT_PROJECT_ID, DEFAULT_TASK_TYPE

# Canonical task-type values. `"text"` was a legacy alias for
# `"text_classification"`; it is no longer a valid stored/output value, but
# `TASK_TYPE_ALIASES` below keeps it accepted on input during a deprecation
# window (see `_normalize_task_type_field`/`_normalize_task_type_list_field`).
TaskType = Literal[
    "classification",
    "object_detection",
    "segmentation",
    "text_classification",
    "summarization",
    "question_answering",
]
DatasetFormat = Literal[
    "yolo",
    "coco",
    "image_folder",
    "image_manifest",
    "text_folder",
    "jsonl",
    "csv",
]
DatasetSource = Literal["reference", "editable"]
SplitName = Literal["unassigned", "train", "valid", "test"]

# `DEFAULT_TASK_TYPE` lives in app.core.defaults as a plain `str` (shared with
# non-Pydantic code), so it needs a one-time cast here to satisfy the `TaskType`
# Literal used by the fields below.
_DEFAULT_TASK_TYPE: TaskType = cast(TaskType, DEFAULT_TASK_TYPE)
_DEFAULT_TASK_TYPES: list[TaskType] = [_DEFAULT_TASK_TYPE]
_DEFAULT_SPLITS: list[SplitName] = ["train", "valid", "test"]
_DEFAULT_AUGMENTATION_SPLITS: list[SplitName] = ["train"]

# Deprecated input aliases for `TaskType` values. Accepted on input (see the
# `_normalize_task_type_*` validators below) and normalized to the canonical
# value before Literal validation runs, so every stored/serialized task_type
# is always canonical.
TASK_TYPE_ALIASES: dict[str, str] = {"text": "text_classification"}


def _normalize_task_type_field(value: object) -> object:
    """`field_validator(mode="before")` for single `task_type: TaskType` fields."""
    if isinstance(value, str):
        return TASK_TYPE_ALIASES.get(value, value)
    return value


def _normalize_task_type_list_field(value: object) -> object:
    """`field_validator(mode="before")` for `task_types: list[TaskType]` fields."""
    if isinstance(value, list):
        return [TASK_TYPE_ALIASES.get(item, item) if isinstance(item, str) else item for item in value]
    return value


class ProjectSummary(BaseModel):
    id: str
    name: str
    description: str | None = None
    task_types: list[TaskType] = Field(default_factory=lambda: list(_DEFAULT_TASK_TYPES))
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    task_types: list[TaskType] = Field(default_factory=lambda: list(_DEFAULT_TASK_TYPES))
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalize_task_types = field_validator("task_types", mode="before")(
        _normalize_task_type_list_field
    )


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    task_types: list[TaskType] | None = None
    metadata: dict[str, Any] | None = None

    _normalize_task_types = field_validator("task_types", mode="before")(
        _normalize_task_type_list_field
    )


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
    question: str | None = Field(default=None, max_length=1000)
    max_length: int = Field(default=120, ge=1, le=2000)


class NlpTextClassificationResult(BaseModel):
    task: Literal["text_classification"] = "text_classification"
    label: str
    scores: dict[str, float] = Field(default_factory=dict)


class NlpSummarizationResult(BaseModel):
    task: Literal["summarization"] = "summarization"
    summary: str


class NlpQuestionAnsweringResult(BaseModel):
    task: Literal["question_answering"] = "question_answering"
    answer: str
    score: float


# Every NLP predictor's `predict_text` returns exactly one of these three
# shapes (see app/ml/nlp/*/predictors.py and app/ml/nlp/keras_classifier.py).
# Pydantic's "smart" union mode picks the right member from the fields
# present, so the `task` tag is optional on input for backward compatibility.
NlpResult = NlpTextClassificationResult | NlpSummarizationResult | NlpQuestionAnsweringResult


class InferenceResult(BaseModel):
    id: str
    project_id: str = DEFAULT_PROJECT_ID
    model_id: str
    input_type: Literal["image", "text"] = "image"
    image_level_label: str
    detections: list[Detection] = Field(default_factory=list)
    class_scores: dict[str, float] = Field(default_factory=dict)
    overlay_url: str | None = None
    original_url: str | None = None
    text_content: str | None = None
    nlp_result: NlpResult | None = None
    parameters: InferenceParameters
    created_at: datetime
    duration_ms: int | None = None
    timings: dict[str, int] = Field(default_factory=dict)


class JobProgress(BaseModel):
    percent: float = Field(default=0.0, ge=0.0, le=100.0)
    processed: int = 0
    total: int | None = None
    current_step: str = "Queued"
    current_item: str | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    logs: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class InferenceJobRead(BaseModel):
    id: str
    project_id: str = DEFAULT_PROJECT_ID
    model_id: str
    status: str
    progress: JobProgress
    result: InferenceResult | None
    artifacts: dict[str, Any]
    error: str | None
    created_at: datetime
    updated_at: datetime


class ModelInfo(BaseModel):
    id: str
    name: str
    family: str
    description: str
    available: bool
    paths: dict[str, str]
    promoted: bool = False
    project_id: str = DEFAULT_PROJECT_ID
    task_type: TaskType = _DEFAULT_TASK_TYPE
    labels: list[str] = Field(default_factory=lambda: DEFAULT_LABELS.copy())
    source: Literal["reference", "trained", "promoted"] = "reference"
    training_job_id: str | None = None
    created_at: datetime | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class ModelUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class PlatformSettingsRead(BaseModel):
    huggingface_hub_token_configured: bool = False


class PlatformSettingsUpdate(BaseModel):
    huggingface_hub_token: str | None = Field(default=None, max_length=4096)


class EvaluationDatasetInfo(BaseModel):
    key: str
    name: str
    project_id: str = DEFAULT_PROJECT_ID
    task_type: TaskType = _DEFAULT_TASK_TYPE
    format: DatasetFormat
    split: str
    available: bool
    path: str
    labels: list[str] = Field(default_factory=lambda: DEFAULT_LABELS.copy())


class EvaluationJobCreate(BaseModel):
    project_id: str = DEFAULT_PROJECT_ID
    model_id: str
    dataset_key: str
    limit: int | None = Field(default=None, ge=1, le=500)


class EvaluationJobBatchCreate(BaseModel):
    project_id: str = DEFAULT_PROJECT_ID
    model_ids: list[str] = Field(min_length=1, max_length=10)
    dataset_key: str
    limit: int | None = Field(default=None, ge=1, le=500)


class EvaluationJobRead(BaseModel):
    id: str
    project_id: str = DEFAULT_PROJECT_ID
    comparison_id: str | None = None
    model_id: str
    dataset_key: str
    status: str
    limit: int | None
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    progress: JobProgress
    error: str | None
    created_at: datetime
    updated_at: datetime


class ComparisonMetricSummary(BaseModel):
    """One row of `api.serializers.comparison_metric_summary`'s fixed output shape.

    Every evaluation task type populates a different subset of the optional
    fields below (e.g. `accuracy`/`macro_f1` for classification,
    `rougeL` for summarization), but the row's key set never changes.
    """

    job_id: str
    model_id: str
    status: str
    samples: int | None = None
    accuracy: float | None = None
    macro_f1: float | None = None
    text_accuracy: float | None = None
    text_macro_f1: float | None = None
    rougeL: float | None = None
    exact_match: float | None = None
    qa_f1: float | None = None
    pixel_dice: float | None = None
    object_recall: float | None = None


class EvaluationComparisonRead(BaseModel):
    comparison_id: str
    project_id: str = DEFAULT_PROJECT_ID
    dataset_key: str
    jobs: list[EvaluationJobRead]
    metrics: list[ComparisonMetricSummary] = Field(default_factory=list)


class EvaluationPerImageRow(BaseModel):
    image: str
    ground_truth: int
    prediction: int
    detections: int
    objects: int
    pixel: dict[str, float] = Field(default_factory=dict)
    object: dict[str, int] = Field(default_factory=dict)
    text_preview: str | None = None
    reference_text: str | None = None
    prediction_text: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)


class TrainingJobCreate(BaseModel):
    project_id: str = DEFAULT_PROJECT_ID
    task_type: TaskType = _DEFAULT_TASK_TYPE
    model_family: str = "yolo"
    model_option_id: str = "yolo_local"
    model_name: str | None = Field(default=None, max_length=120)
    base_model: str | None = None
    epochs: int = Field(default=50, ge=1, le=1000)
    image_size: int = Field(default=512, ge=128, le=2048)
    batch_size: int = Field(default=16, ge=-1, le=256)
    dataset_id: str = "reference_yolo"
    optimizer: str = "AdamW"
    learning_rate: float = Field(default=0.002, gt=0.0, le=1.0)
    architecture: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    device: str = ""
    cache: Literal["disk", "ram", "none"] = "disk"
    workers: int = Field(default=0, ge=0, le=16)
    patience: int = Field(default=50, ge=0, le=500)

    _normalize_task_type = field_validator("task_type", mode="before")(
        _normalize_task_type_field
    )


class TrainingJobRead(BaseModel):
    id: str
    project_id: str = DEFAULT_PROJECT_ID
    task_type: str = DEFAULT_TASK_TYPE
    model_family: str
    status: str
    # Always the stored `TrainingJobCreate.model_dump()` payload for the job
    # (with `model_family` re-resolved to the training option's family), so
    # the request schema doubles as the stable shape of the persisted record.
    parameters: TrainingJobCreate
    artifacts: dict[str, Any]
    artifact_urls: dict[str, str] = Field(default_factory=dict)
    progress: JobProgress
    # Per-task/family metrics payloads (e.g. YOLO detection metrics vs. NLP
    # classification metrics) are genuinely shaped differently, so this stays
    # a deliberate open dict rather than a single fixed model.
    metrics: dict[str, Any] = Field(default_factory=dict)
    # Per-epoch rows read from `results.csv`/`metrics.json`; column names vary
    # by model family, but every value collected by
    # `training.artifacts.parse_training_history` is numeric.
    history: list[dict[str, float | int]] = Field(default_factory=list)
    curves: dict[str, Any] = Field(default_factory=dict)
    promoted_model_id: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class DatasetSplitSummary(BaseModel):
    split: SplitName
    image_count: int = 0
    text_count: int = 0
    item_count: int = 0
    annotation_count: int = 0


class DatasetSummary(BaseModel):
    id: str
    project_id: str = DEFAULT_PROJECT_ID
    name: str
    task_type: TaskType = _DEFAULT_TASK_TYPE
    format: DatasetFormat
    source: DatasetSource
    editable: bool
    path: str
    labels: list[str]
    classes: list[str]
    splits: dict[str, DatasetSplitSummary]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetCreate(BaseModel):
    project_id: str = DEFAULT_PROJECT_ID
    name: str = Field(min_length=1, max_length=120)
    task_type: TaskType = _DEFAULT_TASK_TYPE
    format: DatasetFormat = "yolo"
    labels: list[str] = Field(default_factory=lambda: DEFAULT_LABELS.copy())

    _normalize_task_type = field_validator("task_type", mode="before")(
        _normalize_task_type_field
    )


class DatasetPreprocessConfig(BaseModel):
    enabled: bool = False
    preset: Literal["none", "light", "inspection", "nlp_clean", "nlp_augment"] = "none"
    resize_width: int | None = Field(default=None, ge=32, le=4096)
    resize_height: int | None = Field(default=None, ge=32, le=4096)
    normalize: bool = False
    transforms: list[str] = Field(default_factory=list)
    augmentation_mode: Literal["random", "materialize"] = "random"
    copies_per_image: int = Field(default=4, ge=0, le=20)


class DatasetSplitConfig(BaseModel):
    train: float = Field(default=0.7, ge=0.0, le=1.0)
    valid: float = Field(default=0.2, ge=0.0, le=1.0)
    test: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = 42
    stratify: bool = True
    resplit_all: bool = False


class DatasetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    metadata: dict[str, Any] | None = None
    preprocess: DatasetPreprocessConfig | None = None


class DatasetImportRequest(BaseModel):
    project_id: str = DEFAULT_PROJECT_ID
    path: str
    name: str | None = Field(default=None, max_length=120)
    task_type: TaskType = _DEFAULT_TASK_TYPE
    format: DatasetFormat = "yolo"
    labels: list[str] | None = None

    _normalize_task_type = field_validator("task_type", mode="before")(
        _normalize_task_type_field
    )


class DatasetCloneRequest(BaseModel):
    project_id: str | None = None
    name: str | None = Field(default=None, max_length=120)


class DatasetAnnotation(BaseModel):
    class_id: int = Field(default=0, ge=0)
    class_name: str = ""
    kind: Literal["classification", "box", "polygon", "summary", "qa"] = "polygon"
    bbox: Box | None = None
    polygon: list[list[float]] = Field(default_factory=list)
    text: str | None = None
    question: str | None = None
    answer: str | None = None


class DatasetItemSummary(BaseModel):
    id: str
    dataset_id: str
    split: SplitName
    filename: str
    media_type: Literal["image", "text"] = "image"
    image_url: str = ""
    text_url: str | None = None
    text_preview: str | None = None
    width: int = 0
    height: int = 0
    annotation_count: int
    classes: list[str]
    class_id: int | None = None
    label: str | None = None
    is_labeled: bool = False
    annotations: list[DatasetAnnotation] = Field(default_factory=list)


class DatasetItemDetail(DatasetItemSummary):
    annotations: list[DatasetAnnotation]
    text_content: str | None = None


class DatasetItemPage(BaseModel):
    items: list[DatasetItemSummary] = Field(default_factory=list)
    total: int = 0
    limit: int
    offset: int = 0


class DatasetAnnotationSave(BaseModel):
    annotations: list[DatasetAnnotation]


class DatasetItemLabelUpdate(BaseModel):
    class_id: int | None = Field(default=None, ge=0)
    class_name: str | None = Field(default=None, min_length=1, max_length=80)


class DatasetItemBatchUploadResponse(BaseModel):
    uploaded: list[DatasetItemDetail] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)


class DatasetItemDeleteRequest(BaseModel):
    split: SplitName
    ids: list[str] = Field(min_length=1)


class DatasetItemMoveRequest(BaseModel):
    source_split: SplitName
    target_split: SplitName
    ids: list[str] = Field(min_length=1)


class DatasetItemMoveResponse(BaseModel):
    moved: int = 0
    missing: list[str] = Field(default_factory=list)
    items: list[DatasetItemSummary] = Field(default_factory=list)


class DatasetItemBulkLabelUpdate(BaseModel):
    ids: list[str] = Field(min_length=1)
    class_id: int | None = Field(default=None, ge=0)
    class_name: str | None = Field(default=None, min_length=1, max_length=80)


class DatasetItemBulkLabelUpdateResponse(BaseModel):
    updated: int = 0
    missing: list[str] = Field(default_factory=list)
    items: list[DatasetItemSummary] = Field(default_factory=list)


class DatasetPreprocessPreviewRequest(BaseModel):
    config: DatasetPreprocessConfig | None = None


class DatasetPreprocessPreview(BaseModel):
    dataset_id: str
    split: SplitName
    item_id: str
    media_type: Literal["image", "text"] = "image"
    image_url: str = ""
    text_preview: str | None = None
    config: DatasetPreprocessConfig


class DatasetVersionCreate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    config: DatasetPreprocessConfig | None = None
    splits: list[SplitName] = Field(default_factory=lambda: list(_DEFAULT_SPLITS))
    augmentation_splits: list[SplitName] = Field(default_factory=lambda: list(_DEFAULT_AUGMENTATION_SPLITS))


class DatasetVersionSummary(BaseModel):
    id: str
    dataset_id: str
    name: str
    path: str
    image_count: int = 0
    text_count: int = 0
    item_count: int = 0
    generated_count: int = 0
    splits: dict[str, DatasetSplitSummary]
    config: DatasetPreprocessConfig
    created_at: datetime


class DatasetProcessRequest(BaseModel):
    preprocess: DatasetPreprocessConfig | None = None
    split: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)


class DatasetProcessResponse(BaseModel):
    dataset: DatasetSummary
    moved: dict[str, int] = Field(default_factory=dict)
    split_config: DatasetSplitConfig


class DatasetEdaSummary(BaseModel):
    dataset_id: str
    split: str
    split_counts: dict[str, int]
    class_counts: dict[str, int]
    unlabeled_count: int = 0
    missing_annotation_count: int = 0
    image_count: int = 0
    text_count: int = 0
    item_count: int = 0
    annotation_count: int = 0
    image_size: dict[str, float | int | None] = Field(default_factory=dict)
    aspect_ratio: dict[str, float | None] = Field(default_factory=dict)
    text_length: dict[str, float | int | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DatasetLabelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class DatasetLabelUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class DeleteRequest(BaseModel):
    ids: list[str] = Field(default_factory=list)
    project_id: str | None = None
    clear_all: bool = False


class DeleteResponse(BaseModel):
    deleted: int = 0
    blocked: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class TrainingModelOption(BaseModel):
    id: str
    name: str
    family: str
    task_types: list[TaskType]
    source: Literal["local", "ultralytics", "keras_applications", "huggingface"]
    runnable: bool
    needs_download: bool = False
    description: str
    defaults: dict[str, Any] = Field(default_factory=dict)


class ModelAssetPrepareRequest(BaseModel):
    option_id: str
    download: bool = True


class ModelAssetStatus(BaseModel):
    option_id: str
    status: Literal["ready", "missing", "gated", "failed"]
    path: str | None = None
    message: str | None = None
