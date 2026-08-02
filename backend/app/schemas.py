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
    "llm_finetune",
    # Phase 17: next-token prediction for architectures built in the studio.
    # Distinct from `llm_finetune`, which adapts a pretrained base model.
    "language_modeling",
]
DatasetFormat = Literal[
    "yolo",
    "coco",
    "image_folder",
    "image_manifest",
    "text_folder",
    "jsonl",
    "csv",
    "instruction_jsonl",
    "chat_jsonl",
]
DatasetSource = Literal["reference", "editable"]
# Manifest provenance grouping the catalog into Project / Imported / Shared
# sections (see phase 10). `recipe` is produced by phase 11 document generation.
DatasetOrigin = Literal["created", "imported_hf", "recipe"]
SplitName = Literal["unassigned", "train", "valid", "test"]
ChatRole = Literal["system", "user", "assistant"]

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
# `tabular` came from the removed Parquet-backed dataset kind (commit b70baa3).
# Datasets written by that build still exist on disk; without an alias every
# one of them fails validation and takes the whole dataset list down with it.
# A labelled table read as rows of text is `text_classification` — the same
# thing the removed pipeline ultimately produced.
TASK_TYPE_ALIASES: dict[str, str] = {
    "text": "text_classification",
    "tabular": "text_classification",
}

# Same deprecation window for the format that shipped alongside it.
DATASET_FORMAT_ALIASES: dict[str, str] = {"table": "csv"}


def _normalize_dataset_format_field(value: object) -> object:
    """`field_validator(mode="before")` for `format: DatasetFormat` fields."""
    if isinstance(value, str):
        return DATASET_FORMAT_ALIASES.get(value, value)
    return value


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
    # Stored inside the `metadata` JSON column under the reserved `archived` key
    # (no migration), but surfaced as a typed field so the API contract is explicit.
    archived: bool = False
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
    archived: bool | None = None

    _normalize_task_types = field_validator("task_types", mode="before")(
        _normalize_task_type_list_field
    )


class ProjectStats(BaseModel):
    project_id: str
    datasets: int = 0
    training_jobs: int = 0
    evaluation_jobs: int = 0
    inference_jobs: int = 0
    inference_runs: int = 0
    deletable: bool = True
    # Same strings `delete_project` raises with, so the danger zone and the
    # eventual 409 message cannot drift apart.
    blockers: list[str] = Field(default_factory=list)


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
    source: Literal["reference", "trained", "promoted", "uploaded"] = "reference"
    training_job_id: str | None = None
    # Uploaded LLM adapters point at the base model they were tuned against
    # (phase 12/14). `format` records the on-disk artifact shape (e.g.
    # "safetensors", "gguf", "pt", "keras", "joblib") for uploaded weights.
    base_model_id: str | None = None
    format: str | None = None
    size_bytes: int | None = None
    created_at: datetime | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class ModelUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ModelUploadFileSlot(BaseModel):
    """One file input the upload form should render for a family."""

    key: str
    label: str
    accept: list[str]
    required: bool = True


class ModelUploadField(BaseModel):
    """One extra metadata field the upload form should collect."""

    key: str
    label: str
    type: Literal["text", "labels", "number", "select"]
    required: bool = False
    options: list[str] | None = None
    placeholder: str | None = None
    help: str | None = None


class ModelUploadOption(BaseModel):
    """Per-family descriptor driving the dynamic upload form.

    The frontend renders the form purely from this, so adding a family later
    is a backend-only change.
    """

    family: str
    label: str
    description: str
    kind: Literal["single", "pair", "zip"]
    files: list[ModelUploadFileSlot]
    fields: list[ModelUploadField] = Field(default_factory=list)
    servable: bool = True
    gate_note: str | None = None
    security_note: str | None = None


class ModelUploadResult(BaseModel):
    model: ModelInfo
    # Only structural validation runs at upload time; deep validation stays
    # deferred to first use, exactly like every other registry family.
    validated: Literal["structural"] = "structural"
    duplicate_name: bool = False
    warnings: list[str] = Field(default_factory=list)


class PlatformSettingsRead(BaseModel):
    huggingface_hub_token_configured: bool = False
    # OpenRouter powers phase-11 LLM-assisted recipe generation. The key is
    # write-only: reads only expose whether one is configured, never the value.
    openrouter_api_key_configured: bool = False
    openrouter_model: str | None = None


class PlatformSettingsUpdate(BaseModel):
    huggingface_hub_token: str | None = Field(default=None, max_length=4096)
    openrouter_api_key: str | None = Field(default=None, max_length=4096)
    openrouter_model: str | None = Field(default=None, max_length=200)


class EvaluationDatasetInfo(BaseModel):
    key: str
    name: str
    project_id: str = DEFAULT_PROJECT_ID
    task_type: TaskType = _DEFAULT_TASK_TYPE
    format: DatasetFormat
    split: str
    available: bool
    shared: bool = False
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
    shared: bool = False
    # Provenance used to group the catalog (Project / Imported / Shared). Legacy
    # manifests without the field default to `created`.
    origin: DatasetOrigin = "created"
    origin_ref: str | None = None
    path: str
    labels: list[str]
    classes: list[str]
    splits: dict[str, DatasetSplitSummary]
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Manifests on disk outlive the schema. A dataset written by a build whose
    # task type or format has since been removed must still list, or one stale
    # directory takes the whole catalog down.
    _normalize_task_type = field_validator("task_type", mode="before")(
        _normalize_task_type_field
    )
    _normalize_format = field_validator("format", mode="before")(
        _normalize_dataset_format_field
    )


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
    media_type: Literal["image", "text", "record"] = "image"
    image_url: str = ""
    text_url: str | None = None
    text_preview: str | None = None
    # LLM record excerpts: `text_preview` carries the instruction / first-user
    # excerpt, `output_preview` the output / last-assistant excerpt, and
    # `token_estimate` a heuristic (whitespace + punctuation) count so the
    # Records table never loads a tokenizer.
    output_preview: str | None = None
    token_estimate: int = 0
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
    # Full parsed record for `llm_finetune` items (instruction/chat shape); None
    # for image/text items.
    record: dict[str, Any] | None = None


class DatasetRecordCreate(BaseModel):
    split: SplitName = "unassigned"
    record: dict[str, Any]


class DatasetRecordSave(BaseModel):
    record: dict[str, Any]


class DatasetRecordUploadResponse(BaseModel):
    imported: int = 0
    skipped: int = 0
    warnings: list[str] = Field(default_factory=list)


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
    # `llm_finetune` only: role message counts (chat data) and duplicate/empty
    # counts surfaced as warnings. Empty for image/text datasets.
    role_counts: dict[str, int] = Field(default_factory=dict)
    duplicate_count: int = 0
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


class AdvancedParameterSpec(BaseModel):
    """A single catalog-declared advanced training hyperparameter.

    The frontend renders these generically from `type`/`min`/`max`/`step`/
    `options` and groups them by `group`, so a new model family gets its
    advanced UI purely by declaring specs — the form has no per-family
    knowledge. Values ride the open-ended `TrainingJobCreate.hyperparameters`
    dict, and each runner owns an allowlist that maps `key` to a runner
    argument (see `app/training/runners/advanced.py`).
    """

    key: str
    label: str
    # "code" is "text" that renders as a mono textarea rather than a single
    # line — added in phase 17 for custom-layer node bodies.
    type: Literal["int", "float", "bool", "select", "multiselect", "text", "code"]
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[Any] = Field(default_factory=list)
    help: str | None = None
    # One of the group micro-headers the form renders: Optimization,
    # Augmentation, Regularization, Runtime.
    group: str = "Optimization"


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
    # Empty for untouched catalogs, so they serialize exactly as before.
    advanced_parameters: list[AdvancedParameterSpec] = Field(default_factory=list)


class ModelAssetPrepareRequest(BaseModel):
    option_id: str
    download: bool = True
    # For the "custom Hugging Face model" LLM base option: the hub id the user
    # typed, since it is not resolvable from a fixed catalog entry.
    model_ref: str | None = Field(default=None, max_length=200)


class ModelAssetStatus(BaseModel):
    option_id: str
    status: Literal["ready", "missing", "gated", "failed"]
    path: str | None = None
    message: str | None = None


class LlmEnvironment(BaseModel):
    """What an LLM fine-tuning run would actually do on this machine (phase 14).

    Rendered as an info banner on the training form so the Unsloth-vs-PEFT
    backend choice and any capability gaps are visible before a job is
    submitted, not discovered in the run log.
    """

    device: Literal["cuda", "mps", "cpu"]
    unsloth_available: bool = False
    peft_available: bool = False
    bitsandbytes_available: bool = False
    recommended_backend: Literal["unsloth", "peft"]
    notes: list[str] = Field(default_factory=list)


class LlmModelInfo(BaseModel):
    """Accurate base-model details from the Hugging Face Hub API (phase 14).

    Powers the training form's "model detail" line so download size, file count,
    and gating are real values from the Hub rather than the catalog's rough
    estimate. ``exists`` is false when the repo id is unknown; ``gated`` is true
    when the repo requires an accepted license or the token lacks access.
    """

    model_ref: str
    exists: bool = False
    gated: bool = False
    size_bytes: int | None = None
    file_count: int | None = None
    downloads: int | None = None
    likes: int | None = None
    library: str | None = None
    error: str | None = None


class DatasetHubSearchResult(BaseModel):
    hub_id: str
    author: str | None = None
    downloads: int = 0
    likes: int = 0
    gated: bool = False
    tags: list[str] = Field(default_factory=list)
    updated_at: str | None = None


class DatasetHubSearchResponse(BaseModel):
    results: list[DatasetHubSearchResult] = Field(default_factory=list)
    # Hub failures surface here instead of a 500, so the local catalog keeps
    # rendering when the Hub is unreachable or rate-limited.
    error: str | None = None


class DatasetHubPreview(BaseModel):
    hub_id: str
    config: str | None = None
    split: str | None = None
    configs: list[str] = Field(default_factory=list)
    splits: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    # Detected record shape ("alpaca", "sharegpt", "messages", "qa") when the
    # heuristics succeed, so the mapping dialog can pre-fill and show a banner.
    detected_format: str | None = None
    detected_mapping: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class DatasetHubColumnMapping(BaseModel):
    instruction: str | None = None
    input: str | None = None
    output: str | None = None
    messages: str | None = None
    conversations: str | None = None
    question: str | None = None
    context: str | None = None
    answer: str | None = None


class DatasetHubImportRequest(BaseModel):
    project_id: str = DEFAULT_PROJECT_ID
    hub_id: str = Field(min_length=1, max_length=200)
    config: str | None = None
    split: str = "train"
    task_type: TaskType = "llm_finetune"
    format: DatasetFormat = "instruction_jsonl"
    name: str | None = Field(default=None, max_length=120)
    max_rows: int = Field(default=5000, ge=1, le=5000)
    mapping: DatasetHubColumnMapping = Field(default_factory=DatasetHubColumnMapping)

    _normalize_task_type = field_validator("task_type", mode="before")(
        _normalize_task_type_field
    )


class DatasetHubImportResponse(BaseModel):
    dataset: DatasetSummary
    imported_rows: int = 0
    skipped_rows: int = 0
    warnings: list[str] = Field(default_factory=list)


# --- Phase 11: Data Recipes ------------------------------------------------

# Only the two LLM record shapes are valid recipe outputs; recipes always commit
# to an `llm_finetune` dataset (see phase 10).
RecipeOutputFormat = Literal["instruction_jsonl", "chat_jsonl"]
RecipeStatus = Literal["draft", "extracting", "generating", "ready", "failed"]
RecipeGenerationMode = Literal["auto", "llm", "rules"]
# How the rule-based / LLM prompt frames each chunk. Presets pick one; it stays
# editable afterward.
RecipePromptFlavor = Literal["qa", "instruction", "conversation"]
# Which generator produced a record, so the review table can badge origin and
# per-chunk regeneration knows what to replace.
RecipeGenerator = Literal["llm", "rules"]


class RecipeGenerationSettings(BaseModel):
    mode: RecipeGenerationMode = "auto"
    prompt_flavor: RecipePromptFlavor = "qa"
    chunk_size: int = Field(default=3000, ge=200, le=20000)
    chunk_overlap: int = Field(default=200, ge=0, le=4000)
    records_per_chunk: int = Field(default=3, ge=1, le=20)
    # OpenRouter model override for this recipe; falls back to the platform
    # setting when unset. Never carries a key.
    model: str | None = Field(default=None, max_length=200)


class RecipeSourceRead(BaseModel):
    id: str
    filename: str
    media_type: str
    characters: int = 0
    pages: int | None = None
    excluded: bool = False
    warnings: list[str] = Field(default_factory=list)


class RecipeCreate(BaseModel):
    project_id: str = DEFAULT_PROJECT_ID
    name: str = Field(min_length=1, max_length=120)
    output_format: RecipeOutputFormat = "instruction_jsonl"
    generation: RecipeGenerationSettings = Field(default_factory=RecipeGenerationSettings)


class RecipeRead(BaseModel):
    id: str
    project_id: str = DEFAULT_PROJECT_ID
    name: str
    output_format: RecipeOutputFormat
    status: RecipeStatus
    sources: list[RecipeSourceRead] = Field(default_factory=list)
    generation: RecipeGenerationSettings = Field(default_factory=RecipeGenerationSettings)
    warnings: list[str] = Field(default_factory=list)
    record_count: int = 0
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class RecipeGenerateRequest(BaseModel):
    mode: RecipeGenerationMode = "auto"
    prompt_flavor: RecipePromptFlavor | None = None
    model: str | None = Field(default=None, max_length=200)
    chunk_size: int | None = Field(default=None, ge=200, le=20000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=4000)
    records_per_chunk: int | None = Field(default=None, ge=1, le=20)


class RecipeRecord(BaseModel):
    index: int
    record: dict[str, Any]
    generator: RecipeGenerator
    source_id: str | None = None
    chunk_index: int | None = None


class RecipeRecordPage(BaseModel):
    records: list[RecipeRecord] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 50


class RecipeRecordCreate(BaseModel):
    record: dict[str, Any]


class RecipeRecordUpdate(BaseModel):
    record: dict[str, Any]


class RecipeRecordDeleteRequest(BaseModel):
    indices: list[int] = Field(min_length=1)


class RecipeCommitRequest(BaseModel):
    name: str | None = Field(default=None, max_length=120)


class RecipeCommitResponse(BaseModel):
    dataset: DatasetSummary
    committed_records: int = 0


class OpenRouterModel(BaseModel):
    id: str
    name: str


class OpenRouterModelsResponse(BaseModel):
    models: list[OpenRouterModel] = Field(default_factory=list)
    # True when the list came live from OpenRouter with a configured key; False
    # when it is the curated fallback, so the select never renders empty.
    live: bool = False


# --- Phase 15: LLM export, serving, and chat -------------------------------

ModelExportFormat = Literal[
    "adapter_zip",
    "merged_16bit",
    "gguf_q4_k_m",
    "gguf_q5_k_m",
    "gguf_q8_0",
    "gguf_f16",
]
ModelExportState = Literal["queued", "running", "completed", "failed"]


class ModelExportRequest(BaseModel):
    format: ModelExportFormat


class ModelExportStatus(BaseModel):
    id: str
    model_id: str
    format: ModelExportFormat
    status: ModelExportState = "queued"
    error: str | None = None
    # Manifest-backed job surface (no DB row): the export panel polls this for
    # the live log tail while a job runs.
    current_step: str | None = None
    logs: list[str] = Field(default_factory=list)
    artifact_name: str | None = None
    size_bytes: int | None = None
    # Set when a completed gguf_* export was registered as a servable
    # `llm_gguf` model.
    registered_model_id: str | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None


class ModelExportOption(BaseModel):
    """One export format the detail page can offer for a given model."""

    format: ModelExportFormat
    label: str
    description: str
    available: bool = True
    note: str | None = None


class ServingStartRequest(BaseModel):
    # Serve either a registered GGUF model (model_id) or a GGUF file anywhere on
    # the local machine (model_path). Exactly one is required; the service
    # rejects a request with neither or both.
    model_id: str | None = None
    model_path: str | None = None
    context_length: int | None = Field(default=None, ge=256, le=131072)
    n_gpu_layers: int | None = Field(default=None, ge=-1, le=1000)


class ServingScanEntry(BaseModel):
    path: str
    name: str
    size_bytes: int | None = None
    # "registered" when this GGUF is already a catalog model (its id is set),
    # "file" when it is a loose file discovered on disk.
    kind: Literal["registered", "file"] = "file"
    model_id: str | None = None


class ServingScanResult(BaseModel):
    root: str
    entries: list[ServingScanEntry] = Field(default_factory=list)
    # Non-GGUF model directories found alongside (config.json + safetensors):
    # not directly servable, surfaced so the UI can point at Export.
    exportable_dirs: list[str] = Field(default_factory=list)
    truncated: bool = False


class ServingBrowseDir(BaseModel):
    name: str
    path: str


class ServingBrowseFile(BaseModel):
    name: str
    path: str
    size_bytes: int | None = None
    # A GGUF already in the catalog carries its model id so the UI can prefer it.
    model_id: str | None = None


class ServingBrowseResult(BaseModel):
    """One directory level for the server-side path browser."""

    path: str
    parent: str | None = None
    dirs: list[ServingBrowseDir] = Field(default_factory=list)
    gguf_files: list[ServingBrowseFile] = Field(default_factory=list)


class ServingConfig(BaseModel):
    """Persisted serving preferences (phase 15). ``models_dir`` is the last
    custom directory the user picked, restored and auto-scanned on load."""

    models_dir: str | None = None


class ServingConfigUpdate(BaseModel):
    models_dir: str | None = Field(default=None, max_length=4096)


class ServingPickRequest(BaseModel):
    kind: Literal["folder", "file"] = "folder"


class ServingPickResult(BaseModel):
    # `path` is None when the native dialog was canceled.
    path: str | None = None
    canceled: bool = False
    # Set when the OS dialog is unavailable (headless/unsupported platform), so
    # the UI can fall back to manual path entry.
    unavailable: bool = False
    message: str | None = None


class HubModelFile(BaseModel):
    filename: str
    size_bytes: int | None = None
    quantization: str | None = None


class HubModelResult(BaseModel):
    repo_id: str
    author: str | None = None
    format: Literal["gguf", "mlx"]
    downloads: int = 0
    likes: int = 0
    updated_at: str | None = None
    files: list[HubModelFile] = Field(default_factory=list)


class HubModelSearchResponse(BaseModel):
    results: list[HubModelResult] = Field(default_factory=list)
    # Hub failures surface here instead of a 500 so the picker keeps working.
    error: str | None = None


class HubFilesResponse(BaseModel):
    repo_id: str
    files: list[HubModelFile] = Field(default_factory=list)
    error: str | None = None


class HubRecommendation(BaseModel):
    repo_id: str
    title: str
    params: str
    format: Literal["gguf", "mlx"] = "gguf"
    approx_size_gb: float
    fits: bool
    note: str = ""


class HubRecommendationsResponse(BaseModel):
    device: Literal["cuda", "mps", "cpu"] = "cpu"
    total_memory_gb: float | None = None
    results: list[HubRecommendation] = Field(default_factory=list)


class HubDownloadRequest(BaseModel):
    repo_id: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=400)


class HubDownloadResult(BaseModel):
    path: str
    size_bytes: int | None = None
    servable: bool = True
    message: str | None = None


class ServingStatus(BaseModel):
    state: Literal["stopped", "starting", "running", "stopping"] = "stopped"
    model_id: str | None = None
    model_name: str | None = None
    port: int | None = None
    context_length: int | None = None
    started_at: datetime | None = None
    uptime_seconds: float | None = None
    last_activity_at: datetime | None = None
    # Tail of the subprocess log; populated after a crash or failed start so
    # the UI can surface why.
    stderr_tail: list[str] = Field(default_factory=list)
    error: str | None = None


class WebSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class WebSearchResponse(BaseModel):
    query: str
    results: list[WebSearchResult] = Field(default_factory=list)
    # Search failures surface here instead of a 500 so chat still answers.
    error: str | None = None


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    system: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1, le=32768)


# --- Phase 17: model architecture studio -----------------------------------


class ArchitectureNode(BaseModel):
    """One node on the canvas.

    `params` is intentionally open-ended: values are validated and coerced
    against the node's `NodeSpec.params` (reused `AdvancedParameterSpec`s) at
    parse time, so a stored graph survives a catalog that later adds a param.
    """

    id: str = Field(min_length=1, max_length=64)
    type: str = Field(min_length=1, max_length=64)
    label: str | None = None
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    params: dict[str, Any] = Field(default_factory=dict)


class ArchitectureEdge(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    source: str
    target: str
    # Reserved for multi-port nodes (attention query/key/value). Single-port
    # nodes leave these at "out"/"in".
    source_port: str = "out"
    target_port: str = "in"


class ArchitectureGraph(BaseModel):
    schema_version: int = 1
    nodes: list[ArchitectureNode] = Field(default_factory=list)
    edges: list[ArchitectureEdge] = Field(default_factory=list)
    # Hyperparameter defaults the studio prefills into the training form.
    training_defaults: dict[str, Any] = Field(default_factory=dict)


class Architecture(BaseModel):
    id: str
    project_id: str | None = None
    name: str
    description: str | None = None
    task_type: TaskType = _DEFAULT_TASK_TYPE
    framework: Literal["keras"] = "keras"
    version: int = 1
    graph: ArchitectureGraph = Field(default_factory=ArchitectureGraph)
    created_at: datetime
    updated_at: datetime


class ArchitectureSummary(BaseModel):
    """List-view projection. Omits `graph`, which is large and unused in lists."""

    id: str
    project_id: str | None = None
    name: str
    description: str | None = None
    task_type: TaskType = _DEFAULT_TASK_TYPE
    framework: Literal["keras"] = "keras"
    version: int
    node_count: int = 0
    created_at: datetime
    updated_at: datetime


class ArchitectureCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = None
    project_id: str | None = None
    task_type: TaskType = _DEFAULT_TASK_TYPE
    # Start from a starter graph instead of an empty canvas.
    template_id: str | None = None
    graph: ArchitectureGraph | None = None


class ArchitectureUpdate(BaseModel):
    """Save payload. `version` is the version the client last read.

    A mismatch means another session saved in between, and the service returns
    409 rather than silently overwriting.
    """

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = None
    task_type: TaskType | None = None
    graph: ArchitectureGraph | None = None
    version: int | None = None


class NodePortSpec(BaseModel):
    key: str
    label: str


class NodeSpec(BaseModel):
    """Palette definition for one node type.

    `params` reuses `AdvancedParameterSpec` so the node inspector is the same
    generic field renderer the training form already uses — a new node type
    gets its UI without frontend changes.
    """

    type: str
    name: str
    category: str
    description: str
    params: list[AdvancedParameterSpec] = Field(default_factory=list)
    inputs: list[NodePortSpec] = Field(default_factory=list)
    outputs: list[NodePortSpec] = Field(default_factory=list)
    min_inputs: int = 1
    # -1 means unbounded (merge nodes).
    max_inputs: int = 1
    # Task types this node is offered for; empty means every task type.
    task_types: list[TaskType] = Field(default_factory=list)


class ArchitectureIssue(BaseModel):
    severity: Literal["error", "warning"]
    message: str
    node_id: str | None = None
    edge_id: str | None = None


class ArchitectureValidation(BaseModel):
    """Result of the fast analytic pass. No TensorFlow involved."""

    ok: bool
    issues: list[ArchitectureIssue] = Field(default_factory=list)
    # Per-node output shape, excluding the batch dimension. `None` entries are
    # dimensions the analytic pass cannot resolve without building the model.
    node_shapes: dict[str, list[int | None] | None] = Field(default_factory=dict)
    total_params_estimate: int | None = None
    layer_count: int = 0


class ArchitectureTemplate(BaseModel):
    id: str
    name: str
    description: str
    task_type: TaskType
    graph: ArchitectureGraph


class ArchitectureCode(BaseModel):
    architecture_id: str
    filename: str
    code: str
    framework: Literal["keras", "torch"] = "keras"
