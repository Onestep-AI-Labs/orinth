export type TaskType =
  | "classification"
  | "object_detection"
  | "segmentation"
  | "text"
  | "text_classification"
  | "summarization"
  | "question_answering";
export type DatasetFormat = "yolo" | "coco" | "image_folder" | "image_manifest" | "text_folder" | "jsonl" | "csv";
export type SplitKey = "unassigned" | "train" | "valid" | "test";
export type DatasetSplitFilter = "all" | SplitKey;

export type ProjectSummary = {
  id: string;
  name: string;
  description: string | null;
  task_types: TaskType[];
  metadata: Record<string, any>;
  created_at: string;
  updated_at: string;
};

export type ModelInfo = {
  id: string;
  name: string;
  family: string;
  description: string;
  available: boolean;
  paths: Record<string, string>;
  promoted: boolean;
  project_id: string;
  task_type: TaskType;
  labels: string[];
  source: "reference" | "trained" | "promoted";
  training_job_id: string | null;
  created_at: string | null;
  metrics: Record<string, any>;
  artifacts: Record<string, any>;
};

export type PlatformSettings = {
  huggingface_hub_token_configured: boolean;
};

export type PlatformSettingsUpdate = {
  huggingface_hub_token: string | null;
};

export type Detection = {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: { x: number; y: number; width: number; height: number };
  polygon: number[][];
  mask_area: number;
};

export type InferenceResult = {
  id: string;
  project_id: string;
  model_id: string;
  input_type: "image" | "text";
  image_level_label: string;
  detections: Detection[];
  class_scores: Record<string, number>;
  overlay_url: string | null;
  original_url: string | null;
  text_content: string | null;
  nlp_result: Record<string, any> | null;
  parameters: {
    confidence_threshold: number;
    iou_threshold: number;
    question: string | null;
    max_length: number;
  };
  created_at: string;
  duration_ms: number | null;
  timings: Record<string, number>;
};

export type JobProgress = {
  percent: number;
  processed: number;
  total: number | null;
  current_step: string;
  current_item: string | null;
  elapsed_seconds: number;
  eta_seconds: number | null;
  logs: string[];
  started_at: string | null;
  finished_at: string | null;
};

export type InferenceJob = {
  id: string;
  project_id: string;
  model_id: string;
  status: string;
  progress: JobProgress;
  result: InferenceResult | null;
  artifacts: Record<string, any>;
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type EvaluationDataset = {
  key: string;
  name: string;
  project_id: string;
  task_type: TaskType;
  format: DatasetFormat;
  split: string;
  available: boolean;
  path: string;
  labels: string[];
};

export type EvaluationJob = {
  id: string;
  project_id: string;
  comparison_id: string | null;
  model_id: string;
  dataset_key: string;
  status: string;
  limit: number | null;
  metrics: Record<string, any>;
  artifacts: Record<string, any>;
  progress: JobProgress;
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type EvaluationComparison = {
  comparison_id: string;
  project_id: string;
  dataset_key: string;
  jobs: EvaluationJob[];
  metrics: Array<Record<string, any>>;
};

export type EvaluationPerImageRow = {
  image: string;
  ground_truth: number;
  prediction: number;
  detections: number;
  objects: number;
  pixel: Record<string, number>;
  object: Record<string, number>;
  text_preview: string | null;
  reference_text: string | null;
  prediction_text: string | null;
  scores: Record<string, number>;
};

export type TrainingJob = {
  id: string;
  project_id: string;
  task_type: string;
  model_family: string;
  status: string;
  parameters: Record<string, any>;
  artifacts: Record<string, any>;
  artifact_urls: Record<string, string>;
  progress: JobProgress;
  metrics: Record<string, any>;
  history: Array<Record<string, any>>;
  curves: Record<string, any>;
  promoted_model_id: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type TrainingModelOption = {
  id: string;
  name: string;
  family: string;
  task_types: TaskType[];
  source: "local" | "ultralytics" | "keras_applications" | "huggingface";
  runnable: boolean;
  needs_download: boolean;
  description: string;
  defaults: Record<string, any>;
};

export type ModelAssetStatus = {
  option_id: string;
  status: "ready" | "missing" | "gated" | "failed";
  path: string | null;
  message: string | null;
};

export type DeleteResponse = {
  deleted: number;
  blocked: string[];
  missing: string[];
};

export type DatasetSplitSummary = {
  split: SplitKey;
  image_count: number;
  text_count: number;
  item_count: number;
  annotation_count: number;
};

export type DatasetSummary = {
  id: string;
  project_id: string;
  name: string;
  task_type: TaskType;
  format: DatasetFormat;
  source: "reference" | "editable";
  editable: boolean;
  path: string;
  labels: string[];
  classes: string[];
  splits: Record<string, DatasetSplitSummary>;
  metadata: Record<string, any>;
};

export type DatasetPreprocessConfig = {
  enabled: boolean;
  preset: "none" | "light" | "inspection" | "nlp_clean" | "nlp_augment";
  resize_width: number | null;
  resize_height: number | null;
  normalize: boolean;
  transforms: string[];
  augmentation_mode: "random" | "materialize";
  copies_per_image: number;
};

export type DatasetSplitConfig = {
  train: number;
  valid: number;
  test: number;
  seed: number;
  stratify: boolean;
  resplit_all: boolean;
};

export type DatasetProcessResponse = {
  dataset: DatasetSummary;
  moved: Record<string, number>;
  split_config: DatasetSplitConfig;
};

export type DatasetPreprocessPreview = {
  dataset_id: string;
  split: SplitKey;
  item_id: string;
  media_type: "image" | "text";
  image_url: string;
  text_preview: string | null;
  config: DatasetPreprocessConfig;
};

export type DatasetVersionSummary = {
  id: string;
  dataset_id: string;
  name: string;
  path: string;
  image_count: number;
  text_count: number;
  item_count: number;
  generated_count: number;
  splits: Record<string, DatasetSplitSummary>;
  config: DatasetPreprocessConfig;
  created_at: string;
};

export type DatasetEdaSummary = {
  dataset_id: string;
  split: DatasetSplitFilter;
  split_counts: Record<string, number>;
  class_counts: Record<string, number>;
  unlabeled_count: number;
  missing_annotation_count: number;
  image_count: number;
  text_count: number;
  item_count: number;
  annotation_count: number;
  image_size: Record<string, number | null>;
  aspect_ratio: Record<string, number | null>;
  text_length: Record<string, number | null>;
  warnings: string[];
};

export type DatasetAnnotation = {
  class_id: number;
  class_name: string;
  kind: "classification" | "box" | "polygon" | "summary" | "qa";
  bbox: { x: number; y: number; width: number; height: number } | null;
  polygon: number[][];
  text: string | null;
  question: string | null;
  answer: string | null;
};

export type DatasetItemSummary = {
  id: string;
  dataset_id: string;
  split: SplitKey;
  filename: string;
  media_type: "image" | "text";
  image_url: string;
  text_url: string | null;
  text_preview: string | null;
  width: number;
  height: number;
  annotation_count: number;
  classes: string[];
  class_id: number | null;
  label: string | null;
  is_labeled: boolean;
  annotations: DatasetAnnotation[];
};

export type DatasetItemPage = {
  items: DatasetItemSummary[];
  total: number;
  limit: number;
  offset: number;
};

export type DatasetItemDetail = DatasetItemSummary & {
  annotations: DatasetAnnotation[];
  text_content: string | null;
};
