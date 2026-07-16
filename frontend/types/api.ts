import type { components } from "@/types/generated/api";

// Hand-maintained primitives: these are bare Python `Literal[...]` type aliases in the
// backend (TaskType, DatasetFormat, SplitKey, DatasetSplitFilter). OpenAPI has no named
// component for a bare Literal alias -- it gets inlined per-field wherever it's used, so
// `openapi-typescript` never emits a standalone reusable named type for them. They must
// stay hand-maintained here and kept in sync with `backend/app/schemas.py`.
export type TaskType =
  | "classification"
  | "object_detection"
  | "segmentation"
  | "text_classification"
  | "summarization"
  | "question_answering";
export type DatasetFormat = "yolo" | "coco" | "image_folder" | "image_manifest" | "text_folder" | "jsonl" | "csv";
export type SplitKey = "unassigned" | "train" | "valid" | "test";
export type DatasetSplitFilter = "all" | SplitKey;

// Everything below is a thin facade over the generated OpenAPI types so that the
// generated schema (`@/types/generated/api`) remains the single source of truth, while
// most call sites can keep importing the familiar names from "@/types/api" unchanged.
export type ProjectSummary = components["schemas"]["ProjectSummary"];
export type ModelInfo = components["schemas"]["ModelInfo"];
export type Detection = components["schemas"]["Detection"];
export type InferenceResult = components["schemas"]["InferenceResult"];
export type JobProgress = components["schemas"]["JobProgress"];
export type InferenceJob = components["schemas"]["InferenceJobRead"];
export type EvaluationDataset = components["schemas"]["EvaluationDatasetInfo"];
export type EvaluationJob = components["schemas"]["EvaluationJobRead"];
export type EvaluationPerImageRow = components["schemas"]["EvaluationPerImageRow"];
export type TrainingJob = components["schemas"]["TrainingJobRead"];
export type DatasetSummary = components["schemas"]["DatasetSummary"];
export type DatasetPreprocessConfig = components["schemas"]["DatasetPreprocessConfig"];
export type DatasetSplitConfig = components["schemas"]["DatasetSplitConfig"];
export type DatasetVersionSummary = components["schemas"]["DatasetVersionSummary"];
export type DatasetEdaSummary = components["schemas"]["DatasetEdaSummary"];
export type DatasetAnnotation = components["schemas"]["DatasetAnnotation"];
export type DatasetItemSummary = components["schemas"]["DatasetItemSummary"];
export type DatasetItemPage = components["schemas"]["DatasetItemPage"];
export type DatasetItemDetail = components["schemas"]["DatasetItemDetail"];
