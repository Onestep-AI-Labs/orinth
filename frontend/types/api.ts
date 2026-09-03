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
  | "question_answering"
  | "llm_finetune"
  | "language_modeling";
export type DatasetFormat =
  | "yolo"
  | "coco"
  | "image_folder"
  | "image_manifest"
  | "text_folder"
  | "jsonl"
  | "csv"
  | "instruction_jsonl"
  | "chat_jsonl";
export type SplitKey = "unassigned" | "train" | "valid" | "test";
export type ReadinessState = "ready" | "needs_prep" | "needs_input" | "blocked";
export type ReadinessAction =
  | "upload"
  | "run_prep"
  | "label"
  | "split"
  | "wait"
  | "none";
export type PrepState =
  | "draft"
  | "detecting"
  | "planning"
  | "planned"
  | "applying"
  | "ready"
  | "failed"
  | "cancelled";
// The stage a run is *inside*, which is what the progress readout names. Finer
// than `PrepState`, which answers the different question of whether the dataset
// can be trained on yet.
export type PrepStep =
  | "idle"
  | "staging"
  | "detecting"
  | "planning"
  | "transforming"
  | "applying"
  | "splitting"
  | "done";
export type DatasetSplitFilter = "all" | SplitKey;

// Everything below is a thin facade over the generated OpenAPI types so that the
// generated schema (`@/types/generated/api`) remains the single source of truth, while
// most call sites can keep importing the familiar names from "@/types/api" unchanged.
export type ProjectSummary = components["schemas"]["ProjectSummary"];
export type ProjectStats = components["schemas"]["ProjectStats"];
export type ModelInfo = components["schemas"]["ModelInfo"];
export type Architecture = components["schemas"]["Architecture"];
export type ArchitectureSummary = components["schemas"]["ArchitectureSummary"];
export type ArchitectureGraph = components["schemas"]["ArchitectureGraph"];
export type ArchitectureNode = components["schemas"]["ArchitectureNode"];
export type ArchitectureEdge = components["schemas"]["ArchitectureEdge"];
export type ArchitectureIssue = components["schemas"]["ArchitectureIssue"];
export type ArchitectureValidation = components["schemas"]["ArchitectureValidation"];
export type ArchitectureTemplate = components["schemas"]["ArchitectureTemplate"];
export type NodeSpec = components["schemas"]["NodeSpec"];
export type Detection = components["schemas"]["Detection"];
export type InferenceResult = components["schemas"]["InferenceResult"];
export type JobProgress = components["schemas"]["JobProgress"];
export type InferenceJob = components["schemas"]["InferenceJobRead"];
export type EvaluationDataset = components["schemas"]["EvaluationDatasetInfo"];
export type EvaluationJob = components["schemas"]["EvaluationJobRead"];
export type EvaluationPerImageRow = components["schemas"]["EvaluationPerImageRow"];
export type TrainingJob = components["schemas"]["TrainingJobRead"];
export type TrainingModelOption = components["schemas"]["TrainingModelOption"];
export type AdvancedParameterSpec = components["schemas"]["AdvancedParameterSpec"];
export type LlmEnvironment = components["schemas"]["LlmEnvironment"];
export type DatasetSummary = components["schemas"]["DatasetSummary"];
export type DatasetPreprocessConfig = components["schemas"]["DatasetPreprocessConfig"];
export type DatasetSplitConfig = components["schemas"]["DatasetSplitConfig"];
export type DatasetVersionSummary = components["schemas"]["DatasetVersionSummary"];
export type DatasetEdaSummary = components["schemas"]["DatasetEdaSummary"];
export type DatasetReadiness = components["schemas"]["DatasetReadiness"];
export type DatasetReadinessCheck = components["schemas"]["DatasetReadinessCheck"];
export type DatasetPrepStatus = components["schemas"]["DatasetPrepStatus"];
export type DatasetTablePage = components["schemas"]["DatasetTablePage"];
export type DatasetHubFacets = components["schemas"]["DatasetHubFacets"];
export type DatasetHubFacetOption = components["schemas"]["DatasetHubFacetOption"];
export type DatasetTableColumn = components["schemas"]["DatasetTableColumn"];
export type DatasetTableRow = components["schemas"]["DatasetTableRow"];
export type DatasetDetection = components["schemas"]["DatasetDetection"];
export type DatasetPrepPlan = components["schemas"]["DatasetPrepPlan"];
export type DatasetPrepResponse = components["schemas"]["DatasetPrepResponse"];
export type PrepDecision = components["schemas"]["PrepDecision"];
export type PrepEngine = components["schemas"]["PrepEngine"];
export type PrepTransform = components["schemas"]["PrepTransform"];
export type DatasetAnnotation = components["schemas"]["DatasetAnnotation"];
export type DatasetItemSummary = components["schemas"]["DatasetItemSummary"];
export type DatasetItemPage = components["schemas"]["DatasetItemPage"];
export type DatasetItemDetail = components["schemas"]["DatasetItemDetail"];
export type DatasetHubSearchResult = components["schemas"]["DatasetHubSearchResult"];
export type DatasetHubSearchResponse = components["schemas"]["DatasetHubSearchResponse"];
export type DatasetHubPreview = components["schemas"]["DatasetHubPreview"];
export type DatasetHubColumnMapping = components["schemas"]["DatasetHubColumnMapping"];
export type DatasetHubImportRequest = components["schemas"]["DatasetHubImportRequest"];
export type DatasetHubImportResponse = components["schemas"]["DatasetHubImportResponse"];
export type DatasetOrigin = "created" | "imported_hf" | "recipe";

// Phase 11: data recipes.
export type RecipeRead = components["schemas"]["RecipeRead"];
export type RecipeSourceRead = components["schemas"]["RecipeSourceRead"];
export type RecipeGenerationSettings = components["schemas"]["RecipeGenerationSettings"];
export type RecipeGenerateRequest = components["schemas"]["RecipeGenerateRequest"];
export type RecipeRecord = components["schemas"]["RecipeRecord"];
export type RecipeRecordPage = components["schemas"]["RecipeRecordPage"];
export type RecipeCommitResponse = components["schemas"]["RecipeCommitResponse"];
export type OpenRouterModel = components["schemas"]["OpenRouterModel"];
export type OpenRouterModelsResponse = components["schemas"]["OpenRouterModelsResponse"];

// Phase 15: LLM export, serving, and chat.
export type ModelExportStatus = components["schemas"]["ModelExportStatus"];
export type ModelExportFormat = components["schemas"]["ModelExportRequest"]["format"];
export type ServingStatus = components["schemas"]["ServingStatus"];
export type ChatMessage = components["schemas"]["ChatMessage"];
export type RecipeOutputFormat = "instruction_jsonl" | "chat_jsonl";
export type RecipeStatus = "draft" | "extracting" | "generating" | "ready" | "failed";
export type RecipeGenerationMode = "auto" | "llm" | "rules";
export type RecipePromptFlavor = "qa" | "instruction" | "conversation";
