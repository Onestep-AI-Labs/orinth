import { TERMINAL_STATUSES } from "@/features/platform/constants";
import type { DatasetPreprocessConfig, DatasetSplitConfig, DatasetSummary, TaskType } from "@/types/api";

export function isActiveStatus(status: string): boolean {
  return !TERMINAL_STATUSES.has(status);
}

export function activePollInterval(job: { status: string } | undefined): number | false {
  return job && isActiveStatus(job.status) ? 2500 : false;
}

export function listPollInterval(jobs: Array<{ status: string }> | undefined): number | false {
  return jobs?.some((job) => isActiveStatus(job.status)) ? 3000 : false;
}

export function formatMetric(value: unknown): string {
  return typeof value === "number" ? value.toFixed(3) : value === undefined || value === null ? "-" : String(value);
}

export function formatSeconds(value: number): string {
  if (!value) return "0s";
  if (value < 60) return `${value.toFixed(1)}s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

export function pointsAttr(points: number[][]): string {
  return points.map((point) => `${point[0]},${point[1]}`).join(" ");
}

export function labelColor(index: number): string {
  const colors = ["#46b4a2", "#f47f73", "#6f7bd9", "#d59f31", "#8b5cf6", "#0ea5e9", "#84cc16", "#ec4899"];
  return colors[index % colors.length];
}

export function formatDatasetTask(task: TaskType): string {
  if (task === "object_detection") return "Object detection";
  if (task === "text_classification" || task === "text") return "Text classification";
  if (task === "question_answering") return "Question answering";
  return task.charAt(0).toUpperCase() + task.slice(1).replaceAll("_", " ");
}

export function formatDatasetFormat(format: string): string {
  if (format === "image_folder") return "Image folder";
  if (format === "text_folder") return "Text folder";
  if (format === "jsonl") return "JSONL";
  return format.toUpperCase();
}

export function taskDescription(task: TaskType): string {
  if (task === "classification") return "One class label per image";
  if (task === "object_detection") return "Boxes around objects";
  if (task === "segmentation") return "Pixel or polygon masks";
  if (task === "text_classification" || task === "text") return "One class label per text";
  if (task === "summarization") return "Source text with reference summary";
  if (task === "question_answering") return "Context with question-answer pairs";
  return "Task metadata";
}

export function areTasksCompatible(modelTask: TaskType, datasetTask: TaskType): boolean {
  if (modelTask === datasetTask) return true;
  const shapeTasks = new Set<TaskType>(["object_detection", "segmentation"]);
  return shapeTasks.has(modelTask) && shapeTasks.has(datasetTask);
}

export function isNlpTask(task: TaskType | string | undefined): boolean {
  return task === "text" || task === "text_classification" || task === "summarization" || task === "question_answering";
}

export function isVisionTask(task: TaskType | string | undefined): boolean {
  return !isNlpTask(task);
}

export const VISION_TASK_TYPES: TaskType[] = ["classification", "object_detection", "segmentation"];
export const NLP_TASK_TYPES: TaskType[] = ["text_classification", "summarization", "question_answering"];

export function defaultPreprocessConfig(): DatasetPreprocessConfig {
  return {
    enabled: false,
    preset: "none",
    resize_width: null,
    resize_height: null,
    normalize: false,
    transforms: [],
    augmentation_mode: "random",
    copies_per_image: 4
  };
}

export function preprocessFromDataset(dataset: DatasetSummary): DatasetPreprocessConfig {
  return {
    ...defaultPreprocessConfig(),
    ...((dataset.metadata?.preprocess as Partial<DatasetPreprocessConfig> | undefined) ?? {})
  };
}

export function defaultSplitConfig(): DatasetSplitConfig {
  return {
    train: 0.7,
    valid: 0.2,
    test: 0.1,
    seed: 42,
    stratify: true,
    resplit_all: false
  };
}

export function splitConfigFromDataset(dataset: DatasetSummary): DatasetSplitConfig {
  return {
    ...defaultSplitConfig(),
    ...((dataset.metadata?.split_config as Partial<DatasetSplitConfig> | undefined) ?? {})
  };
}
