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

export function displayModelName(modelId: string, modelNameById: Record<string, string>): string {
  return modelNameById[modelId] ?? modelId;
}

export function formatSeconds(value: number): string {
  if (!value) return "0s";
  if (value < 60) return `${value.toFixed(1)}s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

export function pointsAttr(points: number[][]): string {
  return points.map((point) => `${point[0]},${point[1]}`).join(" ");
}

export function compactNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || Number.isInteger(value) ? 0 : 1)} ${units[unit]}`;
}

const LLM_FAMILIES = new Set(["llm_hf", "llm_adapter", "llm_gguf"]);

export function isLlmModelFamily(family: string): boolean {
  return LLM_FAMILIES.has(family);
}

const LABEL_RAMP_SIZE = 8;

/**
 * Categorical colour for a class index, from the `--label-*` data-visualization
 * ramp in globals.css. See DESIGN.md §8 — this ramp is the one sanctioned
 * exception to the single-accent budget.
 */
export function labelColor(index: number): string {
  return `var(--color-label-${index % LABEL_RAMP_SIZE})`;
}

/**
 * Translucent fill of the same colour, for annotation overlays drawn on top of
 * imagery. Must be used instead of appending a hex alpha suffix to
 * {@link labelColor} — that returns a `var()` reference, not a hex literal, so
 * string concatenation would yield an invalid colour and paint the overlay
 * opaque black over the image beneath it.
 */
export function labelFill(index: number, percent = 20): string {
  return `color-mix(in oklab, ${labelColor(index)} ${percent}%, transparent)`;
}

export function formatDatasetTask(task: TaskType): string {
  if (task === "object_detection") return "Object detection";
  if (task === "text_classification") return "Text classification";
  if (task === "question_answering") return "Question answering";
  if (task === "llm_finetune") return "LLM fine-tuning";
  return task.charAt(0).toUpperCase() + task.slice(1).replaceAll("_", " ");
}

export function formatDatasetFormat(format: string): string {
  if (format === "image_folder") return "Image folder";
  if (format === "text_folder") return "Text folder";
  if (format === "jsonl") return "JSONL";
  if (format === "instruction_jsonl") return "Instruction JSONL";
  if (format === "chat_jsonl") return "Chat JSONL";
  return format.toUpperCase();
}

export function taskDescription(task: TaskType): string {
  if (task === "classification") return "One class label per image";
  if (task === "object_detection") return "Boxes around objects";
  if (task === "segmentation") return "Pixel or polygon masks";
  if (task === "text_classification") return "One class label per text";
  if (task === "summarization") return "Source text with reference summary";
  if (task === "question_answering") return "Context with question-answer pairs";
  if (task === "llm_finetune") return "Instruction or chat SFT records";
  return "Task metadata";
}

export function isLlmTask(task: TaskType | string | undefined): boolean {
  return task === "llm_finetune";
}

export function areTasksCompatible(modelTask: TaskType, datasetTask: TaskType): boolean {
  if (modelTask === datasetTask) return true;
  const shapeTasks = new Set<TaskType>(["object_detection", "segmentation"]);
  return shapeTasks.has(modelTask) && shapeTasks.has(datasetTask);
}

export function isNlpTask(task: TaskType | string | undefined): boolean {
  return task === "text_classification" || task === "summarization" || task === "question_answering";
}

export function isVisionTask(task: TaskType | string | undefined): boolean {
  return !isNlpTask(task);
}

export const VISION_TASK_TYPES: TaskType[] = ["classification", "object_detection", "segmentation"];
export const NLP_TASK_TYPES: TaskType[] = ["text_classification", "summarization", "question_answering"];
export const LLM_TASK_TYPES: TaskType[] = ["llm_finetune"];
export const ALL_TASK_TYPES: TaskType[] = [...VISION_TASK_TYPES, ...NLP_TASK_TYPES, ...LLM_TASK_TYPES];

/**
 * Task types a project may work in — the shared source for the dataset,
 * training, testing, and inference task pickers.
 *
 * Returns `[]` while the project is still resolving. Callers must render a
 * loading state for that case instead of substituting a default: `useProject`
 * reports `null` until the projects list arrives, so falling back to vision
 * here flashes the wrong domain at an NLP project on every page load.
 */
export function allowedTaskTypesForProject(
  project: { task_types?: string[] } | null | undefined
): TaskType[] {
  if (!project) return [];
  const tasks = (project.task_types ?? []).filter((task): task is TaskType =>
    ALL_TASK_TYPES.includes(task as TaskType)
  );
  return tasks.length ? tasks : VISION_TASK_TYPES;
}

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
