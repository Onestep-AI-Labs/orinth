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
  if (task === "language_modeling") return "Language modeling";
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
  if (task === "language_modeling") return "Plain text corpus for next-token prediction";
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
  return (
    task === "text_classification" ||
    task === "summarization" ||
    task === "question_answering" ||
    // Text work, so the forms must not offer it an image size.
    task === "language_modeling"
  );
}

export function isVisionTask(task: TaskType | string | undefined): boolean {
  return !isNlpTask(task);
}

export const VISION_TASK_TYPES: TaskType[] = ["classification", "object_detection", "segmentation"];
export const NLP_TASK_TYPES: TaskType[] = [
  "text_classification",
  "summarization",
  "question_answering",
  // Phase 17: next-token prediction for architectures built in the studio.
  // Grouped with NLP because it is text work — `llm_finetune` is a different
  // job (adapting a pretrained base) and drives different form branches.
  "language_modeling"
];
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

// OSC (window title) and CSI (colour/cursor) sequences. Neither carries any
// visible text, so both are dropped before the cursor walk below.
const OSC_SEQUENCE = /\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)/g;
const CSI_SEQUENCE = /\u001b\[[0-9;?]*[ -/]*[@-~]/g;

/**
 * Render one line of captured terminal output the way a terminal would.
 *
 * Training runners stream raw TTY output. Keras draws its progress bar with
 * ANSI colour codes and then rewinds the cursor with a run of backspaces
 * (`\b`) so the next update overwrites the line in place. Printed verbatim in
 * the run log, those control bytes show up as a long trail of boxes after
 * every `loss:` value.
 *
 * Deleting the control characters is not enough — a backspace *moves* the
 * cursor, it does not erase — so this walks the line with a cursor and lets
 * later text overwrite earlier text. A trailing run of backspaces with nothing
 * written after it therefore leaves the visible text intact, which is exactly
 * the Keras case.
 */
export function normalizeTerminalOutput(line: string): string {
  const withoutAnsi = line.replace(OSC_SEQUENCE, "").replace(CSI_SEQUENCE, "");

  const buffer: string[] = [];
  let cursor = 0;
  for (const char of withoutAnsi) {
    if (char === "\b") {
      if (cursor > 0) cursor -= 1;
    } else if (char === "\r") {
      cursor = 0;
    } else if (char === "\n") {
      // Captured logs are already split per line; a stray newline is noise.
      continue;
    } else {
      buffer[cursor] = char;
      cursor += 1;
    }
  }

  // A hole is only reachable if a writer moved the cursor past the end, which
  // terminals do not do; fill defensively so no `undefined` leaks into the DOM.
  return Array.from(buffer, (char) => char ?? " ").join("").trimEnd();
}
