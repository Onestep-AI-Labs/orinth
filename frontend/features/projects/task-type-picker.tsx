"use client";

import { useState } from "react";
import { BarChart3, Database, FileImage, FileText, MessageSquareText, Shapes, Sparkles } from "lucide-react";
import { Badge } from "@/features/platform/ui";
import {
  ALL_TASK_TYPES,
  LLM_TASK_TYPES,
  NLP_TASK_TYPES,
  VISION_TASK_TYPES,
  formatDatasetTask,
  taskDescription
} from "@/features/platform/utils";
import type { TaskType } from "@/types/api";

export type ProjectDomain = "vision" | "nlp" | "llm";

/**
 * Which side a project leans to. Task types are the source of truth downstream —
 * this only records the lean, so a cross-domain project reads as "mixed". Stored
 * on `project.metadata.domain` by both the create and settings forms so the two
 * can never disagree about what a given task selection means.
 */
export function domainForTasks(tasks: TaskType[]): ProjectDomain | "mixed" {
  const present: ProjectDomain[] = [];
  if (tasks.some((task) => VISION_TASK_TYPES.includes(task))) present.push("vision");
  if (tasks.some((task) => NLP_TASK_TYPES.includes(task))) present.push("nlp");
  if (tasks.some((task) => LLM_TASK_TYPES.includes(task))) present.push("llm");
  if (present.length > 1) return "mixed";
  return present[0] ?? "vision";
}

/** The tab to open on for an existing selection. Mixed projects start on vision. */
export function initialDomainForTasks(tasks: TaskType[]): ProjectDomain | null {
  if (!tasks.length) return null;
  const domain = domainForTasks(tasks);
  return domain === "mixed" ? "vision" : domain;
}

function taskIcon(task: TaskType) {
  if (task === "classification") return <FileImage size={20} />;
  if (task === "object_detection") return <Database size={20} />;
  if (task === "segmentation") return <Shapes size={20} />;
  if (task === "text_classification") return <FileText size={20} />;
  if (task === "summarization") return <BarChart3 size={20} />;
  if (task === "question_answering") return <MessageSquareText size={20} />;
  if (task === "llm_finetune") return <Sparkles size={20} />;
  return <BarChart3 size={20} />;
}

export type TaskTypePickerProps = {
  value: TaskType[];
  onChange: (tasks: TaskType[]) => void;
  /**
   * Which domain tab opens first. `null` shows neither, forcing a deliberate
   * choice — that is what project creation wants. Settings passes the project's
   * existing domain instead, since the choice was already made.
   */
  initialDomain?: ProjectDomain | null;
  /** Reset the domain tab when this changes — e.g. when a new project loads. */
  resetKey?: string;
};

/**
 * Vision/NLP segmented control over a grid of task cards. Both tabs are always
 * reachable regardless of what the project currently declares: adding an NLP
 * task type to a vision-only project is the whole point of the settings form.
 */
export function TaskTypePicker({
  value,
  onChange,
  initialDomain = null,
  resetKey
}: TaskTypePickerProps) {
  const [domain, setDomain] = useState<ProjectDomain | null>(initialDomain);
  const [seenKey, setSeenKey] = useState(resetKey);

  // Reseed the open tab when the underlying record changes, without an effect —
  // rendering with the previous project's tab open for a frame is a visible flash.
  if (resetKey !== seenKey) {
    setSeenKey(resetKey);
    setDomain(initialDomain);
  }

  const domainTasks =
    domain === null
      ? []
      : domain === "vision"
        ? VISION_TASK_TYPES
        : domain === "nlp"
          ? NLP_TASK_TYPES
          : LLM_TASK_TYPES;
  const visionCount = value.filter((task) => VISION_TASK_TYPES.includes(task)).length;
  const nlpCount = value.filter((task) => NLP_TASK_TYPES.includes(task)).length;
  const llmCount = value.filter((task) => LLM_TASK_TYPES.includes(task)).length;
  const orderedSelection = ALL_TASK_TYPES.filter((task) => value.includes(task));

  function toggle(task: TaskType) {
    onChange(value.includes(task) ? value.filter((item) => item !== task) : [...value, task]);
  }

  return (
    <>
      <div className="segmented-control mb-3">
        <button
          className={domain === "vision" ? "segmented-active" : ""}
          type="button"
          onClick={() => setDomain("vision")}
        >
          Vision{visionCount > 0 ? <span className="segmented-count">{visionCount}</span> : null}
        </button>
        <button
          className={domain === "nlp" ? "segmented-active" : ""}
          type="button"
          onClick={() => setDomain("nlp")}
        >
          NLP{nlpCount > 0 ? <span className="segmented-count">{nlpCount}</span> : null}
        </button>
        <button
          className={domain === "llm" ? "segmented-active" : ""}
          type="button"
          onClick={() => setDomain("llm")}
        >
          LLM{llmCount > 0 ? <span className="segmented-count">{llmCount}</span> : null}
        </button>
      </div>
      {domain === null ? (
        <p className="hint-text">Choose Vision, NLP, or LLM to see the task types available.</p>
      ) : (
        <div className="task-choice-grid task-choice-grid-premium">
          {domainTasks.map((task) => {
            const active = value.includes(task);
            return (
              <button
                type="button"
                aria-pressed={active}
                className={`task-choice task-choice-premium task-choice-${task.replaceAll("_", "-")} ${
                  active ? "task-choice-active" : ""
                }`}
                key={task}
                onClick={() => toggle(task)}
              >
                <span className="task-choice-icon">{taskIcon(task)}</span>
                <span className="task-choice-copy">
                  <strong>{formatDatasetTask(task)}</strong>
                  <small>{taskDescription(task)}</small>
                  <em>{active ? "Included" : "Add type"}</em>
                </span>
              </button>
            );
          })}
        </div>
      )}
      {orderedSelection.length > 0 ? (
        <div className="task-selection-summary">
          <span className="task-selection-label">Included task types</span>
          <div className="task-selection-chips">
            {orderedSelection.map((task) => (
              <Badge key={task} tone="neutral">
                {formatDatasetTask(task)}
              </Badge>
            ))}
          </div>
        </div>
      ) : null}
    </>
  );
}
