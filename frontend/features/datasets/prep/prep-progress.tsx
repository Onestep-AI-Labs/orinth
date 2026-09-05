"use client";

import { prepIsRunning } from "@/features/datasets/prep/prep-hooks";
import { JobProgress, type ForegroundJob } from "@/features/platform/foreground-job";
import type { DatasetPrepStatus, PrepStep } from "@/types/api";

/**
 * What the agent is doing right now, named and measured.
 *
 * The old readout was the word "Working…" for as long as the run took — up to
 * forty seconds on a real upload, during which the only honest reading of the
 * screen was "something might be broken". A run is a short ladder of stages and
 * the backend already knows which rung it is on, so this shows the ladder with
 * the current rung lit and the backend's own sentence underneath it.
 *
 * `prepJob` is the mapping and `JobProgress` is the rendering, split because the
 * shell's blocking overlay shows the same run: it publishes this job and draws
 * it there, so the bar is *in front of* the modal rather than behind it. See
 * `features/platform/foreground-job.tsx`.
 *
 * The bar is the part that had been missing. `progress` is the run's overall
 * fraction and `processed`/`total` are the units *inside* the current stage —
 * files copied, rows written, images downloaded — and both are shown, because
 * they answer different questions. A percentage says how far through; a pair of
 * counts says whether it is moving and lets the user check the denominator
 * against the folder they chose. Neither is inferred here: the server writes
 * both on every stage tick, since only it knows the counts.
 */

type Stage = { step: PrepStep; label: string };

//: The stages a run actually passes through, in order. `done` is the terminal
//: marker rather than a rung. `transforming` only happens for data that needs
//: reshaping, so it is rendered only once it has been reached — a stage nobody
//: will visit is noise.
const STAGES: Stage[] = [
  { step: "staging", label: "Take the files in" },
  { step: "detecting", label: "Read the files" },
  { step: "planning", label: "Work out the task" },
  { step: "transforming", label: "Reshape the rows" },
  { step: "applying", label: "Build the splits" }
];

const ORDER: PrepStep[] = ["staging", "detecting", "planning", "transforming", "applying", "done"];

function rank(step: PrepStep | undefined): number {
  const index = ORDER.indexOf(step ?? "idle");
  return index === -1 ? -1 : index;
}

function clamp(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/** The run as the shell's overlay and the in-page panel both render it. */
export function prepJob(status: DatasetPrepStatus | null | undefined): ForegroundJob | null {
  if (!status) return null;
  const step = (status.step ?? "idle") as PrepStep;
  if (!prepIsRunning(status) && step !== "applying") return null;

  const current = rank(step);
  // A run that never needed the sandbox skips `transforming` entirely; showing
  // it as a permanently-pending rung would read as a stall.
  const stages = STAGES.filter(
    (stage) => stage.step !== "transforming" || current >= rank("transforming")
  );

  const total = status.total ?? 0;
  const processed = status.processed ?? 0;
  // The bar tracks the whole run, not the current stage: `progress` is already
  // banded per stage on the server, so a stage finishing does not send it back
  // to zero. Where the server has no fraction at all, the stage's own
  // completion stands in rather than leaving the track empty.
  const fraction =
    typeof status.progress === "number"
      ? clamp(status.progress)
      : total > 0
        ? clamp(processed / total)
        : null;

  // The server's sentence already carries the numbers wherever it can ("Writing
  // record 8,412 of 41,003"), so the count line is only added when it would say
  // something the sentence does not — one fact printed twice reads as two.
  const count = total > 0 ? `${processed.toLocaleString()} of ${total.toLocaleString()}` : null;
  const detail = status.detail || "Orinth is preparing this dataset";

  return {
    label: detail,
    percent: fraction === null ? null : Math.round(fraction * 100),
    count: count && !detail.includes(count) ? count : null,
    stages: stages.map((stage) => {
      const position = rank(stage.step);
      return {
        key: stage.step,
        label: stage.label,
        state:
          position < current ? "done" : position === current ? "active" : ("pending" as const)
      };
    })
  };
}

export function PrepProgress({ status }: { status: DatasetPrepStatus | null | undefined }) {
  const job = prepJob(status);
  if (!job) return null;
  return <JobProgress job={job} />;
}
