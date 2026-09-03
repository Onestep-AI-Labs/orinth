"use client";

import { Check } from "lucide-react";
import { prepIsRunning } from "@/features/datasets/prep/prep-hooks";
import type { DatasetPrepStatus, PrepStep } from "@/types/api";

/**
 * What the agent is doing right now, named rather than spun.
 *
 * The old readout was the word "Working…" for as long as the run took — up to
 * forty seconds on a real upload, during which the only honest reading of the
 * screen was "something might be broken". A run is four ordered stages and the
 * backend already knows which one it is in, so this shows the ladder with the
 * current rung lit and the backend's own sentence underneath it: "Read 312
 * files — looks like classification, checking".
 *
 * The sentence comes from the server (`prep.detail`) rather than being mapped
 * from the step here, because only the server knows the counts, and a count that
 * moves is the difference between progress and a hang.
 */

type Stage = { step: PrepStep; label: string };

//: The stages a run actually passes through, in order. `staging` is the upload
//: itself and `done` is the terminal marker; neither is a rung on the ladder.
//: `transforming` only happens for data that needs reshaping, so it is rendered
//: only once it has been reached — a stage nobody will visit is noise.
const STAGES: Stage[] = [
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

export function PrepProgress({ status }: { status: DatasetPrepStatus | null | undefined }) {
  if (!status) return null;
  const running = prepIsRunning(status);
  const step = (status.step ?? "idle") as PrepStep;
  if (!running && step !== "applying") return null;

  const current = rank(step);
  // A run that never needed the sandbox skips `transforming` entirely; showing
  // it as a permanently-pending rung would read as a stall.
  const stages = STAGES.filter(
    (stage) => stage.step !== "transforming" || current >= rank("transforming")
  );

  return (
    <div className="prep-progress" role="status" aria-live="polite">
      <div className="prep-progress-head">
        {/* The system's one spinner, not a second one: `.spinner` already
            carries the reduced-motion handling and the accent budget. */}
        <span className="spinner prep-progress-spinner" aria-hidden="true" />
        <p className="prep-progress-detail">{status.detail || "Orinth is preparing this dataset"}</p>
      </div>

      <ol className="prep-progress-stages">
        {stages.map((stage) => {
          const position = rank(stage.step);
          const state = position < current ? "done" : position === current ? "active" : "pending";
          return (
            <li className={`prep-stage prep-stage-${state}`} key={stage.step}>
              <span className="prep-stage-mark" aria-hidden="true">
                {state === "done" ? <Check size={12} /> : null}
              </span>
              {stage.label}
            </li>
          );
        })}
      </ol>

      {/* Determinate where the server supplied a fraction; the bar is otherwise
          omitted rather than faked, since a bar that does not track anything is
          worse than the stage list alone. */}
      {typeof status.progress === "number" && (
        <div className="prep-progress-track">
          <div
            className="prep-progress-fill"
            style={{ width: `${Math.round(Math.min(1, Math.max(0, status.progress)) * 100)}%` }}
          />
        </div>
      )}
    </div>
  );
}
