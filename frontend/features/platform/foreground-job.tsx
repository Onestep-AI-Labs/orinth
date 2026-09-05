"use client";

import { useEffect, useSyncExternalStore } from "react";
import { Check } from "lucide-react";

/**
 * The one thing the app is doing right now, and how far through it is.
 *
 * The shell raises a full-screen overlay for any in-flight mutation and used to
 * put the single word "Working" on it. That was already the weakest readout in
 * the app, and once the prep agent learned to count its own rows it became an
 * actively wrong one: the progress bar was rendered *behind* the overlay, so a
 * forty-thousand-row import showed a determinate bar nobody could see and a
 * blocking modal that said nothing.
 *
 * So a long mutation publishes what it knows here, and the overlay renders that
 * instead of the word. The channel is deliberately dumb — a label, a percentage,
 * a count, a stage list — with no notion of datasets or prep runs, because the
 * shell must not have to know what kind of work is being reported on. Whoever
 * is doing the work maps their own state onto this shape.
 *
 * A module-level store rather than a context: the publisher is a leaf component
 * deep inside a route and the consumer is the shell that renders it, so a
 * provider would have to wrap everything to carry a value upwards.
 */

export type ForegroundStageState = "done" | "active" | "pending";

export type ForegroundStage = {
  key: string;
  label: string;
  state: ForegroundStageState;
};

export type ForegroundJob = {
  /** One sentence, written for the person waiting. Carries counts where known. */
  label: string;
  /** 0..100, or null when the work cannot say — the bar goes indeterminate. */
  percent: number | null;
  /** "8,412 of 41,003" — a number that can be checked against the source. */
  count?: string | null;
  /** The ladder of stages, when the work has ordered ones. */
  stages?: ForegroundStage[];
};

let current: ForegroundJob | null = null;
const listeners = new Set<() => void>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Replace the published job. `null` hands the overlay back its plain spinner. */
export function setForegroundJob(job: ForegroundJob | null): void {
  current = job;
  for (const listener of listeners) listener();
}

export function useForegroundJob(): ForegroundJob | null {
  // The server snapshot is always null: nothing is in flight during SSR, and
  // returning a fresh object there would fail hydration.
  return useSyncExternalStore(subscribe, () => current, () => null);
}

/**
 * Publish `job` for as long as this component renders it.
 *
 * Serialized for the dependency because the caller rebuilds the object on every
 * tick; comparing the value rather than the reference is what keeps a progress
 * update from being an infinite render loop. Clearing on unmount matters more
 * than it looks: a surface that navigates away mid-run would otherwise leave the
 * overlay stuck on its last frame forever.
 */
export function usePublishForegroundJob(job: ForegroundJob | null): void {
  const serialized = job ? JSON.stringify(job) : "";
  useEffect(() => {
    setForegroundJob(serialized ? (JSON.parse(serialized) as ForegroundJob) : null);
  }, [serialized]);
  useEffect(() => () => setForegroundJob(null), []);
}

/**
 * One renderer, used by the overlay and by the in-page readouts.
 *
 * Two components drawing the same bar is how they drift, and these two are seen
 * in sequence — the overlay while you wait, the panel when you come back to the
 * dataset — so a difference between them reads as the run having changed.
 */
export function JobProgress({ job, className = "" }: { job: ForegroundJob; className?: string }) {
  const stages = job.stages ?? [];
  return (
    <div className={`prep-progress ${className}`} role="status" aria-live="polite">
      <div className="prep-progress-head">
        {/* The system's one spinner, not a second one: `.spinner` already
            carries the reduced-motion handling and the accent budget. */}
        <span className="spinner prep-progress-spinner" aria-hidden="true" />
        <p className="prep-progress-detail">{job.label}</p>
        {job.percent !== null && <span className="prep-progress-percent">{job.percent}%</span>}
      </div>

      {/* Determinate wherever the work gave a fraction, and indeterminate rather
          than absent when it did not — a run with no bar at all is the "Working"
          screen this replaced. */}
      <div
        className="prep-progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={job.percent ?? undefined}
      >
        <div
          className={`prep-progress-fill ${
            job.percent === null ? "prep-progress-fill-indeterminate" : ""
          }`}
          style={job.percent === null ? undefined : { width: `${job.percent}%` }}
        />
      </div>

      {job.count && <p className="prep-progress-count">{job.count}</p>}

      {stages.length > 0 && (
        <ol className="prep-progress-stages">
          {stages.map((stage) => (
            <li className={`prep-stage prep-stage-${stage.state}`} key={stage.key}>
              <span className="prep-stage-mark" aria-hidden="true">
                {stage.state === "done" ? <Check size={12} /> : null}
              </span>
              {stage.label}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
