"use client";

import { Badge } from "@/features/platform/ui";
import type { BadgeProps } from "@/features/platform/ui";
import type { DatasetReadiness, ReadinessState } from "@/types/api";

/**
 * Whether a dataset can be trained on, as one pill.
 *
 * Rendered in three places — the catalog card, the studio header, and the
 * training page's dataset select — so the answer is identical wherever the user
 * happens to be looking. Before phase 21 it was nowhere: the training page
 * filtered on task type alone and then disabled its Start button without saying
 * why.
 */

type Presentation = { tone: NonNullable<BadgeProps["tone"]>; label: string };

/** Tones are the existing status pairs; no new tokens (DESIGN.md §2, §9). */
const PRESENTATION: Record<ReadinessState, Presentation> = {
  ready: { tone: "ok", label: "Ready to train" },
  // The agent can fix this one on its own, so it is a nudge, not a failure.
  needs_prep: { tone: "warn", label: "Needs prep" },
  // This one needs a person: labels, or more data.
  needs_input: { tone: "fail", label: "Needs input" },
  blocked: { tone: "info", label: "Preparing…" }
};

export function readinessPresentation(state: ReadinessState): Presentation {
  // A manifest written by a newer build could carry a state this one has never
  // heard of. Rendering nothing would be worse than rendering it neutrally.
  return PRESENTATION[state] ?? { tone: "neutral", label: "Unknown" };
}

export function ReadinessBadge({
  readiness,
  className
}: {
  readiness: DatasetReadiness | null | undefined;
  className?: string;
}) {
  if (!readiness) return null;
  const { tone, label } = readinessPresentation(readiness.state as ReadinessState);
  // Analysis does not change the verdict — that is the whole point of `busy`
  // being separate from `state` — so it is a note on the sentence rather than a
  // different pill. A dataset that trains still trains while Orinth re-reads it.
  const summary = readiness.busy
    ? `${readiness.summary} Orinth is looking at this dataset again.`
    : readiness.summary;
  return (
    // `title` carries the full sentence: the badge is one token wide and must
    // stay `white-space: nowrap` (DESIGN.md §8) or it steals width from the
    // card title beside it.
    <Badge tone={tone} className={className} title={summary}>
      {label}
    </Badge>
  );
}
