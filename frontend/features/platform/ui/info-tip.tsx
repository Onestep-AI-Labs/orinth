"use client";

import { Info } from "lucide-react";
import type { ReactNode } from "react";

/**
 * The explanation for a term the user is not required to already know.
 *
 * The Hub browser is full of these — `parquet`, `webdataset`, `10K<n<100K`,
 * `sharegpt` — and a filter chip that says only "webdataset" is a quiz, not a
 * control. Every such term now carries its sentence, and the sentence comes
 * from the backend vocabulary (`services/hub_facets.py`) so the tooltip on a
 * filter chip and the tooltip on the badge showing the same term on a result
 * card cannot say different things.
 *
 * Deliberately native `title` rather than a positioned popover. Page chrome
 * outside a canvas keeps `title` per DESIGN.md §7 — the ~1s delay is a feature
 * here, since these labels are scanned far more often than they are questioned,
 * and an instant tooltip on a dense chip row is a flicker storm. It also means
 * the text is reachable by keyboard focus and by a screen reader without a
 * focus trap.
 */

export function InfoTip({ text, className }: { text: string; className?: string }) {
  return (
    <span className={`info-tip ${className ?? ""}`} title={text} tabIndex={0} role="note">
      <Info size={12} aria-hidden="true" />
      <span className="sr-only">{text}</span>
    </span>
  );
}

/**
 * A term plus its explanation, as one hoverable unit.
 *
 * Preferred over a bare `InfoTip` beside a label: at chip density a separate
 * icon target is a 12px hit area next to a 12px word, and hovering the word is
 * what people actually do.
 */
export function Explained({
  hint,
  children,
  className
}: {
  hint: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span className={className} title={hint}>
      {children}
    </span>
  );
}
