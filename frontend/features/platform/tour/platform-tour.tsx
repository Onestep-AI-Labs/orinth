"use client";

import { useCallback, useEffect, useState } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import { HelpCircle } from "lucide-react";
import { Joyride, STATUS, type EventData, type Options, type Step, type Styles } from "react-joyride";
import { TOURS, tourIdForPath, type TourId } from "./tours";

/**
 * localStorage key holding the ids of tours the user has already seen. A tour
 * auto-plays at most once — after that it is launch-on-demand from the button.
 */
const SEEN_KEY = "onestep-tours-seen";

function loadSeen(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(SEEN_KEY);
    return Array.isArray(JSON.parse(raw ?? "[]")) ? JSON.parse(raw ?? "[]") : [];
  } catch {
    return [];
  }
}

function markSeen(id: string) {
  const next = new Set(loadSeen());
  next.add(id);
  window.localStorage.setItem(SEEN_KEY, JSON.stringify([...next]));
}

/**
 * Colors and geometry are read from the design tokens rather than hardcoded, so
 * the tour inherits the system's single accent and neutral ramp (DESIGN.md §2).
 * `oklch(var(--token))` resolves against :root inside Joyride's portal.
 */
const tourOptions: Partial<Options> = {
  primaryColor: "oklch(var(--accent))",
  backgroundColor: "oklch(var(--surface))",
  arrowColor: "oklch(var(--surface))",
  textColor: "oklch(var(--ink))",
  overlayColor: "oklch(var(--ink) / 0.45)",
  width: 320,
  zIndex: 980,
  showProgress: true,
  skipBeacon: true,
  spotlightPadding: 4,
  spotlightRadius: 8,
  buttons: ["back", "skip", "primary"],
  overlayClickAction: false
};

const tourStyles: Partial<Styles> = {
  tooltip: {
    borderRadius: "var(--radius-lg)",
    border: "1px solid oklch(var(--line))",
    fontFamily: "inherit",
    padding: 16
  },
  tooltipTitle: { fontSize: "0.875rem", fontWeight: 600, margin: 0 },
  tooltipContent: {
    fontSize: "0.8125rem",
    lineHeight: 1.45,
    color: "oklch(var(--ink-muted))",
    textAlign: "left",
    padding: "6px 0 2px"
  },
  tooltipFooter: { marginTop: 8 },
  buttonPrimary: { borderRadius: "var(--radius)", fontSize: "0.8125rem", fontWeight: 500, padding: "6px 12px" },
  buttonBack: { color: "oklch(var(--ink-muted))", fontSize: "0.8125rem" },
  buttonSkip: { color: "oklch(var(--ink-subtle))", fontSize: "0.8125rem" }
};

const tourLocale = { back: "Back", last: "Done", next: "Next", skip: "Skip" };

/**
 * Route-aware guided tours. Mounts once inside the platform shell and:
 *  - auto-plays the workspace tour on the user's first visit to `/projects`,
 *  - shows a floating "Tour" launcher on every surface that has a tour,
 *  - drives react-joyride, which renders in a portal above the app.
 *
 * Joyride touches `document`, so it is held back until after mount to keep the
 * server and first client render identical.
 */
/** Steps whose CSS-selector target is actually in the DOM right now. This keeps
 *  a tour usable when a surface renders a subset of its anchors — the Dataset
 *  Studio detail view vs. the catalog, the LLM vs. vision inference view, an
 *  import button that only appears for some projects. Steps with a non-string
 *  target (unused here) are always kept. */
function presentSteps(steps: Step[]): Step[] {
  return steps.filter((step) =>
    typeof step.target === "string" ? Boolean(document.querySelector(step.target)) : true
  );
}

export function PlatformTour() {
  const pathname = usePathname() ?? "";
  const searchParams = useSearchParams();
  // `/datasets` is the catalog until a dataset is open (`?dataset=…`), which is
  // a query-only change — hence useSearchParams, not just usePathname.
  const tourId = tourIdForPath(pathname, Boolean(searchParams?.get("dataset")));
  const [mounted, setMounted] = useState(false);
  const [activeId, setActiveId] = useState<TourId | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [run, setRun] = useState(false);
  // Remounts Joyride on each launch so a re-run of the same tour starts clean.
  const [launchNonce, setLaunchNonce] = useState(0);

  useEffect(() => setMounted(true), []);

  const start = useCallback((id: TourId) => {
    const resolved = presentSteps(TOURS[id].steps);
    // Nothing to point at (anchors not on this view) — don't open an empty tour.
    if (resolved.length === 0) return;
    setActiveId(id);
    setSteps(resolved);
    setLaunchNonce((value) => value + 1);
    setRun(true);
  }, []);

  // Changing surface (route or catalog↔studio) ends a running tour — its
  // anchors are gone.
  useEffect(() => {
    setRun(false);
    setActiveId(null);
    setSteps([]);
  }, [tourId]);

  // First-run: auto-play the workspace tour once, after the projects list has
  // had a moment to paint so its anchors exist. Marked seen immediately so a
  // quick navigation away does not make it reappear on the next visit.
  useEffect(() => {
    if (!mounted || tourId !== "projects") return;
    if (loadSeen().includes("projects")) return;
    const timer = window.setTimeout(() => {
      markSeen("projects");
      start("projects");
    }, 700);
    return () => window.clearTimeout(timer);
  }, [mounted, tourId, start]);

  const handleEvent = useCallback(
    (data: EventData) => {
      if (data.status === STATUS.FINISHED || data.status === STATUS.SKIPPED) {
        if (activeId) markSeen(activeId);
        setRun(false);
      }
    },
    [activeId]
  );

  return (
    <>
      {tourId && !run && (
        <button
          type="button"
          className="tour-launch"
          onClick={() => start(tourId)}
          title={`Take the ${TOURS[tourId].label.toLowerCase()}`}
          aria-label={`Take the ${TOURS[tourId].label.toLowerCase()}`}
        >
          <HelpCircle size={16} />
          <span>Tour</span>
        </button>
      )}
      {mounted && steps.length > 0 && (
        <Joyride
          key={`${activeId}-${launchNonce}`}
          run={run}
          steps={steps}
          continuous
          scrollToFirstStep
          options={tourOptions}
          styles={tourStyles}
          locale={tourLocale}
          onEvent={handleEvent}
        />
      )}
    </>
  );
}
