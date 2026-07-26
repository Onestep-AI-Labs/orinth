# Spec: Guided Tours

## Status

Implemented

## Goal

Give new users an in-app orientation. Every main platform surface has a short,
interactive walkthrough that highlights its key controls in place, and the
workspace plays a one-time welcome walkthrough on first visit. Built on
[react-joyride](https://react-joyride.com/).

## Scope

In:

- A route-aware tour system mounted once in the platform shell.
- Per-surface tours for: Projects, Datasets (catalog), Dataset Studio (open
  dataset), Training, Testing, Inference, Chat, Models, Settings.
- Step filtering: at launch, steps whose anchor is absent from the DOM are
  dropped, so a tour stays usable when a surface renders a subset of anchors
  (catalog vs. studio, LLM vs. vision inference, conditional buttons).
- A first-run welcome tour that auto-plays once on `/projects`.
- A floating "Tour" launcher that starts the current surface's tour and hides
  while a tour runs.
- `data-tour="…"` anchors on stable structural elements in the shell and pages.

Out:

- Tours for detail views (`/training/[id]`, `/models/[id]`, `/testing/[id]`),
  the recipe builder, and project create/settings — no anchors, no launcher.
- Any backend or storage change. Seen-state is client-only.
- Auto-playing a tour on every page (only the welcome tour auto-plays).

## Interfaces

- API endpoints: none.
- Schemas: none.
- Frontend surfaces:
  - `frontend/features/platform/tour/tours.tsx` — tour definitions + `tourIdForPath`.
  - `frontend/features/platform/tour/platform-tour.tsx` — Joyride runner + launcher.
  - `frontend/features/platform/tour/index.ts` — barrel.
  - `frontend/components/app-shell.tsx` — mounts `<PlatformTour />`; nav/brand carry `data-tour`.
  - Page components carry `data-tour` anchors (datasets, training, testing,
    inference, chat, models, settings, projects).
  - `frontend/app/styles/platform.css` — `.tour-launch` styles (token-driven).
- Storage/DB changes: none. `localStorage["onestep-tours-seen"]` holds seen tour ids.

## Data Flow

`usePathname()` → `tourIdForPath` resolves the tour for the route → the launcher
starts it (or the welcome effect auto-starts `projects` once). react-joyride
renders steps in a portal, anchoring to `[data-tour]` selectors. On
finish/skip, the tour id is written to `localStorage` so it never auto-plays
again. A route change stops any running tour.

## Edge Cases

- **Missing anchor:** a step whose target is absent is waited on
  (`targetWaitTimeout`) then skipped; the launcher only offers tours whose route
  is mounted.
- **SSR:** Joyride touches `document`, so it renders only after mount to keep the
  server and first client render identical.
- **Replaying the same tour:** Joyride is remounted via a launch nonce key so a
  re-run starts from step one.
- **Colors:** all colors resolve from design tokens (`oklch(var(--…))`); no
  hardcoded values.

## Acceptance Criteria

- Welcome tour auto-plays once on first `/projects` visit; not again afterward.
- Each listed surface shows a working "Tour" button that walks its controls.
- `pnpm typecheck`, `pnpm lint`, `pnpm test`, and `pnpm build` pass.

## Validation

Commands to run:

```bash
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm test
cd frontend && pnpm build
```
