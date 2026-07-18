# Spec: Phase 8 — Border-First Design System

## Status

Implemented. Supersedes the design-system portion of `phase-7-frontend-ui-revamp.md`.

## Goal

Replace the phase-7 teal/navy palette with a border-first, near-monochrome system built on a single blue accent, and retire the second typeface. Structure comes from 1px hairlines rather than shadow, which suits an information-dense operational dashboard better than the shadow-based elevation it replaces. Operational density is preserved — the first screen stays the working app.

Design source: a hallmark `study` DNA extraction of Dub, archived at `docs/design/dub-dna/`. **Only the system layer was adopted.** The source is a marketing-site extraction; its composition layer (heroes, feature pills, logo clouds, mockup frames, 30–48px display type, 16px body, 64px section gaps) is rejected by `docs/ai/rules.md` and enumerated as out-of-scope in `frontend/DESIGN.md` §1.

`frontend/DESIGN.md` remains the authoritative contract. `.claude/skills/frontend-design/SKILL.md` enforces it; `hallmark` is scoped to its `audit` / `redesign` verbs.

## Scope

In:

- Token set in `frontend/app/globals.css` rewritten. Neutrals go achromatic (chroma 0). Renames: `teal→accent`, `coral→danger`, `amber→warn`, `green→success`, `indigo→info`, `navy-deep→shadow-ink`; `navy` deleted.
- New tokens: `--line-input`, `--radius-xl`, `--ring-accent`, `--display-tracking`, `--shadow-subtle` (replacing `--shadow-card`), and the `--label-0..7` data-visualization ramp.
- Typography reduced to one face. Space Grotesk removed from `app/layout.tsx`; `--font-display` aliases `--font-sans` (Inter) and carries weight 500–600 plus `-0.02em` tracking. Weight cap lowered 700 → 600.
- Shell converted to light: navy sidebar becomes `--surface`, compact rail becomes `--canvas`, active nav becomes a soft `--accent-tint-strong` fill with `--accent-strong` text and no bar indicator. The alpha-modulated-white inverse-text stack is retired entirely.
- Border-first elevation across `frontend/app/styles/platform.css`: resting surfaces lose `box-shadow`; the only remaining elevations are `--shadow-subtle`, `--shadow-overlay`, and `--ring-accent`.
- `.create-project-hero` deleted — the one landing-page composition in the app. `/projects/new` now opens on `PageHeader` + form.
- `labelColor` returns a `var(--color-label-N)` reference; new `labelFill(index, percent)` uses `color-mix` for annotation overlays.
- Dead CSS removed: ~39 unreferenced classes including the entire `documentation-*` family.
- Documentation: `frontend/DESIGN.md` rewritten, `docs/ai/workflow.md` gains a Design step, `AGENTS.md`, `docs/ai/rules.md`, `.claude/commands/*`, this spec.

Out:

- Dark mode.
- Information-architecture or route changes — the phase-5 navigation model stays authoritative.
- A `Panel` primitive. `.panel` stays a raw class in ~26 call sites; wrapping them adds a zero-behavior boundary to a change set with no visual regression coverage.
- Backend changes (the project-delete fix shipped separately).

## Interfaces

- API endpoints: none added or changed.
- Frontend surfaces: all pages restyled; no component APIs changed except `@/features/platform/utils` gaining `labelFill` and `labelColor` changing return shape from hex literal to `var()` reference.
- Storage/DB changes: none.

## Data Flow

- Plain CSS reads resolved `--color-*` aliases; Tailwind reads raw channel triples through `oklch(var(--token) / <alpha-value>)`. Both are required — the triple form is what makes `<alpha-value>` work.
- `labelColor(i)` → `var(--color-label-i)` → `oklch(var(--label-i))`. `labelFill(i, p)` wraps that in `color-mix(in oklab, … p%, transparent)`.

## Edge Cases

- **`labelColor` no longer returns a hex literal.** Appending a hex alpha suffix (`` `${labelColor(i)}33` ``) now yields an invalid color that renders opaque black over the medical image beneath. Use `labelFill`. Covered by `features/platform/utils.test.ts`.
- **SVG presentation attributes do not resolve `var()`.** `stroke`/`fill` taking a token must be set via `style={{ … }}`. Three sites in `training-components.tsx` were converted.
- `color-mix` requires Chrome 111 / Safari 16.2 / Firefox 113 (all 2023); no explicit browser floor is configured, so Next's default applies.
- `--line` at 1.26:1 is below WCAG 1.4.11's 3:1 for control boundaries. It is fine on non-interactive containers (exempt) but form controls must use `--line-input` (4.74:1).
- `--ring-accent` at 20% alpha is ~1.5:1 and cannot be the sole selection signal; every rule using it also sets a full-opacity accent border.
- `app/global-error.tsx` renders without the CSS bundle and keeps inline hex synced to §2 by hand.
- Ten classes that look unreferenced are constructed at runtime (`toast-${tone}`, `task-choice-${task}`, `annotation-svg-${tool}`, `model-card-source-${source}`) and must not be deleted.

## Acceptance Criteria

- `grep -rnE '#[0-9a-fA-F]{3,8}|rgba?\(|hsla?\(' frontend/app/styles/platform.css frontend/features frontend/components` returns nothing; `app/global-error.tsx` is the sole documented exception.
- `grep -rn 'teal\|coral\|navy\|indigo\|amber\|Grotesk' frontend/app frontend/features frontend/components frontend/tailwind.config.ts` returns nothing.
- `grep -nE 'var\(--(ink|line|surface|accent|canvas|wash|danger|warn|success|info)[a-z0-9-]*\)' frontend/app/styles/platform.css | grep -v 'color-' | grep -v 'oklch('` returns nothing — catches raw channel triples used where a color is required.
- Every `box-shadow` in `platform.css` references `--shadow-subtle`, `--shadow-overlay`, or `--ring-accent`. No resting panel, card, or row has one.
- Accent appears only as primary action target / active nav / focus / progress; `--danger` only for destructive meaning; the `--label-*` ramp only in charts and overlays.
- Every status badge pairs `-tint` background with `-strong` text and clears 4.5:1.
- No type above 24px; no `font-weight: 700`.
- `pnpm typecheck`, `pnpm lint`, `pnpm test`, `pnpm build` pass; backend `uv run pytest` unaffected.

## Validation

Commands to run:

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm test
cd frontend && pnpm build
```

Manual: walk landing → create project → dataset upload/annotate/EDA/config → version → training (ROC chart) → testing → inference (vision **and** NLP paths) → settings → documentation.

**Highest-risk check: the dataset Annotate tab.** Draw a box and a polygon and confirm the translucent fills render as tinted overlays rather than opaque black — that is the `labelFill` failure mode.

Also exercise: sidebar collapse persistence, the 8 responsive breakpoints (`prefers-reduced-motion` included), an overlay set (dialog, option menu, rename popover, toast) to confirm shadows survived, and a forced mutation failure for the inline-error + toast path.

## Notes

There is no visual regression suite. The 11 class-name assertions in `features/platform/ui/primitives.test.tsx` and `features/datasets/detail-tabs.test.tsx` are the only automated guard, which is why this phase renamed **no CSS classes** — only token names and values. Preserve that constraint in future design work, or add visual coverage first.
