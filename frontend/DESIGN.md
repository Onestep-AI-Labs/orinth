# Onestep AI Platform — Design System

AI-readable design contract for the frontend. Tokens live in `app/globals.css`; selectors that consume them live in `app/styles/platform.css`. New code uses primitives from `@/features/platform/ui`. Never hardcode a color value.

## 1. Identity & Voice

Operational research instrument for medical-imaging and NLP experiments — not a marketing site, not a diagnostic device.

- The first screen is the working app. No landing-page composition.
- Labels are nouns ("Training runs"). Buttons are verbs ("Start training").
- Dense over spacious; informative over decorative.
- Never imply final clinical diagnosis or autonomous medical decision-making.

## 2. Color Tokens

Tokens are OKLCH channel triples (`L% C H`) consumed as `oklch(var(--token))` and, in Tailwind, `oklch(var(--token) / <alpha-value>)`.

| Token | OKLCH | Hex fallback | Use |
|---|---|---|---|
| `--ink` | `27.60% 0.0233 248.7` | `#1f2933` | Primary text |
| `--ink-muted` | `48.20% 0.0272 246.4` | `#52606d` | Secondary text |
| `--ink-subtle` | `56.46% 0.0352 248.4` | `#66788a` | Tertiary text, micro-labels |
| `--line` | `89.53% 0.0127 255.5` | `#d7dde5` | Borders, dividers |
| `--line-soft` | `92.68% 0.0063 255.5` | `#e4e7eb` | Soft borders, table rules |
| `--line-strong` | `86.04% 0.0124 248.0` | `#cbd2d9` | Strong borders, input outlines |
| `--wash` | `95.63% 0.0069 247.9` | `#edf1f5` | Hover/pressed wash on light surfaces |
| `--surface` | `100% 0 0` | `#ffffff` | Panels, cards |
| `--surface-2` | `98.42% 0.0034 247.9` | `#f8fafc` | Nested/alt surfaces |
| `--canvas` | `97.52% 0.0034 247.9` | `#f5f7f9` | App background |
| `--teal` | `70.20% 0.1034 180.5` | `#46b4a2` | THE accent — primary action, active nav, focus, progress |
| `--teal-strong` | `51.09% 0.0861 186.4` | `#0f766e` | Accent text/links on light |
| `--teal-tint` | `97.88% 0.0127 172.4` | `#f0fbf7` | Accent background wash |
| `--teal-tint-strong` | `95.67% 0.0268 176.3` | `#dff7f0` | Stronger accent wash |
| `--coral` | `72.57% 0.1455 27.3` | `#f47f73` | Danger accent |
| `--coral-strong` | `52.84% 0.1720 29.9` | `#ba3527` | Danger text |
| `--coral-tint` | `97.74% 0.0111 31.1` | `#fff5f3` | Danger background wash |
| `--amber` | `85.41% 0.1501 88.7` | `#f7c948` | Warning accent |
| `--amber-strong` | `53.10% 0.1190 65.1` | `#9a5b00` | Warning text |
| `--amber-tint` | `98.03% 0.0223 87.1` | `#fff8e8` | Warning wash |
| `--green` | `78.41% 0.1654 134.0` | `#8bcf5a` | Success accent |
| `--green-strong` | `52.59% 0.1378 137.4` | `#3f7c22` | Success text |
| `--green-tint` | `97.68% 0.0243 129.9` | `#f2fbea` | Success wash |
| `--navy` | `26.12% 0.0369 255.4` | `#182536` | Sidebar surface |
| `--navy-deep` | `21.55% 0.0338 256.4` | `#0f1a29` | Sidebar depth, rail |
| `--indigo-strong` | `49.83% 0.1197 274.4` | `#4f5ba6` | Info chip text |
| `--indigo-tint` | `96.19% 0.0179 272.3` | `#eef2ff` | Info chip wash |

**Accent budget law:** teal is the single anchor hue and stays under ~5% of any screen — one primary action, the active nav item, focus rings, progress. Coral appears only for danger/destructive meaning. Amber/green/indigo appear only inside status badges and chips. Everything else is the ink/line/surface neutral ramp.

## 3. Typography

| Role | Face | Size / weight | Where |
|---|---|---|---|
| Display | Space Grotesk `--font-display` | 20–24px / 600–700 | Page titles, brand wordmark, dialog titles, metric numerals |
| Body | Inter `--font-sans` | 14px / 400–500 | Default text, forms |
| Body dense | Inter | 13px / 400–500 | Tables, history lists |
| Panel title | Inter | 16px / 600 | Panel/section headings |
| Micro-label | Inter | 11px / 600, uppercase, `letter-spacing: 0.08em` | Eyebrows, column headers, card metadata |
| Code | `SFMono-Regular, Consolas, "Liberation Mono", monospace` | 13px | Logs, snippets |

Weight cap is **700**. Metric numerals use `--font-display` with `font-variant-numeric: tabular-nums`.

## 4. Spacing & Radius

- Spacing: multiples of 4 (`4 8 12 16 20 24 32 40 48`). No arbitrary values in new code.
- Radius tokens: `--radius-sm: 6px` (inputs, chips), `--radius: 8px` (buttons, small cards), `--radius-lg: 12px` (cards, panels, dialogs), `--radius-pill: 999px` (badges, pills).

## 5. Elevation

Two shadows only:

- `--shadow-card` — resting cards/panels: `0 1px 2px oklch(var(--navy-deep) / 0.06), 0 1px 3px oklch(var(--navy-deep) / 0.1)`
- `--shadow-overlay` — dialogs, toasts, menus: `0 12px 32px oklch(var(--navy-deep) / 0.28)`

## 6. Shell Anatomy

- **Sidebar** (the one asymmetric anchor): `--navy` surface, `--navy-deep` project rail, light text, teal active-item indicator, Space Grotesk wordmark. Collapsible per the phase-5 nav model.
- **Content**: `--canvas` background; white `--surface` panels with hairline `--line` borders.
- **Page header**: uppercase micro-label eyebrow (area/context) above a Space Grotesk title; actions right-aligned.

## 7. Components

Primitives live in `@/features/platform/ui`. Reach for a primitive before raw class names.

| Component | Variants | Notes |
|---|---|---|
| `Button` | `primary` (teal) · `secondary` (outline) · `ghost` · `danger` (coral) × `sm` `md` | One primary per view region; `ButtonLink` is the same style over `next/link` |
| `IconButton` | `danger?` | Always has `aria-label` |
| `Badge` | tone: `neutral` `ok` `warn` `fail` `info` | tint background + strong text pair from one hue family |
| `Panel` / cards | — | `--surface`, hairline `--line`, `--radius-lg`, `--shadow-card`, micro-label metadata row |
| `EmptyState` | with `action` | Every empty state names the next step and links to it |
| `StatusBadge` | job states | Maps run states to Badge tones |
| `ConfirmationDialog` | — | Display-face title, `--shadow-overlay`; destructive confirm uses `danger` |
| `toast` | `success` `error` | Bottom-right, max 3, 6s, `aria-live` |
| Skeletons | `PageSkeleton` `CardGridSkeleton` `TableSkeleton` `InlineSpinner` | See loading contract |

## 8. Patterns

**Loading contract** (extends phase-5):

- Initial query → skeleton scoped to the data component.
- Refetch of visible data → `InlineSpinner`.
- Long mutation → global overlay (automatic via `useIsMutating`).
- Route transition → segment `loading.tsx` renders `PageSkeleton`.

**Errors:** every mutation shows inline `MutationError` in its panel *and* an error toast (automatic via the mutation cache). Never swallow a failure.

**Confirmation:** destructive actions (delete project/dataset/model, cancel running job) require `ConfirmationDialog`.

**Journey continuity:** each stage points to the next — empty states and completion moments link forward (project → dataset → annotate → version → train → test → infer).

**Motion:** transitions use ease-out, 120–200ms; anything animated respects `prefers-reduced-motion`.

## 9. Do / Don't

- **Do** add a token when a new color is genuinely needed; **don't** introduce a hex/rgb/hsl literal anywhere in styles or markup.
- **Do** keep one teal-accented element per region; **don't** put a second accent hue on a screen.
- **Do** keep tables semantic (`<table>`) inside a horizontal-scroll wrapper; **don't** rebuild them as div grids.
- **Do** write empty states with a next-step action; **don't** render bare "No data".
- **Don't** use gradients, icon-tile card grids, or centered hero layouts.
- **Don't** use Space Grotesk at body sizes or below 16px.
- **Don't** add marketing or diagnosis language.

## Exceptions

`app/global-error.tsx` is intentionally token-free (inline styles) — it must render when the CSS bundle fails. Keep its hardcoded values in sync with the hex fallbacks above.
