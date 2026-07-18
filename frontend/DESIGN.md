# Onestep AI Platform — Design System

AI-readable design contract for the frontend. Tokens live in `app/globals.css`; selectors that consume them live in `app/styles/platform.css`. New code uses primitives from `@/features/platform/ui`. Never hardcode a color value.

## 1. Identity & Voice

Operational research instrument for medical-imaging and NLP experiments — not a marketing site, not a diagnostic device.

- The first screen is the working app. No landing-page composition.
- Labels are nouns ("Training runs"). Buttons are verbs ("Start training").
- Dense over spacious; informative over decorative.
- Never imply final clinical diagnosis or autonomous medical decision-making.

### Provenance and scope of the visual system

The color, elevation, radius, and density layers derive from a design DNA extracted from Dub (`docs/design/dub-dna/`). **That source is a marketing-site extraction and only its system layer was adopted.** The following are out of scope by rule and must not be reintroduced:

| Rejected from the source | Why |
|---|---|
| Hero sections, feature pills, logo clouds, product-mockup frames | The first screen is the working app |
| Display type at 30 / 36 / 48px | Nothing in this app exceeds a 24px page title |
| 16px canonical body text | Dense operational UI; body is 13–14px |
| Conic-gradient brand mark, dotted-grid backgrounds | No decorative gradients on UI |
| 64px section gaps, 1200px centered max-width | Full-bleed dashboard with a fixed sidebar |

Two values deliberately deviate from the source; both are documented at their token below.

## 2. Color Tokens

Tokens are OKLCH channel triples (`L% C H`) consumed as `oklch(var(--token))` and, in Tailwind, `oklch(var(--token) / <alpha-value>)`. Neutrals are achromatic (chroma 0) on purpose — the ramp carries no hue.

| Token | OKLCH | Hex | Use |
|---|---|---|---|
| `--ink` | `20.46% 0 0` | `#171717` | Primary text, primary button fill |
| `--ink-muted` | `43.86% 0 0` | `#525252` | Secondary text |
| `--ink-subtle` | `52.78% 0 0` | `#6b6b6b` | Tertiary text, micro-labels |
| `--line` | `92.19% 0 0` | `#e5e5e5` | **The structural hairline.** Cards, panels, dividers |
| `--line-soft` | `94.01% 0 0` | `#ebebeb` | Table rules |
| `--line-strong` | `86.99% 0 0` | `#d4d4d4` | Hover edges, emphasis borders |
| `--line-input` | `55.55% 0 0` | `#737373` | Input/select/textarea borders only |
| `--wash` | `97.02% 0 0` | `#f5f5f5` | Hover/pressed wash |
| `--surface` | `100% 0 0` | `#ffffff` | Panels, cards, sidebar |
| `--surface-2` | `98.51% 0 0` | `#fafafa` | Nested/alt surfaces |
| `--canvas` | `97.02% 0 0` | `#f5f5f5` | App background, compact nav rail |
| `--accent` | `54.61% 0.2152 262.9` | `#2563eb` | THE accent — links, active state, focus, progress |
| `--accent-strong` | `42.44% 0.1809 265.6` | `#1e40af` | Accent text on tint |
| `--accent-tint` | `97.05% 0.0142 254.6` | `#eff6ff` | Faint accent wash |
| `--accent-tint-strong` | `93.22% 0.0328 256.8` | `#dbeaff` | Active nav fill |
| `--danger` | `57.71% 0.2152 27.3` | `#dc2626` | Danger accent |
| `--danger-strong` | `50.54% 0.1905 27.5` | `#b91c1c` | Danger text |
| `--danger-tint` | `93.56% 0.0309 17.7` | `#fee2e2` | Danger wash |
| `--warn` | `64.61% 0.1943 41.1` | `#ea580c` | Warning accent |
| `--warn-strong` | `55.34% 0.1739 38.4` | `#c2410c` | Warning text |
| `--warn-tint` | `95.42% 0.0372 75.2` | `#ffedd5` | Warning wash |
| `--success` | `62.71% 0.1699 149.2` | `#16a34a` | Success accent |
| `--success-strong` | `52.73% 0.1371 150.1` | `#15803d` | Success text |
| `--success-tint` | `96.24% 0.0434 156.7` | `#dcfce7` | Success wash |
| `--info-strong` | `49.07% 0.2412 292.6` | `#6d28d9` | Info chip text |
| `--info-tint` | `94.33% 0.0284 294.6` | `#ede9fe` | Info chip wash |
| `--shadow-ink` | `14.48% 0 0` | `#0a0a0a` | Shadow and scrim ink; primary-button hover |

**Accent budget law:** `--accent` is the single anchor hue and stays under ~5% of any screen — one primary action target, the active nav item, focus rings, progress. `--danger` appears only for destructive meaning. `--warn` / `--success` / `--info` appear only inside status badges and chips. Everything else is the ink/line/surface neutral ramp.

**Documented deviations from the source DNA:**

- **`--canvas` is `#f5f5f5`, not the source's pure white.** At 20+ simultaneous panels on `/datasets`, white-canvas-on-white-panels leaves the hairline as the only structural signal at exactly the density where the eye needs more. `#f5f5f5` is still the source's own Level-1 surface, so border-first is fully preserved.
- **`--line-input` is `#737373`, not the source's `#000000`.** That spec came from a single hero URL field; `/settings` renders 10–20 inputs at once and twenty black rectangles would out-weigh the near-black primary button, inverting the hierarchy. `#737373` keeps the "inputs read darker than containers" signature *and* is the only border weight here that clears WCAG 1.4.11 for a control boundary (4.74:1 vs `--line`'s 1.26:1).
- **`--ink-subtle` is `#6b6b6b`, darkened from the source's `#737373`.** At `#737373` on `--canvas` the ratio is 4.35:1 and fails AA; `#6b6b6b` gives 4.89:1.

### Data-visualization ramp — the one sanctioned exception

`--label-0` … `--label-7` (`#2563eb` `#ea580c` `#7c3aed` `#16a34a` `#dc2626` `#0891b2` `#a16207` `#db2777`) are a categorical ramp for annotation classes, EDA bars, and ROC curves. Charts and multi-class overlays need mutually distinguishable hues **by function**, so this ramp is exempt from the accent budget law. All eight clear 3:1 on `--surface`.

Consume them only through `labelColor(index)` / `labelFill(index)` in `@/features/platform/utils`. Never use them for UI chrome.

## 3. Typography

One typeface. Display is the body face at weight 500–600 with tightened tracking.

| Role | Face | Size / weight | Where |
|---|---|---|---|
| Display | Inter `--font-display` (aliases `--font-sans`) | 18–24px / 600, `letter-spacing: var(--display-tracking)` (`-0.02em`) | Page titles, brand wordmark, dialog titles, metric numerals |
| Body | Inter `--font-sans` | 14px / 400–500 | Default text, forms |
| Body dense | Inter | 13px / 400–500 | Tables, history lists |
| Panel title | Inter | 16px / 600 | Panel/section headings |
| Micro-label | Inter | 11px / 500–600, uppercase, `letter-spacing: 0.08em` | Eyebrows, column headers, badges |
| Code | `SFMono-Regular, Consolas, "Liberation Mono", monospace` | 13px | Logs, snippets |

**Weight cap is 600.** The source system's point is that headings are medium, not bold — 700 reads as shouting at this density. Metric numerals use `--font-display` with `font-variant-numeric: tabular-nums`.

**The largest type in this app is a 24px page title.** The source's 30 / 36 / 48px display steps are not used. At 24px the `-0.02em` tracking is what preserves the "confident, not shouting" quality, which is why the tracking token matters more here than the weight.

## 4. Spacing & Radius

- Spacing: multiples of 4 (`4 8 12 16 20 24 32 40 48`). No arbitrary values in new code.
- Radius vocabulary — do not invent values outside it:

| Token | Value | Applies to |
|---|---|---|
| `--radius-sm` | 6px | Inputs, selects, textareas |
| `--radius` | 8px | Buttons, nav links, tabs, small cards |
| `--radius-lg` | 12px | Cards, popovers |
| `--radius-xl` | 16px | Panels, dialogs — the large-card step |
| `--radius-pill` | 9999px | Badges, chips, pills |

## 5. Elevation — border-first

**Structure comes from 1px `--line` borders, never from shadow.** This is the single most load-bearing rule in the system and the one that most defines whether the UI reads as intended. Resting surfaces — panels, project/model/dataset cards, table rows — carry a border and no shadow. Cards signal interactivity by darkening the border to `--line-strong` on hover, not by lifting.

Exactly three elevation tokens exist:

| Token | Value | Used by |
|---|---|---|
| `--shadow-subtle` | `0 1px 2px oklch(var(--shadow-ink) / 0.05)` | Primary button, active segment, controls floating over imagery |
| `--shadow-overlay` | `0 10px 15px -3px … , 0 4px 6px -4px …` | Dialogs, menus, popovers, toasts, loading panel |
| `--ring-accent` | `0 0 0 2px oklch(var(--accent) / 0.2)` | Selection state |

`--ring-accent` at 20% alpha is ~1.5:1 and is **not sufficient as the only selection signal** — every rule using it must also set `border-color: var(--color-accent)` at full opacity.

**Never add `box-shadow` to a resting panel, card, or row.**

## 6. Shell Anatomy

Three columns separated by hairlines, no dark chrome:

- **Compact nav rail**: `--canvas`, hairline right border.
- **Full sidebar**: `--surface`, hairline right border, `--ink` wordmark, `--ink-muted` links.
- **Content**: `--canvas` background with white `--surface` panels.

Active nav item is a **soft chromatic fill** — `--accent-tint-strong` background with `--accent-strong` text (7.15:1). No left-border bar indicator.

There is no inverse-text context anywhere in the shell. If one is ever reintroduced, add a semantic "text on dark" token rather than alpha-modulating `--surface`.

- **Page header**: uppercase micro-label eyebrow above a 24px display title; actions right-aligned.

## 7. Components

Primitives live in `@/features/platform/ui`. Reach for a primitive before raw class names.

| Component | Variants | Notes |
|---|---|---|
| `Button` | `primary` · `secondary` · `ghost` · `danger` × `sm` `md` | `primary` is the **near-black `--ink` fill** with `--shadow-subtle`, not the accent — one per view region. `ButtonLink` is the same style over `next/link` |
| `IconButton` | `danger?` | Always has `aria-label` |
| `Badge` | tone: `neutral` `ok` `warn` `fail` `info` | Always the `-tint` background + `-strong` text pair; never the base hue (see §9) |
| `.panel` (class) | — | `--surface`, hairline `--line`, `--radius-xl`, **no shadow**. Applied as a raw class in ~26 places; a `Panel` primitive may be added later |
| `EmptyState` | with `action` | Every empty state names the next step and links to it |
| `StatusBadge` | job states | Maps run states to Badge tones |
| `ConfirmationDialog` | — | Display-face title, `--radius-xl`, `--shadow-overlay`; destructive confirm uses `danger` |
| `toast` | `success` `error` | Bottom-right, max 3, 6s, `aria-live` |
| Skeletons | `PageSkeleton` `CardGridSkeleton` `TableSkeleton` `InlineSpinner` | See loading contract |

## 8. Patterns

**Loading contract:** initial query → skeleton scoped to the data component; refetch of visible data → `InlineSpinner`; long mutation → global overlay (automatic via `useIsMutating`); route transition → segment `loading.tsx` renders `PageSkeleton`.

**Errors:** every mutation shows inline `MutationError` in its panel *and* an error toast (automatic via the mutation cache). Never swallow a failure.

**Confirmation:** destructive actions (delete project/dataset/model, cancel running job) require `ConfirmationDialog`.

**Journey continuity:** each stage points to the next — empty states and completion moments link forward (project → dataset → annotate → version → train → test → infer).

**Motion:** transitions use ease-out, 120–200ms; anything animated respects `prefers-reduced-motion`.

**Data visualization:** class colors come from `labelColor(index)`; translucent annotation overlays come from `labelFill(index, percent)`. **Never append a hex alpha suffix to `labelColor`** — it returns a `var()` reference, not a hex literal, so `` `${labelColor(i)}33` `` yields an invalid color and paints the overlay opaque black over the image beneath it. SVG `stroke`/`fill` must be set via `style={{ … }}`, not as presentation attributes — `var()` does not resolve in those.

## 9. Do / Don't

- **Do** add a token when a new color is genuinely needed; **don't** introduce a hex/rgb/hsl literal anywhere in styles or markup.
- **Do** keep one accent-colored element per region; **don't** put a second accent hue on a screen (the `--label-*` ramp is the sole exception, and only in charts and overlays).
- **Do** pair status colors as `-tint` background + `-strong` text. **Don't** use a base status hue as badge text — badges render at 11px, so 4.5:1 is mandatory and the base hues fail it (`--success` on `--success-tint` is 3.00:1; `-strong` gives 4.57:1).
- **Do** use `--line-input` on form controls; **don't** use `--line` there — at 1.26:1 it fails WCAG 1.4.11 for a control boundary.
- **Don't** use `--ink-subtle` for text below 18px on any surface darker than `--surface`.
- **Do** keep tables semantic (`<table>`) inside a horizontal-scroll wrapper; **don't** rebuild them as div grids.
- **Do** write empty states with a next-step action; **don't** render bare "No data".
- **Don't** add elevation to resting surfaces.
- **Don't** set type above 24px, or use weight 700.
- **Don't** use gradients, icon-tile card grids, decorative accent blobs, or centered hero layouts.
- **Don't** add marketing or diagnosis language.

## Exceptions

`app/global-error.tsx` is intentionally token-free (inline styles) — it must render when the CSS bundle fails. Its hardcoded values mirror the hex column in §2 and **must be re-synced by hand** whenever those tokens change. Nothing enforces this automatically.
