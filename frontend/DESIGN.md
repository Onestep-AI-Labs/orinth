# Orinth — Design System

AI-readable design contract for the frontend. Tokens live in `app/globals.css`; selectors that consume them live in `app/styles/platform.css`. New code uses primitives from `@/features/platform/ui`. Never hardcode a color value.

## 1. Identity & Voice

Operational research instrument for medical-imaging and NLP experiments — not a marketing site, not a diagnostic device.

- Every platform screen is the working app. No landing-page composition inside `app/(platform)`.
  The one public entry surface — `/`, the sign-in screen — is scoped in §10 and is the sole exception.
  There is no marketing landing page; `/` is auth.
- The brand mark and name are drawn by `@/components/brand`, never by an image tag or a
  retyped string. The mark is a stroked SVG on `currentColor`, so one asset serves every surface
  and both themes.
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

### Navigation glyphs

One glyph per destination, and **a destination's page header uses the same glyph as its nav entry** —
the icon is how you confirm you landed where the rail said you would. Two entries sharing a glyph is a
bug, not a style choice: it was how Projects ended up wearing Inference's eye and Datasets' database.

| Destination | Glyph | Why it reads |
|---|---|---|
| Projects | `Folder` | A project is a container of scoped work, not a data store |
| Datasets | `Database` | The stored corpus |
| Models | `Boxes` | Discrete artifacts in a registry |
| Training | `Activity` | A run producing a curve over time |
| Testing | `FlaskConical` | A measured experiment |
| Inference | `Rocket` | Putting a trained model to work — domain-neutral across vision, NLP, and LLM |
| Settings | `Settings` (global) / `Settings2` (project) | Two scopes, two glyphs |

Inference carries `Rocket` on every surface it owns: the nav entry (both rail widths), the page header,
and the Result panel and its empty state. `ScanEye` is retired — it read as vision-only on a menu that
also covers NLP and LLM chat. Lucide outline set only; never mix in a filled icon family.

## 7. Components

Primitives live in `@/features/platform/ui`. Reach for a primitive before raw class names.

| Component | Variants | Notes |
|---|---|---|
| `Button` | `primary` · `secondary` · `ghost` · `danger` × `sm` `md` | `primary` is the **near-black `--ink` fill** with `--shadow-subtle`, not the accent — one per view region. `ButtonLink` is the same style over `next/link` |
| `IconButton` | `danger?` | Always has `aria-label`. Square 40px, zero padding — the glyph is the whole target |
| `Badge` | tone: `neutral` `ok` `warn` `fail` `info` | Always the `-tint` background + `-strong` text pair; never the base hue (see §9) |
| `Select` | — | **The** dropdown. Every native `<select>` in the app renders through it, so all dropdowns share one height (40px), `--radius-sm`, `--line-input` border, and focus ring. Wrapping containers may set width, nothing else — a raw `<select>` used to inherit whichever of `.field select` / `.select-label select` / `.bulk-bar select` / `.project-switcher select` its container provided, and those disagreed on the border token |
| `.panel` (class) | — | `--surface`, hairline `--line`, `--radius-xl`, **no shadow**. Applied as a raw class in ~26 places; a `Panel` primitive may be added later |
| `EmptyState` | with `action` | Every empty state names the next step and links to it |
| `StatusBadge` | job states | Maps run states to Badge tones |
| `ConfirmationDialog` | — | Display-face title, `--radius-xl`, `--shadow-overlay`; destructive confirm uses `danger` |
| `toast` | `success` `error` | Bottom-right, max 3, 6s, `aria-live` |
| `Pager` | — | Previous / next over a paged list, with `n / m` between them. **Renders nothing for a single page** — a disabled pager under six cards is chrome asserting there is more. Numbered page links are deliberately absent: they only help when a page number means something, and on a category-ordered grid it does not |
| Skeletons | `PageSkeleton` `CardGridSkeleton` `TableSkeleton` `ListSkeleton` `InlineSpinner` | See loading contract. `ListSkeleton` is the rail/narrow-panel shape — two lines per row, hairline-separated; `TableSkeleton`'s three columns read as a broken table at 280px |
| `MultiSelect` / `TaskSelect` | — | Custom dropdowns: the closed trigger reads as a `<select>`, the open `--shadow-overlay` panel as a listbox. `MultiSelect` is checkbox multi-select; `TaskSelect` is single-select with Vision / NLP / LLM tabs + a search field (see §8) |
| `PlatformTour` / `.tour-launch` | — | Guided tours (react-joyride) in `@/features/platform/tour`. The launcher is a pill that **floats over the working app**, so it carries `--shadow-overlay` (a control over content, not a resting surface) with a `--line-strong` border and an accent icon. Tooltip colors resolve from tokens via `oklch(var(--…))`. Route-aware and mounted once in the shell; see `specs/phase-16-guided-tours.md` |

### Glyph scale — the control owns its icon size

**A lucide `size` prop passed at a call site is advisory; the control's CSS is authoritative.** Call sites across the app pass 12–18px for the same action, so the same button read at a different weight on each page and a 14px glyph inside a 40px target looked incidental beside its own label. `platform.css` overrides the `width`/`height` presentation attributes lucide emits, so every existing call site keeps working unchanged:

| Context | Glyph |
|---|---|
| `Button` / `ButtonLink` (40px, `md`) | 18px |
| `.button-sm` (34px) | 16px |
| `IconButton` (40px square) | 20px |
| `.arch-view-controls` / `.arch-drawer-head` / `.arch-card-actions` icon buttons (32–34px) | 18px |
| `.nav-link` | 19px |
| `.option-menu` row, `.project-back-link` | 17px |

An icon button sitting beside a `sm` button matches that button's height rather than standing 6px taller — the icon supports the labelled action, so it must not out-weigh it.

## 8. Patterns

**Loading contract:** initial query → skeleton scoped to the data component; refetch of visible data → `InlineSpinner`; long mutation → global overlay (automatic via `useIsMutating`); route transition → segment `loading.tsx` renders `PageSkeleton`. **An empty state is never a loading state** — showing "No datasets here yet" while the first request is still out asserts something nobody has checked. And a spinner over an expensive query is a design decision about *staleness*, not only about rendering: react-query's `staleTime` is per observer, so a panel that only needs a convenience list (the notebook dataset rail) asks for a long window rather than re-paying seconds of spinner on every visit.

**Long option lists:** a `.segmented-control` of tabs (each with a `.segmented-count`) plus a `Pager`, not a stack of headed sections. Tabs when the groups are *alternatives* the user picks between; sections when they are all meant to be read. The tab list is derived from the data's own order — never a hardcoded array, which is a second definition that goes stale the moment the data grows a category. Page size is whatever fills about two rows of the grid.

**Code fields:** anything that edits Python renders `features/platform/code/python-editor.tsx` — CodeMirror 6, themed from tokens, 4-space indent, bracket matching, completion, and a diagnostic margin. A `<textarea>` for code is not a smaller version of this; it is a different editor with different Tab semantics, and Python is whitespace-significant. Syntax colour rides the `--label-*` ramp (the sanctioned §2 functional-colour exception), so a keyword reads the same in a notebook cell and in an architecture custom layer.

**Errors:** every mutation shows inline `MutationError` in its panel *and* an error toast (automatic via the mutation cache). Never swallow a failure.

**Confirmation:** destructive actions (delete project/dataset/model, cancel running job) require `ConfirmationDialog`.

**Journey continuity:** each stage points to the next — empty states and completion moments link forward (project → dataset → annotate → version → train → test → infer).

**Motion:** transitions use ease-out, 120–200ms; anything animated respects `prefers-reduced-motion`.

**Data visualization:** class colors come from `labelColor(index)`; translucent annotation overlays come from `labelFill(index, percent)`. **Never append a hex alpha suffix to `labelColor`** — it returns a `var()` reference, not a hex literal, so `` `${labelColor(i)}33` `` yields an invalid color and paints the overlay opaque black over the image beneath it. SVG `stroke`/`fill` must be set via `style={{ … }}`, not as presentation attributes — `var()` does not resolve in those.

**Task selection (tabbed, searchable `TaskSelect`):** every operational task dropdown — training, testing, inference, and LLM serving — uses the shared `TaskSelect` primitive instead of a raw `<select>`. It takes the project's allowed task types and splits them across Vision / NLP / LLM **tabs** (the shared `.segmented-control`, shown only when the project spans more than one domain), so the grouping mirrors the project-creation `TaskTypePicker`. A sticky panel head pairs the tabs with a **search field** (`.task-select-search`, `Search` icon) that filters across *all* domains at once — while a query is active the results ignore the tab and regroup under uppercase domain micro-labels (`.task-select-group-label`), and Enter picks the first match. Each option is an icon + name + one-line description row; the active option carries the `--accent-tint` background with `--accent-strong` name and a `Check`, and the closed trigger shows the selection's icon, name, and a `.task-select-domain-tag`. Panel elevation is the sanctioned `--shadow-overlay`; close on click-outside/Escape matches `MultiSelect`. No new tokens, no shell change.

**Card titles in a grid column:** a card title is an identifier and is often long, so `.model-card-copy strong` clamps to **two** lines (`-webkit-line-clamp: 2` + `overflow-wrap: anywhere`) rather than ellipsing at one, and its description clamps to two below it — the name survives, the card heights stay even, and an unbroken family id has somewhere to break. This only holds if every ancestor between the clamp and its track carries `min-width: 0`: a link or button wrapping the title is a grid item whose automatic minimum size is the *un-wrapped* min-content width, so without it the title escapes its `minmax(0, 1fr)` column and runs under the status badge. `.badge` is `white-space: nowrap` for the same reason — a pill is one token, and a wrapping one steals width from the title beside it.

**Dataset catalog sections (phase 10):** provenance groups (Project / Imported from HuggingFace / Shared samples) are hairline-separated blocks with an uppercase micro-label header (`.section-microlabel`), rendered only when non-empty. No new tokens.

**LLM records (phase 10):** the `llm_finetune` Records tab is a semantic `<table>` in a `.table-wrap` (`overflow-x: auto`) container; the role-aware edit drawer and the hub import dialog are the only elevated surfaces, using the sanctioned `--shadow-overlay`. The hub detected-format banner uses the `success` pair, the manual-mapping hint the `info` pair, and the gated badge the `warn` pair — all existing status tokens.

**Data recipes (phase 11):** the recipes landing is a flat template gallery — `.recipe-template-card` cards are hairline-bordered `--surface` (blank card `--surface-2`, dashed), never Unsloth Studio's gradient/shine surfaces; concept badges are `neutral`. The workspace is a linear `.recipe-stepper` (Sources → Generate → Review → Commit) of pill steps; the active step borders `--accent`, done steps fill the index dot `--accent`. Record provenance badges use the `info`/`neutral` pair (`llm` vs `rules`); the run warnings panel uses the `warn` `-tint`/`-strong` pair; the review edit drawer reuses `.record-drawer` and its `--shadow-overlay`. No new tokens, no shell change.

**Custom model upload (phase 12):** the models page groups cards into hairline-separated `.model-section` blocks (Reference / Trained / Uploaded), each rendered only when non-empty, with a per-card source `Badge` — `neutral` reference, `info` trained/promoted, `ok` uploaded. The `.upload-dialog` is a sanctioned `--shadow-overlay` overlay reusing `.confirmation-overlay`; its family selector is a `.upload-family-chip` pill row (active chip borders `--accent` on `--accent-tint`), the sklearn security note uses the `danger` `-tint`/`-strong` pair, and LLM gate notes use the `warn` pair. The model detail route follows the eyebrow + 24px title `PageHeader` pattern with a `.model-detail-grid` metadata block and hairline `.model-artifact-list`; `llm_*` families show the phase-15 export and serving panels (see the phase-15 note) instead of inference/testing links. No new tokens, no shell change.

**Advanced training settings (phase 13):** the training form adopts a Configure anatomy — a full-width `.training-model-section` header (task · base model · name in a `form-grid-three` row, with the option description as a token-only `.form-caption`) over a `.training-config-grid` that pairs a narrow `.training-config-rail` (Dataset + Run panels stacked) with the wider Parameters panel, collapsing to one column below 1024px. The rail keeps the short decision cards from stranding a lone field beside the tall Parameters panel. This is a layout regrouping only; no field changes ownership between basic and advanced. The Dataset card renders an `EmptyState` with a "Go to datasets" `ButtonLink` when the active task has none; Run actions use the `Button` primitive (`secondary` Prepare + `primary` Start). The Parameters card hosts a collapsed-by-default `.advanced-panel` accordion (reusing the dataset studio's `.accordion-*` classes and `ChevronDown` rotation), hairline-separated from the basic fields above. Fields render generically from the catalog `AdvancedParameterSpec` list — the form has no per-family knowledge, so new families get their UI for free — grouped by `group` under uppercase `.advanced-group-label` micro-headers (Optimization, Augmentation, Regularization, Runtime). Numeric fields clamp to their `min`/`max` via `NumberInput`; the header shows an `.advanced-changed-count` when values differ from catalog defaults. The job detail page renders submitted advanced values as an "Advanced settings" `KeyValueTable`, so a finished run documents its own configuration. No new tokens, no shell change.

**LLM serving, export, and chat (phase 15):** the model detail page gains two hairline `.panel` blocks for `llm_*` families (in `features/models/llm-panels.tsx`). The **Export panel** (`.export-panel`, a 20px vertical-rhythm grid) reads top-to-bottom: a header with the section title and a width-capped `.export-intro` (78ch), a bordered `.export-config` block (`--surface-2`) grouping the format choice under an uppercase `.export-label` micro-header — a `.export-format-chip` pill row (active chip borders `--accent` on `--accent-tint`, matching the phase-12 upload chips), a one-line `.export-format-desc` for the selected format, then a quantization `<select>` (revealed only when the GGUF chip is active) beside the run `Button` — and a `.export-history` section with its own micro-label plus a "N running" `Badge`. Each `.export-row` is a hairline card with a spaced `.export-row-head` (format + status `Badge` + size on the left, download action right), and, while a job runs, an `.export-log` terminal block (dark `--shadow-ink` bg, mono 12px, `overflow-y: auto`) under a "Live output" label — matching the training log. The **Serving panel** pairs a serving-state `Badge` (`ok` running, `info` starting/stopping, `neutral` stopped) with start/stop `Button`s; non-GGUF families show an `EmptyState` "Export to GGUF first" CTA instead. The **chat surface** (`/inference/chat`, `features/inference/chat/chat-page.tsx`) is a `.chat-layout` two-column grid: a sticky `.chat-rail` (served-model status + a `--shadow-overlay` `.chat-model-picker-panel` popover over servable GGUF models, sampler `SliderField`s, system-prompt textarea) beside a flex `.chat-column` whose `.chat-transcript` scrolls above a pinned `.chat-footer-stats` (tokens/s + time-to-first-token) and `.chat-composer`. Transcript rows are hairline-separated `.chat-turn`s with uppercase `.chat-turn-role` micro-labels — no bubbles, no shadow; the streaming `.chat-caret` blinks 160ms ease-out and collapses to static opacity under `prefers-reduced-motion`. Both roles render GitHub-flavored markdown (`.chat-md`, token-styled headings/lists/tables/links) — including inside the `.chat-think` reasoning block, so code and headings in a model's reasoning are parsed rather than shown as literal `###`/```` ``` ````. Code blocks are syntax-highlighted (`react-syntax-highlighter` `PrismAsync` + `oneDark` — the async build auto-loads grammars so highlighting is actually colored while still code-splitting; the *Light* variant renders colorless without manual language registration) carrying a `.chat-code-head` language label and a `.chat-copy-btn` copy control, with a per-message copy button in `.chat-turn-head`. Collapsible `<think>…</think>` reasoning sits in a `.chat-think` disclosure (auto-expanded while streaming, collapsed once done). The composer carries a `.chat-composer-chips` row of `.chat-mode-chip` toggles (active = `--accent-tint`/`--accent-strong` border): **Search** (web search) and **Thinking** (show/hide reasoning) — the reference's Search/Code chip pattern. With Search on, an assistant turn shows a `.chat-citations` block (numbered `--accent` source links + host) above the answer. The served-model control uses the shared `ModelSourcePicker` (`.model-source-picker`, also used by the inference serve view) — a `serve-source-chip` tab toggle over Trained (registered GGUF with sizes), Local (persisted path auto-scanned into a selectable `.model-source-row` list + OS-native dialogs), and Hugging Face (a "Recommended for your hardware" list ranked by fit, then search) — so selection is identical across chat and inference. Options stay selectable while a model runs (starting a new one stops the old). The transcript, footer stats, composer, and chip row share a centered ~820px measure (`.chat-turn` / `.chat-footer-stats-inner` / `.chat-composer-inner` / `.chat-composer-chips`) so prose does not stretch the full panel width. Syntax-highlighter theme colors are the sanctioned functional-color exception (same basis as the docs viewer and the `--label` chart ramp), not chrome. Chat streams token-by-token over SSE through a dedicated `app/api/serving/chat/route.ts` streaming proxy (the config `/api/*` rewrite buffers streaming responses; a route handler returning the upstream `ReadableStream` does not). Model loading keeps the full-screen `.global-loading-overlay` but swaps the circular spinner for an indeterminate `.global-loading-bar` progress bar ("Loading model…") when a `serving-start`-keyed mutation is active; every other mutation keeps the circular spinner. The send button is the default `--ink` primary; `--accent` stays on focus rings only. Serving errors and mid-stream failures surface both in the transcript and the toast layer. Medical-context framing carries through (research instrument, no diagnosis language, no medical-assistant system-prompt preset). The **inference landing** branches on task: for `llm_finetune` it drops the submit-and-render form and renders `LlmServeView` (`features/inference/llm-serve-view.tsx`) — a **single** Serve panel (no Chat panel) with a three-way source toggle (`.serve-source-chip`): **Trained models** (registered `llm_gguf`, each with its file size), **Local folder**, and **Hugging Face**. Starting a server from any tab **auto-redirects to `/inference/chat`** (start returns only when the server is ready); the old "Open chat" launcher panel is gone, with a small Open-chat link kept only while a session is already running (beside Stop). Local folder uses the **OS-native file dialog** (`.serve-browse-actions` — Browse folder / Browse .gguf buttons calling `/serving/pick`, stdlib `osascript`/`zenity`, no third-party lib) plus a path field that **auto-scans** (debounced, no Scan button) and is **persisted** (restored + rescanned on load via `/serving/config`); results list GGUF files with sizes. Hugging Face (`.serve-hub`) is a search field + GGUF/MLX format select over a `.serve-hub-list` of hairline `.serve-hub-repo` rows (downloads/likes), each expanding to its files (quant + size) with a Download action; GGUF downloads then serve, MLX is download-only. When no GGUF exists the Trained-models tab shows an `EmptyState` linking to the models page. The header's "Chat with a served LLM" link is conditional — LLM task with at least one servable model only. The **testing page** gains an `llm` evaluation display kind alongside the existing task kinds — perplexity / token-accuracy / loss metric columns and a per-record table (prompt · reference · perplexity · loss) — reusing the same `testing-components.tsx` column/detail machinery as every other kind; GGUF models are omitted from the testing model picker (tested via chat, not batch eval). No new tokens, no shell change.

**LLM fine-tuning (phase 14):** the LLM training form reuses the phase-13 Configure grid unchanged. The environment banner (`.llm-env-banner`, device + recommended backend + notes) uses the `info` `-tint`/`-strong` pair; gated hub options show a `warn` `Badge` with a lock icon and a "license required" suffix in the select, whose options group into Hub models / Local base models `<optgroup>`s. Selecting "Custom Hugging Face model…" reveals a hub-id `Field`; a token-only `.form-caption` names the trainable base format (HF Transformers) and calls out GGUF/TFLite as non-trainable. The LLM 4-bit/precision advanced defaults auto-detect from the probed device (off/fp32 on MPS/CPU); the vision Device control is an auto-detected `<select>` (Auto + detected accelerator + CPU), not free text. A `.model-detail-line` caption shows live Hugging Face API detail (real download size, file count, downloads, a `warn`-toned "license required" marker, or a `danger`-toned error) for the selected base; the gated badge is driven by this live value. During a run the base-model download streams progress lines into the log tail instead of going silent. The fine-tuning method (LoRA / QLoRA / Full / Continued pretraining) is a prominent basic `<select>` in the Parameters panel with device-aware inline hints, not buried in the accordion (it is filtered out of the generic accordion render). The advanced accordion gains Method / LoRA / Quantization / Sequence group micro-headers ahead of the phase-13 four (backend constants and frontend `GROUP_ORDER` mirror each other). Creating an LLM job navigates straight to the training detail page — the platform's equivalent of Studio's Current Run switch. On the detail page, live step-denominated loss curves reuse the `--label` ramp `MiniLineChart`s, and completed runs render `sample_generations` as `.generation-card`s — hairline `--surface` cards with uppercase micro-labels and the mono stack at 13px. No new tokens, no shell change.

**Architecture studio (phase 17):** the Models section splits into a `Catalog | Architectures` `.models-tabs` segmented control (the shared `.segmented-control`, with `<Link>`s rather than buttons so each face is a real route). The studio at `/models/architectures/[id]` gives the whole row to `.arch-canvas`, over a `.arch-drawer` that toggles between Issues and the generated Python. Nothing else holds a column: the searchable `.arch-palette` is the body of an `.arch-palette-drawer` overlaid on the canvas's left edge (opened by the canvas `+`, a right-click, or a wire's insert button; `--shadow-overlay`, 160ms slide-in with a `prefers-reduced-motion` alternative), and `.arch-inspector` is the body of an `.arch-modal` dialog opened by double-clicking the node or frame it edits. Both close on `Escape`, outermost first.

React Flow ships only its structural `base.css`; **every visible surface is restyled from platform tokens**, so no vendor theme enters the app. `.arch-node` is border-first like any other resting card — hairline `--line`, no shadow, `--line-strong` on hover. Selection sets `border-color: var(--color-accent)` *and* `--ring-accent` (the ring alone is ~1.5:1, per §5); an error swaps the border to `--danger` and a warning to `--warn`, each paired with a matching `AlertTriangle` in the `-strong` text tone. Each node shows a `--font-mono` output shape — the fact the canvas exists to surface. The floating canvas `Controls` and `MiniMap` are controls over content, so they carry the sanctioned `--shadow-overlay`; the `MiniMap` earns its corner only while the viewport is moving, so it fades in on pan or zoom and back out 1400ms after the canvas settles, dropping `pointer-events` with its opacity so an invisible panel never swallows a press meant for the graph; the dot `Background` is a spatial affordance for a pannable canvas, not the decorative dotted-grid the source DNA rejects, and it draws in `--line-strong`.

The node inspector renders `NodeSpec.params` through the phase-13 `AdvancedField`, so a new node type declared on the backend gets its settings UI with no frontend change — the same catalog-driven mechanism as the training form's advanced accordion — and it renders identically in the rail's old chrome-free dialog body, because `.arch-modal .arch-inspector` strips the panel border and radius rather than the component branching on where it is. Canvas real estate is the scarce resource, so nothing that is not the canvas is on screen by default: the `.arch-view-controls` icon group keeps a settings opener (disabled with nothing selected) and a fullscreen toggle (`Escape` to leave) that lifts `.arch-studio-fullscreen` out of the shell with `position: fixed`, and the drawer still collapses on its own. Canvas navigation is trackpad-first: two-finger scroll pans both axes, pinch zooms, wheel-zoom is off, and drag-on-empty always marquee-selects — there is no select/move tool pair and so no tool cursor, only `.arch-canvas-pan`'s grab grip while `Space` is held. A hovered connection reveals an `.arch-edge-actions` pill at its midpoint (insert a node · remove the wire), which lives in React Flow's edge-label portal and so is driven by hover *state* rather than a CSS `:hover` that cannot reach across the portal boundary. A handle with nothing wired to it grows an `.arch-node-stub` — a dashed, faint lead line standing for "no connection here", ending in a solid `--line-input` button that reads as the control it is. Both live in an `.arch-node-shell`, an unclipped wrapper holding them as *siblings* of `.arch-node`: the node itself sets `overflow: hidden` to keep its header tint and category stripe inside the rounded corners, so anything positioned past its edge is clipped out of existence. Icon-only canvas controls (the pill, `.arch-tool`, the zoom stack, the stubs) carry hover text through `data-tip` + `.arch-tip`, an instant `--ink`-on-`--surface` label; page chrome outside the canvas keeps native `title`, where the ~1s delay is a feature. Placement is per-context: default below, `.arch-tip-up` for the wire pill (below lands on whatever the wire runs into), `.arch-tip-right` for the bottom-left zoom stack, and left-edge-aligned inside `.arch-canvas-tools` because the canvas clips overflow and a centred tip on the first button was cut in half. **`.arch-tip` must be declared above any rule that positions a tipped element** — it sets `position: relative`, and as a same-specificity single-class selector it silently won source order over `.arch-node-stub`'s `position: absolute`. Two containers give up `overflow: hidden` so those tips can escape — `.arch-tool-group` and `.react-flow__controls` — and round their end buttons' corners instead. Zoom uses lucide magnifiers rather than React Flow's stock `+`/`−`, which would collide with the `+` that adds a node; base.css styles those buttons for its own *filled* icon set (`fill: currentColor`, `width: 100%`, a 12px cap), and all four declarations are wrong for a lucide outline glyph — `fill` most of all, since lucide's `fill="none"` is a presentation attribute that any CSS rule outranks, so the magnifier lens filled solid and swallowed the `+`/`−` inside it. `.arch-canvas .react-flow__controls-button > svg` restores `fill: none` / `stroke: currentcolor` and drops the size caps; the button itself sets `color`, not `fill`. Issue rows use the `danger`/`warn` `-tint`/`-strong` pairs and carry a mono node-id chip that selects the offending node. A `code`-typed `AdvancedParameterSpec` renders as `.code-editor`: a transparent textarea stacked exactly over a syntax-highlighted `<pre>` sharing one font metric, with a line-number gutter that flags lint findings and a findings list beneath in the `danger`/`warn` pairs. Node categories carry an icon and a colour drawn from the `--label-0..7` ramp — category is categorical data being encoded by colour, the sanctioned §2 exception — confined to the node's 3px stripe and the palette dot, never chrome. Both remaining pane edges — the overlay panel's width and the drawer's height — are `.arch-resize` separators: draggable by pointer, adjustable by arrow keys, persisted per pane in `localStorage`. The generated-code panel is read-only (`react-syntax-highlighter` `PrismAsync` + `oneLight`, the sanctioned functional-color exception) with Copy and Download — the graph is the source of truth, so editing code in place would create a second definition that diverges from what trains. No new tokens, no shell change.

**Dataset Studio restructure (phase 21):** the four peer tabs (Images / Annotate / EDA / Config) become three ordered ones — **Overview · Data · Prepare** — because the old set presented a pipeline as unordered siblings and buried the action that makes a dataset trainable inside a collapsed accordion on the fourth of them. Overview leads: it answers the question the studio could not previously answer at all, which is whether the data is usable yet. Its `.detail-tab-dot` (6px, `--color-success-strong`, inverting to `--color-surface` on the active tab) marks the *dataset* as trainable, never the tab as visited — "you have looked at this" is not information. Data folds the old Annotate tab underneath the item grid rather than beside it, since both showed the same items and the second copy of the grid was the only difference; Prepare pairs the old EDA panel above the preprocessing and split controls, which were always one question asked on two screens. Every panel is the existing component moved, not rewritten, so the merge carries no behavior change. The catalog leads with `.ingest-card`, whose `.ingest-drop` is the one dashed border in the system (`--color-line-strong`, `--color-surface-2`, going `--color-accent` on `--color-accent-tint` while dragging) — dashed reads as "put something here" where the hairline `--line` of a resting panel reads as "this is a region"; it is a 160ms ease-out transition with a `prefers-reduced-motion` alternative, per §8. `DatasetCreatePanel` survives below it as the by-hand path. Agent findings render as a hairline-separated `.prep-decision` list: an uppercase micro-label field name, a provenance `Badge` (`info` for a model, `neutral` for a rule — the phase-11 record-provenance pair), the value, a plain-sentence rationale, and `.prep-decision-evidence` in `--font-mono` because it quotes the data itself (file counts, folder names, column names) and reads as a citation rather than prose. `.prep-notice` uses the `info` `-tint`/`-strong` pair, `.prep-needs-input` and `.prep-warning-list` the `warn` pair, and `.prep-check` icons the `success`/`danger` `-strong` tones. No new tokens, no shell change, no new route — the studio stays at `/datasets?dataset=`.

**Dataset readiness (phase 21):** whether a dataset can be trained on is one `Badge` (`ReadinessBadge`, `features/datasets/readiness-badge.tsx`) rendered in exactly three places — the catalog card's `.dataset-card-summary`, the studio `.dataset-header-actions`, and the training page's Dataset field — so the answer never differs by where the user is standing. Tones are the existing status pairs, no new tokens: `ok` "Ready to train", `warn` "Needs prep" (the agent can fix it alone), `fail` "Needs input" (a person must add labels or data), `info` "Preparing…" (a job is mid-rewrite). The `warn`/`fail` split is the whole point of the badge — it separates a nudge from a blocker — so the two must never collapse to one tone. The full reason rides in `title` rather than a second line, because `.badge` is `white-space: nowrap` per §8 and a wrapping pill steals width from the card title beside it; an unrecognized state (a manifest written by a newer build) falls back to `neutral`/"Unknown" instead of blanking the catalog. On the training page the badge is joined by a token-only `.form-caption` carrying `readiness.summary` and a `Prepare it` `ButtonLink` to the dataset — the §8 journey-continuity pattern — and non-ready datasets stay **listed and selectable** rather than filtered out, since hiding them is what produced "No dataset for this task" while three sat on disk. The Start button's four previously-silent disabled conditions collapse into one `.form-caption` sentence naming the next thing to do. No new tokens, no shell change.

**One way to start a dataset (phase 21):** the catalog leads with a single `NewDatasetPanel`, whose header pairs the panel title with the shared `.segmented-control` as a **source** switcher — Files · HuggingFace · Documents · Empty. It replaced four peer affordances (the ingest card plus three toolbar buttons of equal weight), which asked four questions before a byte moved and made the wrong one the default: the overwhelming case is "I have files", and the drop card handles it without asking anything. Only one source renders at a time, so `.ingest-card` no longer carries its own `.panel` wrapper — a panel inside a panel is two borders saying one thing (§5). The two LLM-only sources are **hidden**, not disabled, in a project that cannot hold an LLM dataset, so the row stays honest about what exists. Documents is a deliberate handoff rather than an inline flow: generating Q&A records from PDFs needs a model and real choices, so `.source-handoff` states that in one sentence and links to `/datasets/recipes`.

**Named progress, not a spinner (phase 21):** while the prep agent runs, `.prep-progress` shows the stage ladder — *read the files · work out the task · reshape the rows · build the splits* — with the server's own sentence above it ("Read 312 files — looks like classification, checking"). A run takes tens of seconds, and the word "working" for that long is indistinguishable from a hang. `--accent` marks only the rung in progress and the determinate `.prep-progress-fill`; completed rungs fall back to the neutral ink ramp so the accent budget stays at one element per panel (§2). Stage state is never carried by hue alone — `.prep-stage-mark` is a ring that fills with a `Check`, so the row reads as a checklist in greyscale. The transform rung renders only once reached, because a stage nobody will visit reads as a stall. The block is `role="status"` / `aria-live="polite"`, and reuses the system's one `.spinner` rather than introducing a second. It appears under the drop card during upload and on Overview afterwards, so navigating into the dataset moves the readout instead of restarting the explanation. Correspondingly, Overview's primary action on an already-trainable dataset is **View data**, not "Prepare with Orinth" — offering the agent as the headline action on finished data made a redo look like the next step.

**One grid for every modality (phase 21):** the Data tab opens on `.data-grid`, a semantic `<table>` in a `.grid-scroll` container (`overflow: auto`, so the page body never scrolls sideways per §8), with a **Table / Gallery** `.segmented-control` above it. It replaced three separate browsers — a thumbnail wall, a text preview list, and a records table — which meant the answer to "what is in this data" depended on which task you happened to have. The base `table` rules already supply the sticky micro-label header and hairline rows, so the grid adds only what a grid needs: a 44px `.grid-thumb` first column for image datasets, right-aligned `.grid-cell-number`, one-line `.grid-cell-text` with the full value in `title`, and `.grid-cell-select` — an in-row `<select>` that drops the form-control padding and the heavy `--line-input` border, which exists to make a standalone input read darker than its container and would tile at one per row. Editable cells are the label and the split only, because those are the two with a real write route behind them; `editable` arrives from the server beside the column definition rather than being guessed client-side, since a cell that looks editable and silently discards the edit is worse than a read-only one. Detection and segmentation get a class list and a region count instead of a single label cell — one label would be a lie about which of many boxes it names — and editing them stays the annotation editor's job in the Gallery view.

**Faceted Hub browse (phase 10, revised):** the HuggingFace panel is a `.hub-browse` two-column grid — a 200px `.hub-rail` of filter chips beside a `.hub-grid` of result cards — because the old flat list of ids was a lookup tool that only works when you already know the dataset's name. Chips are border-first like every other resting control: hairline `--line` at rest, `--accent` on `--accent-tint` when active, nothing lifts (§5). A term Orinth can browse but not import gets a **dashed** border and `--ink-subtle` rather than being hidden or disabled — hiding it leaves the user wondering where the image datasets went, and disabling it removes a filter that legitimately works. Cards clamp their id to two lines (the model-card rule, and for the same reason: an id is an identifier, often long) and carry the metadata that actually distinguishes two similar results — a `Viewer` badge, modality and format badges, a two-line summary, and download/like counts abbreviated to `226.0k`, since a card is scanned rather than audited.

**Explain every term the user is not required to know (phase 10, revised):** `parquet`, `webdataset`, `10K<n<100K`, `sharegpt` — a chip that names only a term is a quiz. Every filter chip, result badge, sort option, mapping field, and format toggle in the Hub browser carries a one-sentence `title`, and that sentence is **served** from `services/hub_facets.py` rather than written in the client, so the tooltip on a filter chip and the tooltip on the badge showing the same term on a card cannot drift apart. Native `title` is deliberate over a positioned popover: page chrome outside a canvas keeps `title` per §7 — the ~1s delay is right for labels that are scanned far more often than they are questioned, and it stays reachable by keyboard focus and screen reader with no focus trap. `InfoTip` / `Explained` in `@/features/platform/ui` are the shared primitives; prefer wrapping the term itself over adding a separate 12px icon target beside a 12px word.

**Look before you configure (phase 10, revised):** selecting a Hub dataset opens `.hub-detail-view`, which reads top-to-bottom as the three questions actually asked in order — is this the right dataset (title, badges, summary), what is in it (`.hub-viewer-table`, read-only, with each column's dtype under its name in the micro-label register), and how do its columns become records (`.hub-mapping`, a short form of named fields). The previous panel put a column-mapping `<select>` inside every table header, which made "look at this data" and "configure an import" the same gesture and forced a mapping decision before the user had decided they wanted the dataset. Sections are separated by a `--line` top border, not a card each. No new tokens, no shell change.

**Notebooks (phase 22):** `/notebooks` leads with the **template gallery**, not a "New" button, because the templates are the SDK's documentation — a user learns what `orinth` can do by opening *Dataset EDA*, not by reading signatures, and "does it fit in four notebooks" is the constraint that keeps the SDK small. The editor is a `.nb-layout` two-column grid: cells left, a runs rail right, collapsing below 1100px. A cell is a **bordered row, not a card** — they stack by the dozen and a shadow each would turn the page into corduroy (§5) — with a 44px gutter carrying run, the `[n]`/`*` execution counter in the mono stack, and delete. Focus moves the cell's border to `--accent`; nothing lifts. The runtime banner is a **status line with an action**, not a failure banner: a stopped kernel gateway is a normal state (it is a subprocess that costs memory, exactly like the llama.cpp server), and a missing optional extra is an install command shown as one in the `warn` pair, never `danger` — nothing is broken.

**Kernel output (phase 22):** rendering is an explicit MIME allowlist in dispatch order — the SDK's run bundle, PNG/JPEG, SVG, JSON, then `text/plain` — and anything outside it renders a labelled row naming the type with a download action. It never silently disappears: a user who cannot see their output has no way to know whether the cell worked. `text/html` is deliberately **not** rendered; it falls through to its `text/plain` alternative with a note, because sanitizing arbitrary kernel HTML needs DOMPurify and a policy, and a half-sanitized `dangerouslySetInnerHTML` fed by executed code is a script-injection hole in the studio. SVG is shown via a `data:` URI in `src` rather than inlined, which cannot execute. Streams and tracebacks reuse the dark `--shadow-ink` terminal block phase 15 established for `.export-log`, so a kernel's stdout and a conversion job's stdout read as the same thing, and output is truncated at 200,000 characters with the cut stated rather than letting a runaway loop take the tab down. ANSI colour maps onto the `--label-*` ramp — the sanctioned §2 exception, because a traceback's red and green are categorical data, not chrome — and codes the parser does not model (backgrounds, 256-colour) are dropped rather than approximated, since a wrong colour is worse than none. The CodeMirror 6 theme is written from platform tokens, never imported: a vendor theme is the same line phase 17 drew for React Flow.

## 9. Do / Don't

- **Do** add a token when a new color is genuinely needed; **don't** introduce a hex/rgb/hsl literal anywhere in styles or markup.
- **Do** keep one accent-colored element per region; **don't** put a second accent hue on a screen (the `--label-*` ramp is the sole exception, and only in charts and overlays).
- **Do** pair status colors as `-tint` background + `-strong` text. **Don't** use a base status hue as badge text — badges render at 11px, so 4.5:1 is mandatory and the base hues fail it (`--success` on `--success-tint` is 3.00:1; `-strong` gives 4.57:1).
- **Do** use `--line-input` on form controls; **don't** use `--line` there — at 1.26:1 it fails WCAG 1.4.11 for a control boundary.
- **Don't** use `--ink-subtle` for text below 18px on any surface darker than `--surface`.
- **Do** keep tables semantic (`<table>`) inside a horizontal-scroll wrapper; **don't** rebuild them as div grids.
- **Do** write empty states with a next-step action; **don't** render bare "No data".
- **Don't** add elevation to resting surfaces.
- **Don't** set type above 24px on a `(platform)` route, or use weight 700 anywhere. The auth display
  step in §10 is the only sanctioned way past 24px, and it is scoped to `/`.
- **Don't** use gradients, icon-tile card grids, decorative accent blobs, or centered hero layouts.
- **Don't** add diagnosis language anywhere, or marketing language on a `(platform)` route.
- **Don't** reintroduce a marketing landing page. `/` is the sign-in screen; a public marketing
  surface belongs on the Orinth site, not in this app.

## 10. Auth scope — `/` only

The one public route is the sign-in screen. It is the only place the system may compose outside the
dashboard shell, and the allowance is deliberately narrow. Tokens live in `app/globals.css`;
selectors live in `app/styles/auth.css`.

**What the auth scope may do that a platform route may not:**

| Allowance | Token | Why it is scoped |
|---|---|---|
| Type above 24px | `--display-section` (26→34px) | The page has one heading and no page header. Nothing inside the app is composed that way |
| Full-viewport two-column split | — | The shell is a fixed sidebar; sign-in has no sidebar and no project context |

**What does not change, and must not:** one typeface (Inter), weight cap 600, token-only color, the
single-accent budget, and border-first elevation. `--accent` here is focus rings only — the submit
button is the near-black `--ink` fill, exactly as `Button` renders it everywhere else. No shadow on a
resting surface.

**Content rules.** No invented metrics, testimonials, logo walls, or user counts. The sign-in notice
must keep saying that auth is not wired up, for as long as that is true. The medical-context rule
applies with full force: the page may say the platform *measures model behavior*, never that it
diagnoses.

## Exceptions

`app/global-error.tsx` is intentionally token-free (inline styles) — it must render when the CSS bundle fails. Its hardcoded values mirror the hex column in §2 and **must be re-synced by hand** whenever those tokens change. Nothing enforces this automatically.
