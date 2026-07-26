# Onestep AI Platform — Design System

AI-readable design contract for the frontend. Tokens live in `app/globals.css`; selectors that consume them live in `app/styles/platform.css`. New code uses primitives from `@/features/platform/ui`. Never hardcode a color value.

## 1. Identity & Voice

Operational research instrument for medical-imaging and NLP experiments — not a marketing site, not a diagnostic device.

- Every platform screen is the working app. No landing-page composition inside `app/(platform)`.
  The one public entry surface — `/` and `/signin` — is scoped in §10 and is the sole exception.
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
| `MultiSelect` / `TaskSelect` | — | Custom dropdowns: the closed trigger reads as a `<select>`, the open `--shadow-overlay` panel as a listbox. `MultiSelect` is checkbox multi-select; `TaskSelect` is single-select with Vision / NLP / LLM tabs + a search field (see §8) |

## 8. Patterns

**Loading contract:** initial query → skeleton scoped to the data component; refetch of visible data → `InlineSpinner`; long mutation → global overlay (automatic via `useIsMutating`); route transition → segment `loading.tsx` renders `PageSkeleton`.

**Errors:** every mutation shows inline `MutationError` in its panel *and* an error toast (automatic via the mutation cache). Never swallow a failure.

**Confirmation:** destructive actions (delete project/dataset/model, cancel running job) require `ConfirmationDialog`.

**Journey continuity:** each stage points to the next — empty states and completion moments link forward (project → dataset → annotate → version → train → test → infer).

**Motion:** transitions use ease-out, 120–200ms; anything animated respects `prefers-reduced-motion`.

**Data visualization:** class colors come from `labelColor(index)`; translucent annotation overlays come from `labelFill(index, percent)`. **Never append a hex alpha suffix to `labelColor`** — it returns a `var()` reference, not a hex literal, so `` `${labelColor(i)}33` `` yields an invalid color and paints the overlay opaque black over the image beneath it. SVG `stroke`/`fill` must be set via `style={{ … }}`, not as presentation attributes — `var()` does not resolve in those.

**Task selection (tabbed, searchable `TaskSelect`):** every operational task dropdown — training, testing, inference, and LLM serving — uses the shared `TaskSelect` primitive instead of a raw `<select>`. It takes the project's allowed task types and splits them across Vision / NLP / LLM **tabs** (the shared `.segmented-control`, shown only when the project spans more than one domain), so the grouping mirrors the project-creation `TaskTypePicker`. A sticky panel head pairs the tabs with a **search field** (`.task-select-search`, `Search` icon) that filters across *all* domains at once — while a query is active the results ignore the tab and regroup under uppercase domain micro-labels (`.task-select-group-label`), and Enter picks the first match. Each option is an icon + name + one-line description row; the active option carries the `--accent-tint` background with `--accent-strong` name and a `Check`, and the closed trigger shows the selection's icon, name, and a `.task-select-domain-tag`. Panel elevation is the sanctioned `--shadow-overlay`; close on click-outside/Escape matches `MultiSelect`. No new tokens, no shell change.

**Dataset catalog sections (phase 10):** provenance groups (Project / Imported from HuggingFace / Shared samples) are hairline-separated blocks with an uppercase micro-label header (`.section-microlabel`), rendered only when non-empty. No new tokens.

**LLM records (phase 10):** the `llm_finetune` Records tab is a semantic `<table>` in a `.table-wrap` (`overflow-x: auto`) container; the role-aware edit drawer and the hub import dialog are the only elevated surfaces, using the sanctioned `--shadow-overlay`. The hub detected-format banner uses the `success` pair, the manual-mapping hint the `info` pair, and the gated badge the `warn` pair — all existing status tokens.

**Data recipes (phase 11):** the recipes landing is a flat template gallery — `.recipe-template-card` cards are hairline-bordered `--surface` (blank card `--surface-2`, dashed), never Unsloth Studio's gradient/shine surfaces; concept badges are `neutral`. The workspace is a linear `.recipe-stepper` (Sources → Generate → Review → Commit) of pill steps; the active step borders `--accent`, done steps fill the index dot `--accent`. Record provenance badges use the `info`/`neutral` pair (`llm` vs `rules`); the run warnings panel uses the `warn` `-tint`/`-strong` pair; the review edit drawer reuses `.record-drawer` and its `--shadow-overlay`. No new tokens, no shell change.

**Custom model upload (phase 12):** the models page groups cards into hairline-separated `.model-section` blocks (Reference / Trained / Uploaded), each rendered only when non-empty, with a per-card source `Badge` — `neutral` reference, `info` trained/promoted, `ok` uploaded. The `.upload-dialog` is a sanctioned `--shadow-overlay` overlay reusing `.confirmation-overlay`; its family selector is a `.upload-family-chip` pill row (active chip borders `--accent` on `--accent-tint`), the sklearn security note uses the `danger` `-tint`/`-strong` pair, and LLM gate notes use the `warn` pair. The model detail route follows the eyebrow + 24px title `PageHeader` pattern with a `.model-detail-grid` metadata block and hairline `.model-artifact-list`; `llm_*` families show the phase-15 export and serving panels (see the phase-15 note) instead of inference/testing links. No new tokens, no shell change.

**Advanced training settings (phase 13):** the training form adopts a Configure anatomy — a full-width `.training-model-section` header (task · base model · name in a `form-grid-three` row, with the option description as a token-only `.form-caption`) over a `.training-config-grid` that pairs a narrow `.training-config-rail` (Dataset + Run panels stacked) with the wider Parameters panel, collapsing to one column below 1024px. The rail keeps the short decision cards from stranding a lone field beside the tall Parameters panel. This is a layout regrouping only; no field changes ownership between basic and advanced. The Dataset card renders an `EmptyState` with a "Go to datasets" `ButtonLink` when the active task has none; Run actions use the `Button` primitive (`secondary` Prepare + `primary` Start). The Parameters card hosts a collapsed-by-default `.advanced-panel` accordion (reusing the dataset studio's `.accordion-*` classes and `ChevronDown` rotation), hairline-separated from the basic fields above. Fields render generically from the catalog `AdvancedParameterSpec` list — the form has no per-family knowledge, so new families get their UI for free — grouped by `group` under uppercase `.advanced-group-label` micro-headers (Optimization, Augmentation, Regularization, Runtime). Numeric fields clamp to their `min`/`max` via `NumberInput`; the header shows an `.advanced-changed-count` when values differ from catalog defaults. The job detail page renders submitted advanced values as an "Advanced settings" `KeyValueTable`, so a finished run documents its own configuration. No new tokens, no shell change.

**LLM serving, export, and chat (phase 15):** the model detail page gains two hairline `.panel` blocks for `llm_*` families (in `features/models/llm-panels.tsx`). The **Export panel** (`.export-panel`, a 20px vertical-rhythm grid) reads top-to-bottom: a header with the section title and a width-capped `.export-intro` (78ch), a bordered `.export-config` block (`--surface-2`) grouping the format choice under an uppercase `.export-label` micro-header — a `.export-format-chip` pill row (active chip borders `--accent` on `--accent-tint`, matching the phase-12 upload chips), a one-line `.export-format-desc` for the selected format, then a quantization `<select>` (revealed only when the GGUF chip is active) beside the run `Button` — and a `.export-history` section with its own micro-label plus a "N running" `Badge`. Each `.export-row` is a hairline card with a spaced `.export-row-head` (format + status `Badge` + size on the left, download action right), and, while a job runs, an `.export-log` terminal block (dark `--shadow-ink` bg, mono 12px, `overflow-y: auto`) under a "Live output" label — matching the training log. The **Serving panel** pairs a serving-state `Badge` (`ok` running, `info` starting/stopping, `neutral` stopped) with start/stop `Button`s; non-GGUF families show an `EmptyState` "Export to GGUF first" CTA instead. The **chat surface** (`/inference/chat`, `features/inference/chat/chat-page.tsx`) is a `.chat-layout` two-column grid: a sticky `.chat-rail` (served-model status + a `--shadow-overlay` `.chat-model-picker-panel` popover over servable GGUF models, sampler `SliderField`s, system-prompt textarea) beside a flex `.chat-column` whose `.chat-transcript` scrolls above a pinned `.chat-footer-stats` (tokens/s + time-to-first-token) and `.chat-composer`. Transcript rows are hairline-separated `.chat-turn`s with uppercase `.chat-turn-role` micro-labels — no bubbles, no shadow; the streaming `.chat-caret` blinks 160ms ease-out and collapses to static opacity under `prefers-reduced-motion`. Both roles render GitHub-flavored markdown (`.chat-md`, token-styled headings/lists/tables/links) — including inside the `.chat-think` reasoning block, so code and headings in a model's reasoning are parsed rather than shown as literal `###`/```` ``` ````. Code blocks are syntax-highlighted (`react-syntax-highlighter` `PrismAsync` + `oneDark` — the async build auto-loads grammars so highlighting is actually colored while still code-splitting; the *Light* variant renders colorless without manual language registration) carrying a `.chat-code-head` language label and a `.chat-copy-btn` copy control, with a per-message copy button in `.chat-turn-head`. Collapsible `<think>…</think>` reasoning sits in a `.chat-think` disclosure (auto-expanded while streaming, collapsed once done). The composer carries a `.chat-composer-chips` row of `.chat-mode-chip` toggles (active = `--accent-tint`/`--accent-strong` border): **Search** (web search) and **Thinking** (show/hide reasoning) — the reference's Search/Code chip pattern. With Search on, an assistant turn shows a `.chat-citations` block (numbered `--accent` source links + host) above the answer. The served-model control uses the shared `ModelSourcePicker` (`.model-source-picker`, also used by the inference serve view) — a `serve-source-chip` tab toggle over Trained (registered GGUF with sizes), Local (persisted path auto-scanned into a selectable `.model-source-row` list + OS-native dialogs), and Hugging Face (a "Recommended for your hardware" list ranked by fit, then search) — so selection is identical across chat and inference. Options stay selectable while a model runs (starting a new one stops the old). The transcript, footer stats, composer, and chip row share a centered ~820px measure (`.chat-turn` / `.chat-footer-stats-inner` / `.chat-composer-inner` / `.chat-composer-chips`) so prose does not stretch the full panel width. Syntax-highlighter theme colors are the sanctioned functional-color exception (same basis as the docs viewer and the `--label` chart ramp), not chrome. Chat streams token-by-token over SSE through a dedicated `app/api/serving/chat/route.ts` streaming proxy (the config `/api/*` rewrite buffers streaming responses; a route handler returning the upstream `ReadableStream` does not). Model loading keeps the full-screen `.global-loading-overlay` but swaps the circular spinner for an indeterminate `.global-loading-bar` progress bar ("Loading model…") when a `serving-start`-keyed mutation is active; every other mutation keeps the circular spinner. The send button is the default `--ink` primary; `--accent` stays on focus rings only. Serving errors and mid-stream failures surface both in the transcript and the toast layer. Medical-context framing carries through (research instrument, no diagnosis language, no medical-assistant system-prompt preset). The **inference landing** branches on task: for `llm_finetune` it drops the submit-and-render form and renders `LlmServeView` (`features/inference/llm-serve-view.tsx`) — a **single** Serve panel (no Chat panel) with a three-way source toggle (`.serve-source-chip`): **Trained models** (registered `llm_gguf`, each with its file size), **Local folder**, and **Hugging Face**. Starting a server from any tab **auto-redirects to `/inference/chat`** (start returns only when the server is ready); the old "Open chat" launcher panel is gone, with a small Open-chat link kept only while a session is already running (beside Stop). Local folder uses the **OS-native file dialog** (`.serve-browse-actions` — Browse folder / Browse .gguf buttons calling `/serving/pick`, stdlib `osascript`/`zenity`, no third-party lib) plus a path field that **auto-scans** (debounced, no Scan button) and is **persisted** (restored + rescanned on load via `/serving/config`); results list GGUF files with sizes. Hugging Face (`.serve-hub`) is a search field + GGUF/MLX format select over a `.serve-hub-list` of hairline `.serve-hub-repo` rows (downloads/likes), each expanding to its files (quant + size) with a Download action; GGUF downloads then serve, MLX is download-only. When no GGUF exists the Trained-models tab shows an `EmptyState` linking to the models page. The header's "Chat with a served LLM" link is conditional — LLM task with at least one servable model only. The **testing page** gains an `llm` evaluation display kind alongside the existing task kinds — perplexity / token-accuracy / loss metric columns and a per-record table (prompt · reference · perplexity · loss) — reusing the same `testing-components.tsx` column/detail machinery as every other kind; GGUF models are omitted from the testing model picker (tested via chat, not batch eval). No new tokens, no shell change.

**LLM fine-tuning (phase 14):** the LLM training form reuses the phase-13 Configure grid unchanged. The environment banner (`.llm-env-banner`, device + recommended backend + notes) uses the `info` `-tint`/`-strong` pair; gated hub options show a `warn` `Badge` with a lock icon and a "license required" suffix in the select, whose options group into Hub models / Local base models `<optgroup>`s. Selecting "Custom Hugging Face model…" reveals a hub-id `Field`; a token-only `.form-caption` names the trainable base format (HF Transformers) and calls out GGUF/TFLite as non-trainable. The LLM 4-bit/precision advanced defaults auto-detect from the probed device (off/fp32 on MPS/CPU); the vision Device control is an auto-detected `<select>` (Auto + detected accelerator + CPU), not free text. A `.model-detail-line` caption shows live Hugging Face API detail (real download size, file count, downloads, a `warn`-toned "license required" marker, or a `danger`-toned error) for the selected base; the gated badge is driven by this live value. During a run the base-model download streams progress lines into the log tail instead of going silent. The fine-tuning method (LoRA / QLoRA / Full / Continued pretraining) is a prominent basic `<select>` in the Parameters panel with device-aware inline hints, not buried in the accordion (it is filtered out of the generic accordion render). The advanced accordion gains Method / LoRA / Quantization / Sequence group micro-headers ahead of the phase-13 four (backend constants and frontend `GROUP_ORDER` mirror each other). Creating an LLM job navigates straight to the training detail page — the platform's equivalent of Studio's Current Run switch. On the detail page, live step-denominated loss curves reuse the `--label` ramp `MiniLineChart`s, and completed runs render `sample_generations` as `.generation-card`s — hairline `--surface` cards with uppercase micro-labels and the mono stack at 13px. No new tokens, no shell change.

## 9. Do / Don't

- **Do** add a token when a new color is genuinely needed; **don't** introduce a hex/rgb/hsl literal anywhere in styles or markup.
- **Do** keep one accent-colored element per region; **don't** put a second accent hue on a screen (the `--label-*` ramp is the sole exception, and only in charts and overlays).
- **Do** pair status colors as `-tint` background + `-strong` text. **Don't** use a base status hue as badge text — badges render at 11px, so 4.5:1 is mandatory and the base hues fail it (`--success` on `--success-tint` is 3.00:1; `-strong` gives 4.57:1).
- **Do** use `--line-input` on form controls; **don't** use `--line` there — at 1.26:1 it fails WCAG 1.4.11 for a control boundary.
- **Don't** use `--ink-subtle` for text below 18px on any surface darker than `--surface`.
- **Do** keep tables semantic (`<table>`) inside a horizontal-scroll wrapper; **don't** rebuild them as div grids.
- **Do** write empty states with a next-step action; **don't** render bare "No data".
- **Don't** add elevation to resting surfaces.
- **Don't** set type above 24px on a `(platform)` route, or use weight 700 anywhere. The landing type
  scale in §10 is the only sanctioned way past 24px, and it is scoped to `/` and `/signin`.
- **Don't** use gradients, icon-tile card grids, decorative accent blobs, or centered hero layouts.
- **Don't** add diagnosis language anywhere, or marketing language on a `(platform)` route.

## 10. Landing scope — `/` and `/signin` only

The public entry surface is a landing page and a sign-in screen. It is the one place the system is
allowed to compose like a marketing page, and the allowance is deliberately narrow. Tokens live in
`app/globals.css`; selectors live in `app/styles/marketing.css`.

**What the landing scope may do that a platform route may not:**

| Allowance | Token | Why it is scoped |
|---|---|---|
| Type above 24px | `--display-hero` (36→56px), `--display-section` (26→34px) | A headline is the page's whole job. Nothing inside the app has that job |
| Section rhythm past 48px | `--landing-gap` (64→112px) | The app is a full-bleed dashboard; the landing page is a read-through document |
| Centered content column | `--landing-max` (72rem), `--landing-measure` (34rem) | The shell is a fixed sidebar; the landing page has no sidebar |

**What does not change, and must not:** one typeface (Inter), weight cap 600, token-only color, the
single-accent budget, and border-first elevation. `--accent` on the landing page is links and focus
rings only — the primary CTA is the near-black `--ink` fill, exactly as `Button` renders it everywhere
else. No shadow on a resting surface, including the nav, the figures, and the spec table.

**Content rules.** Product captures are real screenshots from `public/brand` in a hairline `<figure>` —
never a re-drawn browser frame, phone mockup, or fake IDE chrome. No invented metrics, testimonials,
logo walls, or user counts. The medical-context rule applies with full force: the page may say the
platform *measures model behavior*, never that it diagnoses.

**Beware the global table rules.** `app/styles/platform.css` styles bare `table` / `th` / `td` for
dense dashboards — `white-space: nowrap`, ellipsis truncation, a 340px cap, sticky uppercase headers.
Any unscoped `<table>` inherits them. `.landing-table` resets them explicitly; a new prose table must
do the same rather than weakening the global rule.

**Beware `next/image` intrinsic sizing.** A sized `<Image>` lays out at its intrinsic width unless the
CSS sets `max-width: 100%`. `width: 100%` alone is not enough, and a wrapper's `overflow: hidden` will
hide the resulting document overflow instead of preventing it.

## Exceptions

`app/global-error.tsx` is intentionally token-free (inline styles) — it must render when the CSS bundle fails. Its hardcoded values mirror the hex column in §2 and **must be re-synced by hand** whenever those tokens change. Nothing enforces this automatically.
