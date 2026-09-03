# Spec: Phase 26 Landing Page Refresh

## Status

**Draft.** Nothing here is built. The work lands in the sibling repository
`../orinth-landing-page`, not in this one.

## Where this lives

```
Orinth/
  orinth/                 this app (backend, frontend, desktop, specs)
  orinth-landing-page/    the marketing site — static Next.js, `next build` → out/
```

The spec is tracked here because the site describes *this* product and goes
stale when this repo ships something. `docs/ai/rules.md` puts the public
marketing surface on the Orinth site and keeps it out of the app; this is the
other half of that rule written down — the site is where the marketing lives,
and it is this repo's job to say when it is wrong.

**Today it is wrong in four ways**, all measurable:

1. `app/page.tsx` is 832 lines and reads as a document. The task matrix alone
   enumerates nine tasks with three fields each before a visitor has seen the
   product do anything.
2. **There is no motion.** Every product surface is a hand-built React mock
   (`components/studio/*`). A studio whose whole claim is "you can do this
   locally, in one app" is showing stills of itself.
3. **The docs are 545 lines across twelve pages, and predate five phases.**
   Nothing documents the `orinth` CLI (phase 23), notebooks or the Python SDK
   (phase 22), the architecture studio (phase 17), compute targets (phase 24),
   or the desktop app (phase 18).
4. It argues coverage where it should show a path. The page answers "what does
   it support"; the visitor is asking "how do I get from my folder of images to
   a model".

## Goal

A visitor who has never heard of Orinth sees, within one screen, **what it does
and what using it looks like** — moving, not described. A visitor who has
decided reaches docs that match the software they are about to install.

## Scope

In:

- **Cut the landing page to roughly a third.** Fewer sections, shorter copy,
  every claim earning its line.
- **Motion**: short looping screen captures (GIF/WebM) of the real app doing the
  four things that matter, replacing the static mocks in the hero and lifecycle
  sections.
- **A "how" spine**: one linear path — bring data → train → test → serve — as
  the page's structure rather than a feature list.
- **Docs brought current**, including new pages for the CLI, notebooks + the
  Python SDK, the architecture studio, compute targets, and the desktop app.
- **A quickstart that fits on one screen** and is copy-pasteable.

Out:

- Any change to the app in this repo. A landing-page phase that edits
  `frontend/` has misunderstood which product it is working on.
- A CMS, analytics beyond what is there, pricing changes, or the Hub section's
  information architecture.
- Rewriting the design system. `design_references/` is the contract the site
  already implements; this is a content and motion pass, not a restyle.

## The decisions

### Show, then tell — and the captures are of the real app

Four captures, 6–10 seconds each, silent, looping, no cursor trails:

| Capture | What it shows | Recorded on |
| --- | --- | --- |
| **Prepare** | A folder dropped into Dataset Studio; the prep agent detects the task and splits it. | `storage/datasets/` sample data |
| **Train** | A run started, the live loss curve, the run finishing and registering a model. | A short real run |
| **Test** | Confusion matrix and per-item review, a wrong item opened. | An existing evaluation |
| **Serve & chat** | A GGUF model served, a chat streaming token by token. | A small local model |

Rules that keep this honest:

- **Real app, real data.** No After Effects, no invented numbers, no fake
  latency. The compliance and no-diagnosis framing in `lib/site.ts` applies to
  frames of video exactly as it does to prose.
- **WebM (VP9) with a GIF fallback**, `autoplay muted loop playsinline`, a
  poster frame, `prefers-reduced-motion` → the poster only. Budget: **under 2 MB
  per capture**; the site is a static export and someone will open it on a
  phone.
- **Captured at 1440×900, cropped to the panel that matters.** A full desktop
  screenshot scaled into a card is unreadable, which is the usual reason product
  GIFs get skipped.
- They live in `public/captures/`, and a `content/captures.md` note records the
  app version each was taken from — a capture of a UI that no longer exists is
  worse than a screenshot.

### Less text, and the text that stays is second person

The current page explains the product to itself. The rewrite:

- **Hero**: one line of what it is, one line of who it is for, two buttons, and
  the Prepare capture. No paragraph.
- **Delete the task matrix** as a landing section. Nine tasks × three fields is
  reference material; it moves to a single docs page and is linked as "every
  task and its metrics".
- **One sentence per section**, then the capture, then at most three bullets.
- **No adjectives that cannot be checked.** "Runs on your machine" survives;
  "powerful" does not.
- Target: `app/page.tsx` under 300 lines, and the visible word count on the
  landing route cut by more than half.

### The page's structure is the product's path

Sections, in order, mirroring what a user actually does:

1. **Hero** — what it is + Prepare capture.
2. **Bring your data** — folder, CSV, YOLO/COCO export, Hugging Face import.
3. **Train** — the model library, the architecture studio in one line, the
   compute the run picks.
4. **Test** — task-aware metrics, per-item review.
5. **Serve & chat** — GGUF export, local serving.
6. **Work your way** — the three surfaces: studio, `orinth` CLI, notebooks with
   the Python SDK. **This section is new**, and it is the one that answers "can
   I script this".
7. **Install** — the quickstart, and the desktop download.

Everything else on the current page (comparison table, long feature grids,
repeated CTAs) is cut or moved to docs.

### Docs match the shipped software, and say when they were checked

New pages:

- **CLI** — `orinth` as an HTTP client of the backend: `doctor`, dataset,
  training, and serving verbs; completion; `ORINTH_BACKEND`. Source:
  `specs/phase-23-cli.md`.
- **Notebooks & the Python SDK** — the runtime, the editor, and `import orinth`:
  `datasets` (including `upload()` from a path), `models`, `train`, `evaluate`,
  `inference`, `runs`. One runnable example per module, lifted from the shipped
  templates so the docs cannot drift from what the product ships. Source:
  `specs/phase-22-notebooks.md`.
- **Architecture studio** — composing a graph, custom layers, and exporting
  Keras or PyTorch. Source: `specs/phase-17-model-architecture-studio.md`.
- **Compute** — per-framework device detection, what `auto` resolves to, and the
  declared remote providers with `available: false` stated as the fact it is.
  Source: `specs/phase-24-compute-targets.md`.
- **Desktop app** — the macOS `.dmg`, what it bundles, where its storage lives.
  Source: `specs/phase-18-macos-desktop-app.md`.

Existing pages get a pass for accuracy, and every page gains
`updated: YYYY-MM-DD` frontmatter rendered in the page header. A doc without a
date is a doc nobody can tell is stale.

**The SDK reference is generated, not typed.** A script reads the public
signatures and docstrings out of `../orinth/backend/orinth/` and writes
`content/docs/sdk-reference.md`. Hand-copied API docs are wrong within one
release, and this repo already treats the templates as the SDK's documentation —
the site should quote them, not paraphrase them.

### One quickstart, and it is the one in the README

```bash
git clone <repo> && cd orinth
make check      # verifies uv, pnpm, Python 3.11, Node
make dev        # backend :8000 + frontend :3000
```

Plus the desktop download for people who do not want a terminal. Anything longer
belongs in the installation doc, which is where the current page's install
section is going.

## Interfaces

No API. The changes are content and components in `../orinth-landing-page`:

- `app/page.tsx` — rewritten to the seven sections above.
- `components/landing/capture.tsx` — **new**. `<video>` with poster, reduced-
  motion fallback, lazy loading below the fold, and a caption naming the version.
- `components/studio/*` — kept where a capture cannot show it (the interactive
  demo), deleted where a capture replaces it. Deleting a mock the page no longer
  renders is part of the work, not a follow-up.
- `content/docs/{cli,notebooks,sdk-reference,architecture-studio,compute,desktop}.md`
  — new; `lib/content.ts` gains the `updated` frontmatter field.
- `scripts/generate-sdk-reference.ts` — **new**, run in `pnpm build`.
- `public/captures/*.webm|.gif|.png`.

## Data Flow

Static. Markdown under `content/` is read at build time; captures are static
assets. `next build` emits `out/`. The SDK reference is generated *before* the
build from the sibling repo's source, so a missing sibling checkout fails the
build loudly rather than shipping an empty page.

## Edge cases

- **`prefers-reduced-motion`** → poster frames, no autoplay, everywhere.
- **A capture fails to load** → the poster is the `<video>`'s own fallback; the
  section reads correctly with no motion at all.
- **The sibling repo is absent** when generating the SDK reference → the script
  fails with the path it looked for. Never a silent empty page.
- **A doc references a feature that has been removed** → caught by the accuracy
  pass; the `updated` date is what makes a stale page visible afterwards.
- **Mobile** → captures are cropped to the panel, not the window, and are the
  first thing dropped from the byte budget if they exceed it.

## Acceptance criteria

- `app/page.tsx` is under 300 lines and the landing route's visible word count is
  less than half of today's.
- Four captures render, autoplay muted, loop, and fall back to posters under
  `prefers-reduced-motion`; each is under 2 MB.
- Every navigation item resolves; no route 404s (`pnpm build` + a link check
  over `out/`).
- The docs cover the CLI, notebooks, the SDK, the architecture studio, compute,
  and the desktop app; `sdk-reference.md` is generated and matches
  `backend/orinth/`'s public surface.
- Every doc page shows an `updated` date.
- Lighthouse performance ≥ 90 on the landing route with the captures in place.
- No claim on the page that cannot be traced to this repo's README, specs, or a
  capture of the running app.

## Validation

```bash
cd ../orinth-landing-page
pnpm install
pnpm typecheck
pnpm build          # static export, and the SDK reference generation
```

Then open `out/` and watch the landing page end to end on a throttled
connection, with reduced motion on and off.
