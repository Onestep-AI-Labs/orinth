# Phase 9 — Entry flow and project settings

## Status

Implemented.

Supersedes the separate `phase-9-landing-and-signin.md` and `phase-12-project-settings.md` drafts,
which are merged here. They ship together and share a route tree, a navigation model, and the
`(marketing)` / `(platform)` split, so keeping them apart meant two specs describing one router.

## Goal

Two related changes to how a user gets into and configures a workspace.

1. **Entry flow.** Give the project a public landing page and put a sign-in step in front of the
   workspace: `/` → `/signin` → `/projects`. Previously the workspace itself was mounted at `/`.
2. **Project settings.** Give each project a real settings surface for editing its name,
   description, and task types, seeing what it contains, archiving it, and deleting it.

The near-term motivation for (2): task types determine which datasets and bundled samples a project
can use. The backend has always supported editing them through `PATCH /api/projects/{id}`, but the
UI exposed only an inline rename popover, so a project's task types were frozen at creation and
adopting a new one meant recreating the project.

## Scope

**In**

- `/` landing page, OSS / self-host positioning.
- `/signin` — presentational only, no auth backend.
- Route regrouping so the marketing surface does not mount `AppShell`.
- `/projects/{projectId}/settings` — general settings, contents overview, identity, archive, delete.
- Domain-aware (Vision / NLP) task-type selection, shared with project creation.
- Project settings reachable from the project card menu **and** the project sidebar.

**Out**

- Real authentication, sessions, users, or route protection. `docs/ai/rules.md` still requires auth
  before multi-user or network-exposed deployment; this phase does not deliver it.
- Registration. The landing CTA goes to `/signin` only.
- Any third marketing route (pricing, blog, changelog) — needs its own spec.
- **Project visibility (private/public).** Deliberate: with no authentication, a visibility flag is
  not a security control, it is a label implying protection that does not exist, and the failure mode
  is a user trusting it. Meaningful when auth lands, not before.
- Cascade delete. Delete stays blocked while a project owns datasets or history.
- Per-project members, roles, or sharing — all blocked on authentication.
- Project duplication, export, or changing a project's id.

## Interfaces

### Routing

`app/` is split into two route groups. Route groups do not affect URLs, so every pre-existing app URL
is unchanged.

| Group | Layout | Routes |
|---|---|---|
| `app/(marketing)` | root layout only — no shell, no project context | `/`, `/signin` |
| `app/(platform)` | `AppShell` + project context | `/projects`, `/projects/new`, `/projects/{id}/settings`, `/datasets`, `/models`, `/training`, `/testing`, `/inference`, `/settings`, `/documentation` |

The projects list moved from `/` to `/projects`. Six in-app links that pointed at `/` now point at
`/projects`: the sidebar wordmark, the Projects nav item, the project back-link, the create-project
back-link, `not-found.tsx`, and `RouteErrorPanel`.

`app/(platform)/loading.tsx` and `app/(platform)/error.tsx` are the moved root-level boundaries, so
every platform route keeps the fallback behavior it had.

`features/marketing/routes.ts` is the single source for the entry hops — `LANDING_HREF`,
`SIGN_IN_HREF`, `WORKSPACE_HREF`. Wiring real auth later is a change to that file plus the sign-in
submit handler.

### Navigation model

Project settings is **project-scoped**, so the project sidebar owns it:

- `isProjectArea` matches `/^\/projects\/[^/]+\/settings/` in addition to the five workspace
  sections. `/projects` and `/projects/new` deliberately do not match — they are global, not
  project-scoped.
- The Settings entry sits in a **sticky footer** at the bottom of the project sidebar, separated from
  the five workflow sections by a hairline. The rail scrolls its nav, so pinning keeps it reachable
  without hunting for the end of a long project nav. Sticky only above 960px, where the sidebar
  actually scrolls.
- The **global** Projects nav item is not active on a settings route. Two nav items lighting up for
  one destination misreports where the user is.
- The project switcher navigates when it changes project on a settings route. Without that, switching
  renames the sidebar while the form underneath keeps editing the previous project.
- The settings page adopts the route's project **once per project id**, via a ref. The switcher sets
  the active project and *then* navigates, so there is a render where the URL still names the old
  project while context names the new one; re-running the adopt would read that intermediate state as
  a deep link and pull the user back.

### API endpoints

- `PATCH /api/projects/{project_id}` — existing, now also accepts `archived`.
- `DELETE /api/projects/{project_id}` — existing, unchanged.
- `GET /api/projects/{project_id}/stats` — new. Returns the counts that populate the overview and
  explain a blocked delete.

  ```json
  {
    "project_id": "retina-study-a1b2c3d4",
    "datasets": 3,
    "training_jobs": 2,
    "evaluation_jobs": 1,
    "inference_jobs": 0,
    "inference_runs": 12,
    "deletable": false,
    "blockers": ["3 datasets", "2 training jobs", "1 testing job", "12 inference results"]
  }
  ```

  `blockers` reuses the exact strings `DELETE` would return, so the danger zone and the eventual error
  message cannot drift apart.

No new single-project `GET`. The app shell already loads every project into context.

### Schemas

- `ProjectSummary` gains `archived: bool = False`.
- `ProjectUpdate` gains `archived: bool | None = None`.
- New `ProjectStats` with the fields above.

### Storage & DB

**No migration.** `archived` is stored inside the existing `projects.metadata` JSON column under the
reserved key `archived`, and surfaced as a typed boolean so the API contract stays explicit. The
per-project auto data prep model uses the same column under `default_openrouter_model`; it is not
promoted to a typed schema field, because putting an OpenRouter-specific field on the core project
schema before any OpenRouter code exists is coupling paid for in advance.

> **Reserved-key hazard.** `ProjectUpdate.metadata` replaces the metadata dict wholesale, so a client
> sending `metadata` without `archived` would silently un-archive the project. `update_project`
> therefore treats `archived` as service-owned: when `metadata` is replaced, the current archived
> state is re-applied unless `payload.archived` is set explicitly in the same request. Covered by a
> required test.
>
> The frontend has a **second instance of the same hazard**: `metadata.domain`. It is written at
> creation and describes which side a project leans to. Round-tripping it through the settings form
> would leave a project that moved from vision to NLP task types claiming the domain it just left, so
> `domain` is recomputed from the task selection on every save and listed alongside `archived` in
> `SERVICE_OWNED_METADATA`.

### Frontend surfaces

New:
- `app/(marketing)/page.tsx`, `app/(marketing)/signin/page.tsx` — thin route entrypoints.
- `app/(platform)/layout.tsx` — mounts `AppShell`.
- `app/(platform)/projects/[projectId]/settings/{page,loading,error}.tsx`.
- `features/marketing/{landing-page,signin-page,routes}.tsx|ts`.
- `features/projects/project-settings-page.tsx`.
- `features/projects/task-type-picker.tsx` — the Vision/NLP segmented control plus task cards,
  **shared by project creation and project settings**. Extracted rather than duplicated: two paths to
  the same edit invite them to diverge, and the domain rules must match exactly.
- `app/styles/marketing.css`.

Changed:
- `components/platform-pages.tsx` — export `ProjectSettingsPage`.
- `lib/api/projects.ts` — add `projectStats(projectId)`, preserving `import { api } from "@/lib/api"`.
- `features/projects/project-landing-page.tsx` — Settings and Archive/Unarchive card-menu items, a
  "Show archived" toggle, and a domain-relevant empty-state icon. **The inline rename popover is
  removed**; renaming lives in settings now.
- `features/projects/project-create-page.tsx` — consumes `TaskTypePicker`.
- `components/app-shell.tsx` — archived projects excluded from the switcher except the active one;
  the navigation model above.
- `features/platform/ui/index.tsx` — `PageHeader` gains an optional `actions` slot. DESIGN.md §6
  already specifies "actions right-aligned" as part of the header anatomy; it was simply unimplemented.
- `app/globals.css` — landing type/rhythm tokens and a `--font-mono` token.
- `app/styles/platform.css` — settings layout, identity list, save bar, danger zone, sidebar footer.
- `types/generated/api.ts` — regenerated via `pnpm generate:api` and committed.

## Design

Governed by `frontend/DESIGN.md`, which this phase extended with §10 (landing scope).

### Landing and sign-in

- Five-stage narrative workflow (Organize → Prepare → Train → Test → Inspect), chosen because the app
  has a genuine sequence and DESIGN.md §8 already names journey continuity as a house pattern.
- Edge-aligned nav with a hairline bottom border. The genre-default floating pill was rejected for
  requiring a shadow on a resting surface (DESIGN.md §5).
- Type above 24px is allowed here and only here, via `--display-hero` / `--display-section`.
- One typeface, weight cap 600, token-only color, single-accent budget all still hold. The primary CTA
  is the `--ink` fill, not the accent.
- Real screenshots from `public/brand` in hairline `<figure>` elements. No re-drawn browser chrome, no
  invented metrics, testimonials, or logo walls. No diagnosis language.

### Project settings

- **Layout.** Two columns above 68rem: the editable form on the left, read-only context on the right.
  Collapses to one column below. This replaces the earlier three-stacked-panel plan — on a wide
  monitor that put form inputs on a 1200px measure.
- **Contents vs Identity.** Counts are metric tiles; project id, created, and updated are a labelled
  list. They are different kinds of data and a 40-character project id has nowhere to wrap in a tile.
- **Save bar.** Sticky at the page bottom, always present, `--surface` with a hairline border and no
  shadow. Always present rather than appearing on first edit: Save is a fixed target you can find
  before touching anything, and the contract is that it is *disabled* when clean, not absent. Carries
  a Discard action and a state line.
- **Danger zone.** Set apart by a `--danger` border and a `--danger-tint` header, never by elevation.
  Rows are evenly padded with hairline dividers and stack under 40rem with full-width actions. The
  buttons inside supply the one destructive accent.
- **Spacing.** `PanelTitle` already carries `mb-4`; content follows it directly. The earlier page
  added `mt-4` on top, doubling every panel's first gap to 32px and disagreeing with every other page.
- **Elevation, typography, color, inputs, loading, voice** — unchanged from the design contract:
  panels are `--surface` + 1px `--line` + `--radius-xl` + no shadow; nothing above 24px or weight 600;
  `--line-input` borders; skeleton on initial query, `InlineSpinner` on refetch; inline
  `MutationError` **and** an error toast on every mutation; labels are nouns, buttons are verbs.
- **Icons.** `EmptyState` defaults to a picture icon, which is wrong outside dataset surfaces; project
  empty states pass a relevant icon. The active-nav-item chart glyph was removed — the active state is
  already carried by the soft chromatic fill (DESIGN.md §6), so the glyph was a second signal for the
  same thing.

## Data flow

Landing and sign-in are static and make no API calls: `/` is prerendered at 114 kB First Load JS vs
183 kB for shell-mounted platform routes.

**Edit.** Settings reads the project from app-shell context and fetches `GET /projects/{id}/stats`.
Form state is seeded from the project and reseeded whenever the stored project changes. Save issues one
`PATCH` with the changed fields, then refreshes the shell's project list so the switcher and card
reflect the new name immediately.

**Archive.** `PATCH` with `archived: true`. No confirmation — reversible and destroys nothing. The
project drops out of the switcher and the default list, staying visible under "Show archived" with a
badge. If it was active, the shell falls back to the default project. Unarchive is the same call with
`false`.

**Delete.** The danger zone reads `deletable` and `blockers` from stats; when blocked, the button is
disabled and the blockers are listed, so the user learns why before clicking rather than after. When
deletable, delete goes through `ConfirmationDialog`, then falls back to the default project and
returns to the project list.

## Edge cases

- **Sign-in implies auth it does not have.** The screen carries a visible notice that credentials are
  not checked, stored, or transmitted, and either field may be left empty. It must not be removed
  while the form is non-functional.
- **Default project** — cannot be deleted (existing `409`) and cannot be archived; archiving it would
  leave the switcher with nothing selectable and `ensure_default` would keep recreating it. Both are
  enforced server-side and shown as disabled controls with the reason stated.
- **Unknown project id in the route** — renders an `EmptyState` naming the problem and linking back,
  rather than crashing on a `null`.
- **Deep link before projects load** — `loading.tsx` renders `PageSkeleton`; the page waits for
  context rather than seeding a form with empty strings.
- **Deep link to another project's settings** — the route wins and becomes the active project, once.
- **Name emptied** — Save disabled client-side; `min_length=1` rejects it server-side.
- **Description over 500 characters** — `max_length=500` rejects it; a live character count shows the
  limit before submitting.
- **All task types deselected** — rejected client-side. A project with no task types can create no
  datasets, and `project_read` would silently substitute the default, so stored and displayed values
  would disagree.
- **Task type removed while datasets still use it** — allowed, warned about inline and again in a
  confirmation naming the affected count. Blocking traps a project whose early experiments are done;
  silently allowing strands datasets with no warning.
- **A vision-only project adopting an NLP task type** — both domain tabs are always reachable
  regardless of what the project currently declares. This is the concrete gap the phase closes.
- **Archived project is the active project** — kept selectable until the user switches away.
- **`metadata` replaced without `archived`** — the service re-applies the current archived state.
- **Stats fetch fails** — the overview shows an inline error and the danger zone stays disabled. A
  delete button enabled because the blocker check failed is worse than one that is disabled.
- **Concurrent edit** — last write wins. Single-user platform, no versioning anywhere else.
- **Auto data prep model** — persists to metadata, labeled as taking effect when auto data prep ships.
  Not presented as active; no validation against a model catalog is claimed, because none exists.
- **Documentation linked from the landing footer** lives in `(platform)` and renders with the shell.
  Intentional — it is app documentation. If auth later fronts `(platform)`, that link becomes a dead
  end for logged-out visitors.
- **Global table styles leak.** `platform.css` styles bare `table`/`th`/`td` for dense dashboards, so
  the landing spec table resets `max-width`, `overflow`, `text-overflow`, `white-space`, and the
  uppercase row-header treatment.
- **`next/image` intrinsic width.** Sized images lay out at intrinsic width without `max-width: 100%`,
  pushing the document to ~2000px on phones while `overflow: hidden` hides the evidence.

## Acceptance criteria

### Entry flow

- `/` renders the landing page with no `AppShell` markup; `/projects` renders with the shell.
- Every pre-existing app URL still resolves.
- No horizontal overflow at 320 / 375 / 414 / 768.
- No hardcoded hex / rgb / hsl in `marketing.css`.

### Project settings

- `/projects/{projectId}/settings` renders name, description, and task types seeded from the project,
  plus contents, identity, and a danger zone.
- Saving a new name updates the project card, the switcher, and the page header without a reload.
- Adding a task type to an existing project makes datasets of that type creatable in it.
- Removing a task type still used by existing datasets warns with the affected count and proceeds when
  confirmed.
- Deselecting every task type disables Save.
- Archiving removes a project from the switcher and default list, shows it under "Show archived" with
  a badge, and is reversible in one click.
- The default project cannot be archived or deleted; both controls are disabled with the reason shown,
  and the server rejects both independently of the UI.
- The danger zone lists blockers from stats and disables delete while any exist.
- `PATCH` with a `metadata` object omitting `archived` does **not** un-archive the project.
- Saving after moving a project's task types across domains updates `metadata.domain`.

### Navigation

- Opening settings from the project card menu or the sidebar shows the project sidebar, with Settings
  active in its sticky footer and the global Projects item **not** active.
- Switching project from the switcher while on settings moves to that project's settings and the form
  reseeds.

### Tests

Backend, extending `tests/test_project_service.py`: archive/unarchive round-trip; `update_project`
with a replacement `metadata` dict preserves archived state (the reserved-key regression); an explicit
`archived` alongside `metadata` wins; archiving the default project raises `409`; `project_stats`
returns counts and a `blockers` list identical to the strings `delete_project` raises with, asserted
against the same fixture; `deletable` tracks blockers.

New `tests/test_project_stats_route.py`: registered, `404` for unknown, counts for known.

Frontend `features/projects/project-settings-page.test.tsx`: renders seeded values; Save disabled when
nothing changed, when the name is empty, and when no task type is selected; delete disabled with
blockers listed; archived badge renders; **opens on the domain the project already works in**;
**reaches the other domain so a vision project can adopt an NLP task type**, asserting the saved
payload's `task_types` and recomputed `metadata.domain`.

### Manual checks

1. Open a project's settings from the card menu; confirm seeding.
2. Rename and confirm the switcher and card update immediately.
3. Add `text_classification` to a vision-only project and confirm an NLP dataset is then creatable.
4. Archive; confirm it leaves the switcher and appears under "Show archived". Unarchive.
5. Confirm the default project's archive and delete controls are disabled with a stated reason.
6. On a project with datasets, confirm delete is disabled and blockers listed; delete the datasets and
   confirm delete becomes available.
7. Switch project from the sidebar while on settings; confirm the URL and form both follow.

## Validation

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm test
cd frontend && pnpm build
```

Or `make check`.

**Highest-risk check:** the metadata reserved-key interaction, in both its instances. `archived` is
the server-side one — any client sending metadata without it silently un-archives the project, a
data-loss-shaped bug with no error anywhere. `domain` is the client-side one, and goes stale silently.

## Deferred

- Real authentication, registration, and route protection.
- An OG/social preview image for `/`.
- Cascade delete.
