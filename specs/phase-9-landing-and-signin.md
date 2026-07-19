# Phase 9 — Public landing page and sign-in

## Goal

Give the project a public entry surface that promotes it, and put a sign-in step between that surface
and the workspace. Entry flow is `/` → `/signin` → `/projects`.

Before this phase the workspace itself was mounted at `/`. `docs/ai/rules.md` said "keep the first
screen as the working app, not a landing page". That rule was written when the app had no public
surface at all; this phase narrows it rather than deleting it — every route from `/projects` inward is
still the working app, and marketing framing is still banned there.

## Scope

**In scope**

- `/` — landing page promoting the platform (OSS / self-host positioning).
- `/signin` — presentational sign-in screen. No auth backend.
- Route regrouping so the marketing surface does not mount `AppShell`.

**Out of scope**

- Real authentication, sessions, users, or route protection. `docs/ai/rules.md` still requires auth
  before multi-user or network-exposed deployment; this phase does not deliver it.
- Any third marketing route (pricing, blog, changelog). Adding one needs its own spec.

## Interfaces

### Routing

`app/` is split into two route groups. Route groups do not affect URLs, so every existing app URL is
unchanged.

| Group | Layout | Routes |
|---|---|---|
| `app/(marketing)` | root layout only — no shell, no project context | `/`, `/signin` |
| `app/(platform)` | `AppShell` + project context | `/projects`, `/datasets`, `/models`, `/training`, `/testing`, `/inference`, `/settings`, `/documentation` |

The projects list moved from `/` to `/projects`. Six in-app links that pointed at `/` now point at
`/projects`: the sidebar wordmark, the Projects nav item, the project back-link, the create-project
back-link, `not-found.tsx`, and `RouteErrorPanel`.

`app/(platform)/loading.tsx` and `app/(platform)/error.tsx` are the moved root-level boundaries, so
every platform route keeps the fallback behavior it had before.

### Entry flow

`features/marketing/routes.ts` is the single source for the three hops:

```ts
LANDING_HREF   = "/"
SIGN_IN_HREF   = "/signin"   // landing CTAs target this
WORKSPACE_HREF = "/projects"  // sign-in redirects here
```

Wiring real auth later = submit against the auth endpoint in `signin-page.tsx` and keep the redirect
target in this file. No JSX hunt.

### Files

| File | Role |
|---|---|
| `features/marketing/landing-page.tsx` | Landing composition. Stage and model-family content are module consts |
| `features/marketing/signin-page.tsx` | Sign-in form. Client component; submit calls `router.push(WORKSPACE_HREF)` |
| `features/marketing/routes.ts` | The three route constants |
| `app/styles/marketing.css` | All landing/sign-in selectors. Imported from the root layout |
| `app/globals.css` | Landing type + rhythm tokens, and a new `--font-mono` token |

## Design

Governed by `frontend/DESIGN.md` §10, which this phase added. Summary of the decisions:

- Macrostructure is a five-stage narrative workflow (Organize → Prepare → Train → Test → Inspect),
  chosen because the app has a genuine sequence and DESIGN.md §8 already names journey continuity as a
  house pattern.
- The nav is edge-aligned with a hairline bottom border. The genre default (a floating pill with blur
  and soft shadow) was rejected because DESIGN.md §5 forbids elevation on a resting surface.
- Type above 24px is allowed here and only here, via `--display-hero` / `--display-section`.
- One typeface, weight cap 600, token-only color, and the single-accent budget all still hold. The
  primary CTA is the `--ink` fill, not the accent.
- Screenshots are the real app from `public/brand`, in hairline `<figure>` elements. No re-drawn
  browser chrome, no stock imagery, no invented metrics or testimonials.

## Data flow

None. Both routes are static and make no API calls — `/` is prerendered at 114 kB First Load JS versus
183 kB for the shell-mounted platform routes.

## Edge cases

- **Sign-in implies auth it does not have.** The screen carries a visible notice saying credentials are
  not checked, stored, or transmitted, and either field may be left empty. It must not be removed while
  the form is non-functional.
- **Global table styles leak.** `platform.css` styles bare `table`/`th`/`td` for dense dashboards.
  `.landing-table` resets `max-width`, `overflow`, `text-overflow`, `white-space`, and the uppercase
  row-header treatment. Removing that reset silently truncates the spec sheet.
- **`next/image` intrinsic width.** The captures are 1915×927; without `max-width: 100%` they lay out at
  1915px and push the document to ~2000px wide on phones, which the figure's `overflow: hidden` hides.
- **Documentation is linked from the landing footer** but lives in `(platform)`, so it renders with the
  shell. That is intentional — it is app documentation, not marketing content.

## Acceptance checks

- [x] `/` renders the landing page with no `AppShell` markup; `/projects` renders with the shell.
- [x] Every pre-existing app URL still resolves (21 routes build).
- [x] No horizontal overflow at 320 / 375 / 414 / 768 — `scrollWidth === clientWidth` at each.
- [x] No hardcoded hex / rgb / hsl in `marketing.css`.
- [x] Landing spec table shows full text, not ellipsis-truncated.
- [x] `pnpm typecheck`, `pnpm lint`, `pnpm test` (79), `pnpm build` all pass.

## Deferred

- Real authentication and route protection.
- Registration. The user deferred it with sign-in; the landing CTA goes to `/signin` only.
- An OG/social preview image for `/`.
