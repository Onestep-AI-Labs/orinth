# Spec: Phase 7 — Frontend UI Revamp and Hardening

## Status

In progress

## Goal

Give the entire frontend a visibly new, coherent UI driven by a single OKLCH token system, and harden the client: security headers and CSP, runtime validation at critical API boundaries, a toast layer, route-level loading/error boundaries, shared component primitives, and end-to-end journey continuity. Operational density is preserved — the first screen stays the working app.

Design references: usehallmark.com methodology (one anchor hue under ~5%, paired typefaces, 4px scale, asymmetric bias) and styles.refero.design pattern (AI-readable `frontend/DESIGN.md` contract). `frontend/DESIGN.md` is the authoritative design contract; `.claude/skills/frontend-design/SKILL.md` enforces it for agent work.

## Scope

In:

- One OKLCH token set in `frontend/app/globals.css`; Tailwind consumes tokens via `oklch(var(--token) / <alpha-value>)`.
- `frontend/app/styles/platform.css` fully tokenized (no raw hex) and restyled: dark navy sidebar with teal active accent, editorial page headers, unified card/table/form/dialog anatomy.
- Typeface pairing via `next/font/google`: Inter (`--font-sans`, body) + Space Grotesk (`--font-display`, display sites only).
- CVA-based primitives (`Button`, `IconButton`, `Badge`, `Chip`) in the `@/features/platform/ui` package; call sites migrated off raw class names.
- Toast layer (no new UI dependency) with global mutation-error routing through the React Query `MutationCache`.
- Route-level `loading.tsx` per segment and scoped `error.tsx` boundaries.
- Zod runtime validation for critical API responses via `jsonFetchChecked` (`frontend/lib/api/`); `jsonFetch` signature unchanged.
- Security headers and CSP in `frontend/next.config.mjs` (report-only first, then enforced).
- End-to-end journey pass: empty-state CTAs and stage-continuity links from project creation through inference.
- Documentation: `frontend/DESIGN.md`, `docs/ai/workflow.md`, `docs/ai/rules.md`, `CLAUDE.md`, `AGENTS.md`, this spec.

Out:

- Dark mode.
- Nonce/strict-dynamic CSP (would force whole-app dynamic rendering).
- Full OpenAPI-surface runtime validation or zod codegen.
- Information-architecture or route changes — the phase-5 navigation model stays authoritative.
- Backend changes.

## Interfaces

- API endpoints: none added or changed.
- Schemas: `frontend/lib/api/schemas.ts` — zod schemas for the FastAPI error envelope, job progress, project summaries, dataset detail, and inference response, each tethered to generated types via `satisfies z.ZodType<T>`.
- Frontend surfaces: all pages restyled; `frontend/features/platform/ui/` package (primitives, feedback, skeletons, confirmation, route error) and `frontend/features/platform/toast.tsx` added; `app/loading.tsx` + per-segment `loading.tsx`/`error.tsx` added.
- Storage/DB changes: none.

## Data Flow

- Mutation error → React Query `MutationCache.onError` → error toast (unless `meta.silentError`) + existing inline `MutationError` in the owning panel.
- Critical API reads → `jsonFetchChecked(path, schema)` → `safeParse` → typed data, or `ApiValidationError` naming the failing path, surfaced like any other API error.
- Fonts self-host through `next/font` at `/_next/static`, so CSP `font-src 'self'` holds; `/api` and `/media` stay same-origin through Next rewrites, so `connect-src 'self'` and `img-src 'self' blob: data:` hold.

## Edge Cases

- Dev vs prod CSP: dev adds `'unsafe-eval'` (React refresh) and `ws: wss:` (HMR); prod omits both.
- `NEXT_PUBLIC_BACKEND_URL` set (browser hits backend cross-origin, bypassing rewrites): that origin must be appended to `connect-src` and `img-src`, or requests will be blocked once CSP enforces.
- Offline `pnpm build`: `next/font/google` needs network once to download fonts; fall back to `next/font/local` with committed woff2 if builds must be fully offline.
- Validation failure on a critical endpoint fails loud (`ApiValidationError` toast + inline), never silently renders corrupt data.
- `app/global-error.tsx` renders without the CSS bundle; it keeps inline styles synced to the token hex fallbacks by hand.

## Acceptance Criteria

- `grep -E '#[0-9a-fA-F]{3,8}' frontend/app/styles/platform.css` returns nothing; the only palette source is `globals.css`.
- Teal appears only as primary action / active nav / focus / progress; coral only for danger.
- `curl -I` against a production server shows `Content-Security-Policy`, `X-Content-Type-Options`, `Referrer-Policy`, `X-Frame-Options`, `Permissions-Policy`, and no `X-Powered-By`.
- Killing the backend mid-session produces an error toast plus inline panel errors; no silent failures.
- Every empty state names and links the next step; each journey stage links forward on completion.
- Corrupt payloads on validated endpoints raise `ApiValidationError` naming the failing path.
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

Manual: walk landing → create project → dataset upload/annotate → version → training → testing → inference → settings/docs in `pnpm dev` (HMR alive, console free of CSP violations) and `pnpm build && pnpm start` (headers via `curl -I`, fonts served only from `/_next/static`, narrow-viewport table scroll per phase-5).
