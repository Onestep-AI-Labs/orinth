---
name: frontend-design
description: Design methodology for all frontend UI and styling work — layout, color, typography, spacing, components, or CSS in frontend/. Use before writing or reviewing any UI code, styles, or new screens.
---

# Frontend Design

Read `frontend/DESIGN.md` first. It is the contract; this skill is how you apply it.

## The Eight Rules

1. **Type** — Pair the display face with the body face. Space Grotesk (`--font-display`) for display sites only; Inter (`--font-sans`) for everything else. Never one font doing both jobs.
2. **Colour** — OKLCH tokens only, defined in `frontend/app/globals.css`. Teal is the single anchor accent and stays under five percent of any screen. Coral means danger, never decoration.
3. **Space** — The 4px scale. No arbitrary 17-pixel paddings. Radii come from the four radius tokens.
4. **Motion** — Exponential ease-out. Every animation has a `prefers-reduced-motion` alternative.
5. **Voice** — Operational instrument register: labels are nouns, buttons are verbs. Never the SaaS-default neutral middle, never marketing copy.
6. **Layout** — Bias the page. The dark navy sidebar is the one asymmetric anchor; content sits on canvas. Centred everything is a tell.
7. **Hierarchy** — Display, body, micro-label: a weight ladder you can read in two seconds. Weight cap is 700.
8. **Restraint** — Better nothing than bad something. The strongest fail-state is silence.

## Refusals

Never produce:

- Purple or multi-hue gradients.
- Inter (or any body face) as the display font.
- Centred hero layouts or landing-page composition — the first screen is the working app.
- Generic icon-tile card grids.
- AI-standard navbars (logo left, links center, CTA right).
- Hardcoded hex, rgb(), or hsl() values — add or reuse a token.
- A second accent hue on a screen.
- Hidden errors — backend failures surface in the relevant panel and the toast layer.
- Clinical-diagnosis language — this is a research platform, not a diagnostic device.

## Checklist

Before finishing any UI change:

- [ ] Colors come from tokens in `frontend/app/globals.css` (`grep -E '#[0-9a-fA-F]{3,8}' frontend/app/styles/platform.css` stays clean).
- [ ] Interactive elements use primitives from `@/features/platform/ui` (Button, Badge, EmptyState, toast) — no raw class strings in new code.
- [ ] Loading follows the contract: initial query → skeleton, refetch → inline spinner, long mutation → global overlay.
- [ ] Empty states include the next-step action.
- [ ] `cd frontend && pnpm typecheck && pnpm lint && pnpm test && pnpm build` passes.
