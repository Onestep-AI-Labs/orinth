# Dub design DNA — extraction archive

Provenance for the phase-8 design system. Produced by a `hallmark study` extraction of Dub, then locked
to a portable `design.md` plus token exports.

| File | What it is |
|---|---|
| `DESIGN.md` | The full style-extraction report |
| `variables.css` | Plain `:root` custom properties (hex) |
| `theme.css` | Tailwind v4 `@theme` block |
| `tokens.json` | DTCG-format token file |

## This is reference material, not the contract

**`frontend/DESIGN.md` is the authoritative design contract.** These files record where the system came
from; they do not describe what this app looks like, and they should not be applied literally.

The extraction is from a **marketing site**. Only its system layer — color ramp, border-first elevation,
radius vocabulary, density — was adopted. Its composition layer was rejected outright:

- hero sections, floating feature pills, logo clouds, product-mockup frames
- 30 / 36 / 48px display type, 16px canonical body
- conic-gradient brand mark, dotted-grid backgrounds
- 64px section gaps, 1200px centered max-width

Those conflict with `docs/ai/rules.md` ("keep the first screen as the working app, not a landing page"
and "prefer dense, operational UI over marketing-style presentation").

Three values in the shipped system also deliberately deviate from this source — the canvas color, the
input border weight, and `--ink-subtle`. Each is documented with its rationale in `frontend/DESIGN.md` §2.

See `specs/phase-8-design-system-dub.md` for the full adoption record.
