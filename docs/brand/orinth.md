# Orinth Brand Guide

## Name

- Product name: **Orinth**
- Pronunciation: **OR-inth** /ˈɔːrɪnθ/ — a coined name, so carry the guide wherever the name is introduced to a new reader.
- Parent brand: **Onestep AI Labs**
- Preferred lockup text: **Orinth**, with supporting copy **by Onestep AI Labs** when parent context is needed.
- Short descriptor: **The local AI studio**

The name is "orient" plus the *-inth* of *labyrinth*: finding the path through something complicated.

Do not refer to the product as a diagnostic system or autonomous medical decision-maker.

## Logo

The mark is a path turning through a rounded frame — the same idea the name carries. It is drawn as a
stroked polyline rather than a filled illustration, which is what keeps it legible at 16px and lets a
single asset serve light and dark surfaces.

Sources of truth:

- `frontend/components/brand.tsx` — `LogoMark`, the in-app mark. Strokes are `currentColor`, so the
  surrounding surface sets the ink and there is no light/dark variant to keep in sync.
- `frontend/public/brand/orinth-mark.svg` — the standalone tile used for the browser favicon and for
  README/presentation placements, where `currentColor` has nothing to inherit from. Ink Black tile,
  white path, Blush Peach terminal dot.

Usage guidance:

- Render the mark through `LogoMark` anywhere inside the app. Do not add an `<img>` or a second copy.
- Keep clear space of at least the frame's corner radius on all sides.
- Do not stretch, crop, recolor arbitrarily, add a drop shadow, or place the mark on busy imagery.
- Do not set the wordmark above weight 600 — the design system's cap applies to the brand too.

## Color

`frontend/DESIGN.md` §2 is authoritative for every color the product renders, and
`frontend/app/globals.css` holds the tokens. Nothing in this file overrides it.

Two colors belong to the mark specifically, and appear nowhere else in the app:

| Name | Hex | Use |
| --- | --- | --- |
| Ink Black | `#17191c` | The favicon tile background |
| Blush Peach | `#fbe1d1` | The terminal dot on the favicon tile |

Inside the app the mark is monochrome `currentColor`, so neither value is needed there.

## Voice

Concise, confident product language:

- Good: "Create a workspace for vision and NLP experiments."
- Good: "Compare model behavior across datasets."
- Good: "Prepare local datasets without rewriting originals."
- Avoid: "Automatically diagnose disease."
- Avoid: "Clinically approved decision engine."

## UI Tone

Orinth should feel like focused product software: dense, calm, precise, and trustworthy. Use cards for
repeated entities, compact chips for metadata, and restrained color accents for task states. The app
opens on sign-in and then on work — there is no marketing surface inside it.
