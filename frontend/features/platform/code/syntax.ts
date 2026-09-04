import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags } from "@lezer/highlight";

/**
 * Python syntax colours, from platform tokens.
 *
 * The first cut of the cell editor shipped `python()` for parsing and a theme
 * for chrome, but **no `HighlightStyle`** — so every cell rendered in one flat
 * ink colour. The parser was building a syntax tree nothing consumed.
 *
 * Colour here is functional, not decorative: it encodes what a token *is*,
 * which is the sanctioned §2 exception the phase-15 code viewer already relies
 * on. It rides the `--label-*` ramp so a keyword in a notebook is the same hue
 * as a keyword in the architecture studio's generated code, rather than a
 * second palette doing the same job.
 *
 * Deliberately narrow. Eight token classes cover Python; a fifty-entry style
 * sheet mostly adds distinctions the reader does not use and the eye cannot
 * hold.
 */

//: Weight, not only hue. The first cut coloured eight token classes at 400 and
//: leaned entirely on the ramp, which reads as washed at 13px on a light
//: surface — and disappears altogether for anyone with reduced colour vision.
//: 600 is the floor for anything that carries structure; comments and
//: punctuation stay light on purpose, because their job is to recede.
const STRUCTURE = "600";
const EMPHASIS = "700";

export const orinthHighlight = HighlightStyle.define([
  // Keywords and operators carry the structure, so they get weight as well as
  // hue — the distinction survives when someone turns colour down.
  // `def`, `class`, `import`, `return` — the words that carry the shape of the
  // file, so they get the heaviest weight as well as the strongest hue.
  { tag: [tags.keyword, tags.modifier], color: "oklch(var(--label-7))", fontWeight: EMPHASIS },
  { tag: [tags.controlKeyword, tags.moduleKeyword], color: "oklch(var(--label-7))", fontWeight: EMPHASIS },
  { tag: [tags.definitionKeyword], color: "oklch(var(--label-7))", fontWeight: EMPHASIS },

  // A string is the one token type a reader scans for by shape as much as by
  // colour, and green is the near-universal convention for it.
  { tag: [tags.string, tags.special(tags.string)], color: "oklch(var(--label-3))", fontWeight: STRUCTURE },
  { tag: [tags.number, tags.bool, tags.null], color: "oklch(var(--label-1))", fontWeight: STRUCTURE },

  { tag: [tags.function(tags.variableName), tags.function(tags.propertyName)], color: "oklch(var(--label-0))", fontWeight: STRUCTURE },
  { tag: [tags.definition(tags.variableName)], color: "oklch(var(--label-0))", fontWeight: STRUCTURE },
  { tag: [tags.className, tags.typeName], color: "oklch(var(--label-5))", fontWeight: EMPHASIS },

  // Decorators are how a reader spots `@app.route` and `@dataclass` at a
  // glance; they behave like keywords and are coloured like them.
  { tag: [tags.meta], color: "oklch(var(--label-6))", fontWeight: STRUCTURE },

  // `self`, `None`, `True` — named constants and the receiver read as fixed
  // points in a body, so they are marked rather than left as plain identifiers.
  { tag: [tags.self, tags.standard(tags.variableName)], color: "oklch(var(--label-6))", fontWeight: STRUCTURE },
  { tag: [tags.propertyName], color: "var(--color-ink)" },

  // Comments recede rather than compete. Italic because prose inside code is
  // the one place the distinction is worth a second axis.
  { tag: [tags.comment, tags.lineComment, tags.blockComment], color: "var(--color-ink-subtle)", fontStyle: "italic" },

  { tag: [tags.operator, tags.punctuation, tags.bracket], color: "var(--color-ink-muted)" },
  { tag: [tags.invalid], color: "var(--color-danger-strong)" }
]);

export const pythonHighlighting = syntaxHighlighting(orinthHighlight);
