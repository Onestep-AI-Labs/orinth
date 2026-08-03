import {
  Blocks,
  Boxes,
  Braces,
  Brain,
  Combine,
  Grid3x3,
  Layers,
  LogIn,
  Package,
  Repeat,
  Ruler,
  ShieldCheck,
  Sparkles
} from "lucide-react";
import { labelColor, labelFill } from "@/features/platform/utils";

/**
 * Per-category icon and colour index.
 *
 * `frontend/DESIGN.md` §2 keeps `--accent` as the single UI hue, so node accents
 * come from the `--label-0..7` data-visualization ramp instead — the sanctioned
 * exception. A node's category is categorical data being encoded by colour,
 * which is exactly what that ramp is for, and it stays confined to the node's
 * header tint, its stripe, and the palette dot.
 *
 * Twelve categories over an eight-colour ramp means some share a hue. The
 * repeats are chosen to be semantically honest rather than merely spaced:
 * Transformer primitives share LLM blocks' violet because one composes into the
 * other, and Backbone shares Convolution's green for the same reason.
 */
const CATEGORY_VISUALS: Record<string, { icon: typeof Blocks; label: number }> = {
  "Input & Output": { icon: LogIn, label: 0 },
  "Vision blocks": { icon: Layers, label: 3 },
  "LLM blocks": { icon: Boxes, label: 5 },
  Core: { icon: Blocks, label: 1 },
  Convolution: { icon: Grid3x3, label: 2 },
  Normalization: { icon: Ruler, label: 6 },
  Recurrent: { icon: Repeat, label: 4 },
  Transformer: { icon: Brain, label: 5 },
  Merge: { icon: Combine, label: 7 },
  Regularization: { icon: ShieldCheck, label: 3 },
  Backbone: { icon: Package, label: 2 },
  Custom: { icon: Braces, label: 4 },
  Unavailable: { icon: Sparkles, label: 7 }
};

const FALLBACK = { icon: Blocks, label: 1 };

export function categoryIcon(category: string) {
  return (CATEGORY_VISUALS[category] ?? FALLBACK).icon;
}

export function categoryLabelIndex(category: string): number {
  return (CATEGORY_VISUALS[category] ?? FALLBACK).label;
}

/**
 * The category's colour, for a stripe, dot, or icon.
 *
 * Goes through `labelColor`, which resolves `--color-label-N`. The raw
 * `--label-N` token is an unwrapped oklch triple, not a colour — using it
 * directly yields an invalid value and the element paints with no colour at
 * all, which is what the node stripes were doing.
 */
export function categoryColorVar(category: string): string {
  return labelColor(categoryLabelIndex(category));
}

/**
 * The pastel wash behind a node header or palette group.
 *
 * `labelFill` is the sanctioned way to get a translucent label colour;
 * concatenating an alpha suffix onto `labelColor` yields an invalid value
 * (DESIGN.md §8). Kept low so a canvas of forty nodes reads as tinted cards
 * rather than a colour chart — the tint is there to group, not to shout.
 */
export function categoryTint(category: string, percent = 12): string {
  return labelFill(categoryLabelIndex(category), percent);
}

export function CategoryIcon({
  category,
  size = 13,
  className
}: {
  category: string;
  size?: number;
  className?: string;
}) {
  const Icon = categoryIcon(category);
  return <Icon size={size} className={className} aria-hidden />;
}
