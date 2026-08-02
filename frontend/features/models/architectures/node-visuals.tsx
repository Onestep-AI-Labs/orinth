import {
  Blocks,
  Braces,
  Brain,
  Combine,
  Grid3x3,
  LogIn,
  Package,
  Repeat,
  Ruler,
  ShieldCheck,
  Sparkles
} from "lucide-react";

/**
 * Per-category icon and colour index.
 *
 * `frontend/DESIGN.md` §2 keeps `--accent` as the single UI hue, so node
 * accents come from the `--label-0..7` data-visualization ramp instead — the
 * sanctioned exception. A node's category is categorical data being encoded by
 * colour, which is exactly what that ramp is for, and it never leaks into
 * chrome: the tint appears only on the node's category strip and palette dot.
 */
const CATEGORY_VISUALS: Record<string, { icon: typeof Blocks; label: number }> = {
  "Input & Output": { icon: LogIn, label: 0 },
  Core: { icon: Blocks, label: 1 },
  Convolution: { icon: Grid3x3, label: 2 },
  Normalization: { icon: Ruler, label: 3 },
  Recurrent: { icon: Repeat, label: 4 },
  Transformer: { icon: Brain, label: 5 },
  Merge: { icon: Combine, label: 6 },
  Regularization: { icon: ShieldCheck, label: 7 },
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
 * The category's colour as a CSS custom property value.
 *
 * Returns a `var()` reference, never a hex literal — appending an alpha suffix
 * to this would produce an invalid colour, the trap `DESIGN.md` §8 calls out.
 */
export function categoryColorVar(category: string): string {
  return `var(--label-${categoryLabelIndex(category)})`;
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
