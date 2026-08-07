/**
 * Orinth brand primitives — the one place the mark and the name are drawn.
 *
 * The mark is a path turning through a frame: "orient" plus the -inth of
 * labyrinth. It is a single stroked polyline rather than an image, so it stays
 * crisp at any size and inherits `currentColor` — which is what lets the same
 * component sit on the sidebar and on the sign-in aside without a light and a
 * dark asset. Sizing is the caller's job; the SVG carries no color of its own.
 */
export function LogoMark({ size = 24 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <rect x="1" y="1" width="22" height="22" rx="7" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M7 17V11.5C7 9.01 9.01 7 11.5 7S16 9.01 16 11.5V13"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="16" cy="16.5" r="1.9" fill="currentColor" />
    </svg>
  );
}

/** The product name. Coined names need a single spelling — never retype it. */
export const BRAND_NAME = "Orinth";

/** Sub-label used wherever the name needs one line of context. */
export const BRAND_TAGLINE = "The local AI studio";

/** The organisation behind the product, for footers and quiet attribution. */
export const BRAND_PARENT = "Onestep AI Labs";
