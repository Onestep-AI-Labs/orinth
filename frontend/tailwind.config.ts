import type { Config } from "tailwindcss";

const token = (name: string) => `oklch(var(--${name}) / <alpha-value>)`;

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./features/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: token("ink"),
          muted: token("ink-muted"),
          subtle: token("ink-subtle")
        },
        line: {
          DEFAULT: token("line"),
          soft: token("line-soft"),
          strong: token("line-strong"),
          input: token("line-input")
        },
        wash: token("wash"),
        surface: {
          DEFAULT: token("surface"),
          2: token("surface-2")
        },
        panel: token("surface"),
        canvas: token("canvas"),
        accent: {
          DEFAULT: token("accent"),
          strong: token("accent-strong"),
          tint: token("accent-tint"),
          "tint-strong": token("accent-tint-strong")
        },
        danger: {
          DEFAULT: token("danger"),
          strong: token("danger-strong"),
          tint: token("danger-tint")
        },
        warn: {
          DEFAULT: token("warn"),
          strong: token("warn-strong"),
          tint: token("warn-tint")
        },
        success: {
          DEFAULT: token("success"),
          strong: token("success-strong"),
          tint: token("success-tint")
        },
        info: {
          strong: token("info-strong"),
          tint: token("info-tint")
        },
        "shadow-ink": token("shadow-ink")
      },
      fontFamily: {
        sans: ["var(--font-sans)", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"]
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        DEFAULT: "var(--radius)",
        lg: "var(--radius-lg)",
        xl: "var(--radius-xl)",
        pill: "var(--radius-pill)"
      },
      boxShadow: {
        subtle: "var(--shadow-subtle)",
        overlay: "var(--shadow-overlay)"
      },
      letterSpacing: {
        display: "var(--display-tracking)"
      }
    }
  },
  plugins: [
    require("@tailwindcss/typography"),
  ]
};

export default config;
