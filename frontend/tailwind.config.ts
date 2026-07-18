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
          soft: token("line-soft")
        },
        surface: {
          DEFAULT: token("surface"),
          2: token("surface-2")
        },
        panel: token("surface"),
        canvas: token("canvas"),
        teal: {
          DEFAULT: token("teal"),
          strong: token("teal-strong"),
          tint: token("teal-tint"),
          "tint-strong": token("teal-tint-strong")
        },
        coral: {
          DEFAULT: token("coral"),
          strong: token("coral-strong"),
          tint: token("coral-tint")
        },
        amber: {
          DEFAULT: token("amber"),
          strong: token("amber-strong"),
          tint: token("amber-tint")
        },
        green: {
          DEFAULT: token("green"),
          strong: token("green-strong"),
          tint: token("green-tint")
        },
        navy: {
          DEFAULT: token("navy"),
          deep: token("navy-deep")
        },
        indigo: {
          strong: token("indigo-strong"),
          tint: token("indigo-tint")
        }
      },
      fontFamily: {
        sans: ["var(--font-sans)", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"]
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        DEFAULT: "var(--radius)",
        lg: "var(--radius-lg)",
        pill: "var(--radius-pill)"
      },
      boxShadow: {
        card: "var(--shadow-card)",
        overlay: "var(--shadow-overlay)"
      }
    }
  },
  plugins: [
    require("@tailwindcss/typography"),
  ]
};

export default config;
