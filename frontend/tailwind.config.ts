import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#1f2933",
        panel: "#ffffff",
        line: "#d7dde5",
        teal: "#46b4a2",
        coral: "#f47f73"
      }
    }
  },
  plugins: []
};

export default config;
