"use client";

import { useEffect } from "react";

// Intentionally token-free: this boundary must render even when the CSS
// bundle fails. Values mirror the hex fallbacks in frontend/DESIGN.md.
export default function GlobalError({
  error,
  reset
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
          background: "#f5f7f9",
          color: "#1f2933"
        }}
      >
        <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
          <div
            style={{
              maxWidth: 420,
              width: "100%",
              border: "1px solid #d7dde5",
              borderRadius: 12,
              background: "#ffffff",
              padding: 24
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
              <span
                style={{
                  display: "inline-flex",
                  width: 36,
                  height: 36,
                  borderRadius: 999,
                  background: "#fff5f3",
                  color: "#ba3527",
                  alignItems: "center",
                  justifyContent: "center",
                  fontWeight: 700
                }}
                aria-hidden="true"
              >
                !
              </span>
              <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>Application error</h1>
            </div>
            <p style={{ margin: "0 0 4px", fontSize: 13, color: "#66788a" }}>
              A critical error stopped the app from rendering.
            </p>
            <p style={{ margin: "8px 0 16px", fontSize: 13, color: "#ba3527" }}>{error.message || "Unknown error"}</p>
            <button
              type="button"
              onClick={() => reset()}
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                border: "1px solid transparent",
                borderRadius: 6,
                minHeight: 40,
                padding: "0 14px",
                fontWeight: 600,
                background: "#0f766e",
                color: "#ffffff",
                cursor: "pointer"
              }}
            >
              Try again
            </button>
          </div>
        </div>
      </body>
    </html>
  );
}
