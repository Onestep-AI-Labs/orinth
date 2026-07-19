"use client";

import { useEffect } from "react";

// Intentionally token-free: this boundary must render even when the CSS
// bundle fails. Values mirror the hex fallbacks in frontend/DESIGN.md §2 and
// must be re-synced by hand whenever those tokens change.
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
          background: "#f5f5f5",
          color: "#171717"
        }}
      >
        <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
          <div
            style={{
              maxWidth: 420,
              width: "100%",
              border: "1px solid #e5e5e5",
              borderRadius: 16,
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
                  background: "#fee2e2",
                  color: "#b91c1c",
                  alignItems: "center",
                  justifyContent: "center",
                  fontWeight: 600
                }}
                aria-hidden="true"
              >
                !
              </span>
              <h1 style={{ margin: 0, fontSize: 20, fontWeight: 500, letterSpacing: "-0.02em" }}>Application error</h1>
            </div>
            <p style={{ margin: "0 0 4px", fontSize: 13, color: "#6b6b6b" }}>
              A critical error stopped the app from rendering.
            </p>
            <p style={{ margin: "8px 0 16px", fontSize: 13, color: "#b91c1c" }}>{error.message || "Unknown error"}</p>
            <button
              type="button"
              onClick={() => reset()}
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 8,
                border: "1px solid transparent",
                borderRadius: 8,
                minHeight: 40,
                padding: "0 14px",
                fontWeight: 600,
                background: "#171717",
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
