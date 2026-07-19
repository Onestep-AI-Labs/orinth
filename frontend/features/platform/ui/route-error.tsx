"use client";

import { useEffect } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button, ButtonLink } from "./primitives";

export function RouteErrorPanel({
  error,
  reset,
  area = "this page"
}: {
  error: Error & { digest?: string };
  reset: () => void;
  area?: string;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="space-y-5">
      <header className="page-header">
        <div>
          <span>
            <AlertTriangle size={22} />
          </span>
          <div>
            <h2>Something went wrong</h2>
            <p>An unexpected error interrupted {area}.</p>
          </div>
        </div>
      </header>
      <section className="panel">
        <p className="error-text">{error.message || "Unknown error"}</p>
        {error.digest && <p className="text-sm text-ink-subtle">Reference: {error.digest}</p>}
        <div className="flex items-center gap-3" style={{ marginTop: 16 }}>
          <Button onClick={reset}>
            <RefreshCw size={16} /> Try again
          </Button>
          <ButtonLink variant="secondary" href="/">
            Back to projects
          </ButtonLink>
        </div>
      </section>
    </div>
  );
}
