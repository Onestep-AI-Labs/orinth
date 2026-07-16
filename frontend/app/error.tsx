"use client";

import { useEffect } from "react";
import Link from "next/link";
import { AlertTriangle, RefreshCw } from "lucide-react";

export default function Error({
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
    <div className="space-y-5">
      <header className="page-header">
        <div>
          <span>
            <AlertTriangle size={22} />
          </span>
          <div>
            <h2>Something went wrong</h2>
            <p>An unexpected error interrupted this page.</p>
          </div>
        </div>
      </header>
      <section className="panel">
        <p className="error-text">{error.message || "Unknown error"}</p>
        {error.digest && <p className="text-sm text-slate-500">Reference: {error.digest}</p>}
        <div className="flex items-center gap-3" style={{ marginTop: 16 }}>
          <button className="primary-button" type="button" onClick={reset}>
            <RefreshCw size={16} /> Try again
          </button>
          <Link className="secondary-button" href="/">
            Back to dashboard
          </Link>
        </div>
      </section>
    </div>
  );
}
