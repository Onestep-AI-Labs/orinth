"use client";

import Link from "next/link";

/**
 * The two faces of the Models section: the catalog of trained/uploaded weights,
 * and the architectures you compose yourself.
 *
 * A segmented control rather than a second sidebar entry — architectures are
 * models, and splitting them into a top-level nav item would imply otherwise.
 */
export function ModelsTabs({ active }: { active: "catalog" | "architectures" }) {
  return (
    <nav className="segmented-control models-tabs" aria-label="Models sections">
      <Link
        href="/models"
        className={active === "catalog" ? "segmented-active" : ""}
        aria-current={active === "catalog" ? "page" : undefined}
      >
        Catalog
      </Link>
      <Link
        href="/models/architectures"
        className={active === "architectures" ? "segmented-active" : ""}
        aria-current={active === "architectures" ? "page" : undefined}
      >
        Architectures
      </Link>
    </nav>
  );
}
