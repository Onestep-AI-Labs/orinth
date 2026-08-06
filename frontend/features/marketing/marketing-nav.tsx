"use client";

import Image from "next/image";
import Link from "next/link";
import { ButtonLink } from "@/features/platform/ui";
import { ModelSearch } from "./model-search";
import { MODELS_HREF, DOCS_HREF, SIGN_IN_HREF } from "./routes";

const DOWNLOAD_HREF = "https://github.com/L007/onestep-ai-platform/archive/refs/heads/main.zip";

export function MarketingNav() {
  return (
    <header className="landing-nav">
      <div className="landing-nav-left">
        <Link className="landing-wordmark" href="/">
          <Image src="/brand/logo_transparent.png" alt="Onestep AI Logo" width={30} height={30} priority />
          <strong>Onestep AI Platform</strong>
        </Link>

        <nav className="landing-nav-links" aria-label="Main Navigation">
          <Link className="landing-nav-link" href={MODELS_HREF}>
            Models
          </Link>
          <Link className="landing-nav-link" href={DOCS_HREF}>
            Documentations
          </Link>
        </nav>
      </div>

      <div className="landing-nav-center">
        <ModelSearch />
      </div>

      <div className="landing-nav-actions">
        <ButtonLink variant="secondary" href={SIGN_IN_HREF} size="sm">
          Sign in
        </ButtonLink>
        <ButtonLink href={DOWNLOAD_HREF} size="sm" target="_blank" rel="noreferrer">
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ marginRight: "6px", display: "inline-block", verticalAlign: "middle" }}
          >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Download
        </ButtonLink>
      </div>
    </header>
  );
}
