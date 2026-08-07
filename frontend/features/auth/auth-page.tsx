"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/features/platform/ui";
import { BRAND_NAME, LogoMark } from "@/components/brand";
import { WORKSPACE_HREF } from "./routes";

/**
 * Presentational sign-in, and the app's entry route. There is no auth backend
 * yet — docs/ai/rules.md is explicit that auth lands before multi-user or
 * network-exposed deployment, so this screen deliberately does not validate,
 * store, or transmit anything. The notice below says so on the page rather than
 * letting the form imply otherwise.
 *
 * Wiring real auth later: submit against the auth endpoint here, and keep the
 * redirect target in ./routes.ts.
 */
export function SignInPage() {
  const router = useRouter();

  return (
    <div className="signin">
      <div className="signin-panel">
        <div className="signin-form-wrap">
          <span className="signin-wordmark">
            <LogoMark size={22} />
            <strong>{BRAND_NAME}</strong>
          </span>

          <h1 className="signin-title">Sign in</h1>
          <p className="signin-sub">Continue to your {BRAND_NAME} workspace.</p>

          <form
            className="signin-form"
            onSubmit={(event) => {
              event.preventDefault();
              router.push(WORKSPACE_HREF);
            }}
          >
            <label className="signin-field">
              <span>Email</span>
              <input type="email" name="email" autoComplete="email" placeholder="you@lab.org" />
            </label>
            <label className="signin-field">
              <span>Password</span>
              <input
                type="password"
                name="password"
                autoComplete="current-password"
                placeholder="••••••••"
              />
            </label>
            <Button type="submit">Continue to workspace</Button>
          </form>

          <p className="signin-notice">
            Authentication is not wired up yet — this screen does not check, store, or send
            credentials. Either field can be left empty. The workspace runs locally and is
            single-user until auth ships.
          </p>
        </div>
      </div>

      <aside className="signin-aside">
        <div className="signin-aside-inner">
          <span className="signin-aside-mark">
            <LogoMark size={34} />
          </span>
          <p className="signin-aside-lede">
            One workspace that carries a dataset from raw files through to measured model behavior.
          </p>
          <ol className="signin-aside-list">
            <li>
              <b>01</b>
              <span>Organize projects across vision and NLP</span>
            </li>
            <li>
              <b>02</b>
              <span>Prepare, annotate, and version datasets</span>
            </li>
            <li>
              <b>03</b>
              <span>Train, test, and inspect the results</span>
            </li>
          </ol>
        </div>
      </aside>
    </div>
  );
}
