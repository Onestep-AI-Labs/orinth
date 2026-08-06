/**
 * Entry flow: landing -> sign-in -> workspace.
 *
 * Sign-in is presentational only right now; there is no auth backend behind it
 * (see docs/ai/rules.md "Add auth before multi-user or network-exposed
 * deployment"). Both hops are named here so wiring real auth later is a change
 * to this file plus the sign-in submit handler, not a hunt through JSX.
 */
export const LANDING_HREF = "/";

/** Where the landing page's calls to action send a visitor. */
export const SIGN_IN_HREF = "/signin";

/** Models catalog and showcase page. */
export const MODELS_HREF = "/model-catalog";

/** Documentation and platform guide page. */
export const DOCS_HREF = "/docs";

/** Where sign-in hands off once the visitor is through. */
export const WORKSPACE_HREF = "/projects";

