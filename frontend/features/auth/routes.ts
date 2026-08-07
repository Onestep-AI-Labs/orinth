/**
 * Entry flow: sign-in -> workspace.
 *
 * `/` is the sign-in screen. There is no public landing page — an unauthenticated
 * visitor sees the form and nothing else.
 *
 * Sign-in is presentational only right now; there is no auth backend behind it
 * (see docs/ai/rules.md "Add auth before multi-user or network-exposed
 * deployment"). Both hops are named here so wiring real auth later is a change
 * to this file plus the sign-in submit handler, not a hunt through JSX.
 */
export const SIGN_IN_HREF = "/";

/** Where sign-in hands off once the visitor is through. */
export const WORKSPACE_HREF = "/projects";
