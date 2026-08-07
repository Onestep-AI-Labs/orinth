//! Retarget the backend origin baked into the Next build.
//!
//! `next.config.mjs` resolves `BACKEND_PROXY_ORIGIN` when the app is *built*,
//! and Next serializes the resulting `/api` + `/media` rewrite destinations
//! into `routes-manifest.json` and `required-server-files.json`. The standalone
//! server reads those manifests instead of re-evaluating the config, so setting
//! the environment variable at launch does nothing for rewrites — the built-in
//! default (`127.0.0.1:8000`) wins, and every `/api` call lands on whatever
//! happens to own port 8000 on the user's machine. That is not hypothetical:
//! a second FastAPI project on 8000 answers with its own `{"detail":"Not
//! Found"}`, so the UI reports "Not Found" while both of our servers are
//! healthy.
//!
//! The desktop build therefore bakes an unroutable placeholder host, and this
//! module rewrites it to the real loopback origin once the backend's port is
//! known. The currently applied origin is recorded in a marker so the next
//! launch — which gets a different ephemeral port — knows what to replace.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::paths::AppPaths;

/// Baked in at build time by `prepare-resources.mjs`. `.invalid` is reserved by
/// RFC 2606 and can never resolve, so a missed substitution fails loudly
/// instead of silently reaching a real host.
pub const PLACEHOLDER_ORIGIN: &str = "http://orinth-backend.invalid";

const MARKER: &str = "frontend-origin";

/// Point the built manifests at `new_origin`. Returns the number of files
/// rewritten.
pub fn retarget(paths: &AppPaths, new_origin: &str) -> Result<usize> {
    let current = std::fs::read_to_string(paths.marker(MARKER))
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|_| PLACEHOLDER_ORIGIN.to_string());

    if current == new_origin {
        return Ok(0);
    }

    let next_dir = paths.frontend.join(".next");
    if !next_dir.is_dir() {
        bail!("the interface build is missing at {}", next_dir.display());
    }

    let mut targets = Vec::new();
    collect_rewritable(&next_dir, &mut targets)?;

    let mut rewritten = 0usize;
    for path in targets {
        let Ok(contents) = std::fs::read_to_string(&path) else {
            // Next also emits binary artifacts under .next; skip anything that
            // is not valid UTF-8 rather than failing the launch.
            continue;
        };
        if !contents.contains(&current) {
            continue;
        }
        std::fs::write(&path, contents.replace(&current, new_origin))
            .with_context(|| format!("failed to rewrite {}", path.display()))?;
        rewritten += 1;
    }

    if rewritten == 0 {
        bail!(
            "no occurrence of `{current}` was found in {}. The interface build was not \
             produced with BACKEND_PROXY_ORIGIN={PLACEHOLDER_ORIGIN}; rebuild with `make desktop`.",
            next_dir.display()
        );
    }

    paths.write_marker(MARKER, new_origin)?;
    Ok(rewritten)
}

/// Reset the record to the placeholder, called after a fresh unpack so the
/// next retarget replaces the right string.
pub fn reset(paths: &AppPaths) -> Result<()> {
    paths.write_marker(MARKER, PLACEHOLDER_ORIGIN)
}

/// Manifests and compiled server chunks are the only places the origin can
/// appear; static assets and source maps are not worth walking.
fn collect_rewritable(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("failed to read {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            // `static/` is client assets only; skipping it keeps this fast.
            if path.file_name().is_some_and(|name| name == "static") {
                continue;
            }
            collect_rewritable(&path, out)?;
        } else if matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("json") | Some("js")
        ) {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> AppPaths {
        let root = std::env::temp_dir().join(format!("orinth-origin-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let paths = AppPaths::rooted(root);
        paths.ensure().unwrap();

        let next = paths.frontend.join(".next");
        std::fs::create_dir_all(next.join("server").join("app")).unwrap();
        std::fs::create_dir_all(next.join("static").join("chunks")).unwrap();
        std::fs::write(
            next.join("routes-manifest.json"),
            format!(r#"{{"destination":"{PLACEHOLDER_ORIGIN}/api/:path*"}}"#),
        )
        .unwrap();
        std::fs::write(
            next.join("required-server-files.json"),
            format!(r#"{{"dest":"{PLACEHOLDER_ORIGIN}/media/:path*"}}"#),
        )
        .unwrap();
        std::fs::write(
            next.join("server").join("app").join("route.js"),
            format!(r#"const o="{PLACEHOLDER_ORIGIN}";"#),
        )
        .unwrap();
        paths
    }

    #[test]
    fn retarget_rewrites_every_manifest_that_mentions_the_placeholder() {
        let paths = fixture("basic");
        let count = retarget(&paths, "http://127.0.0.1:5555").unwrap();
        assert_eq!(count, 3, "both manifests and the server chunk must be rewritten");

        let manifest =
            std::fs::read_to_string(paths.frontend.join(".next").join("routes-manifest.json"))
                .unwrap();
        assert!(manifest.contains("http://127.0.0.1:5555/api/:path*"));
        assert!(!manifest.contains("invalid"));

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn retarget_is_repeatable_across_launches_with_new_ports() {
        let paths = fixture("repeat");
        retarget(&paths, "http://127.0.0.1:5555").unwrap();
        // A later launch gets a different ephemeral port and must still find
        // and replace the origin the previous launch wrote.
        let count = retarget(&paths, "http://127.0.0.1:6666").unwrap();
        assert!(count > 0, "a second retarget must rewrite the previous origin");

        let manifest =
            std::fs::read_to_string(paths.frontend.join(".next").join("routes-manifest.json"))
                .unwrap();
        assert!(manifest.contains("6666"));
        assert!(!manifest.contains("5555"));

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn retarget_is_a_noop_when_the_origin_is_unchanged() {
        let paths = fixture("noop");
        retarget(&paths, "http://127.0.0.1:5555").unwrap();
        assert_eq!(retarget(&paths, "http://127.0.0.1:5555").unwrap(), 0);

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn reset_restores_the_placeholder_after_a_fresh_unpack() {
        let paths = fixture("reset");
        retarget(&paths, "http://127.0.0.1:5555").unwrap();

        // Simulate an app upgrade: the tree is re-unpacked with the placeholder
        // back in place and the marker reset alongside it.
        std::fs::write(
            paths.frontend.join(".next").join("routes-manifest.json"),
            format!(r#"{{"destination":"{PLACEHOLDER_ORIGIN}/api/:path*"}}"#),
        )
        .unwrap();
        reset(&paths).unwrap();

        assert!(retarget(&paths, "http://127.0.0.1:7777").unwrap() > 0);

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn retarget_fails_loudly_when_the_build_was_not_prepared() {
        let paths = fixture("unprepared");
        // A build made without the placeholder: nothing to replace.
        std::fs::write(
            paths.frontend.join(".next").join("routes-manifest.json"),
            r#"{"destination":"http://127.0.0.1:8000/api/:path*"}"#,
        )
        .unwrap();
        std::fs::remove_file(paths.frontend.join(".next").join("required-server-files.json"))
            .unwrap();
        std::fs::remove_file(paths.frontend.join(".next").join("server").join("app").join("route.js"))
            .unwrap();

        let error = retarget(&paths, "http://127.0.0.1:5555").unwrap_err().to_string();
        assert!(error.contains("make desktop"), "unhelpful error: {error}");

        let _ = std::fs::remove_dir_all(&paths.root);
    }
}
