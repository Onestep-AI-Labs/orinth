//! Filesystem layout for the desktop app.
//!
//! Two roots matter. The *bundle* root is read-only and signed: it holds the
//! backend source, the Next standalone build, and the pinned `uv` binary that
//! ship inside the `.app`. The *data* root under `~/Library/Application
//! Support` is writable and survives app upgrades: it holds the provisioned
//! Python/Node runtimes and every artifact the platform generates.
//!
//! Nothing is ever written back into the bundle — a signed `.app` that mutates
//! itself breaks its own signature, and a user with the app in `/Applications`
//! may not even have write permission there.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

pub const APP_ID: &str = "ai.onestep.platform";

/// Writable per-user layout. Every field is an absolute path; none of them are
/// guaranteed to exist until [`AppPaths::ensure`] runs.
#[derive(Debug, Clone)]
pub struct AppPaths {
    pub root: PathBuf,
    pub runtime: PathBuf,
    /// Working copy of the bundled backend source; also hosts `.venv`.
    pub backend: PathBuf,
    /// Working copy of the bundled Next standalone build.
    pub frontend: PathBuf,
    pub node: PathBuf,
    pub python: PathBuf,
    /// Working copy of the bundled starter datasets.
    pub sample_data: PathBuf,
    pub cache: PathBuf,
    pub markers: PathBuf,
    pub storage: PathBuf,
    pub models: PathBuf,
    pub datasets: PathBuf,
    pub logs: PathBuf,
}

impl AppPaths {
    pub fn resolve() -> Result<Self> {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or_else(|| anyhow!("HOME is not set; cannot locate Application Support"))?;
        let root = home
            .join("Library")
            .join("Application Support")
            .join(APP_ID);
        Ok(Self::rooted(root))
    }

    /// Build a layout under an arbitrary root. Used by tests so they never
    /// touch the real Application Support directory.
    pub fn rooted(root: PathBuf) -> Self {
        let runtime = root.join("runtime");
        Self {
            backend: runtime.join("backend"),
            frontend: runtime.join("frontend"),
            node: runtime.join("node"),
            python: runtime.join("python"),
            sample_data: runtime.join("sample_data"),
            cache: runtime.join("cache"),
            markers: runtime.join("markers"),
            storage: root.join("storage"),
            models: root.join("models"),
            datasets: root.join("datasets"),
            logs: root.join("logs"),
            runtime,
            root,
        }
    }

    pub fn ensure(&self) -> Result<()> {
        for dir in [
            &self.root,
            &self.runtime,
            &self.backend,
            &self.node,
            &self.python,
            &self.sample_data,
            &self.cache,
            &self.markers,
            &self.storage,
            &self.models,
            &self.datasets,
            &self.logs,
        ] {
            std::fs::create_dir_all(dir)
                .map_err(|e| anyhow!("failed to create {}: {e}", dir.display()))?;
        }
        Ok(())
    }

    pub fn venv_python(&self) -> PathBuf {
        self.backend.join(".venv").join("bin").join("python")
    }

    pub fn node_bin(&self) -> PathBuf {
        self.node.join("bin").join("node")
    }

    pub fn frontend_server(&self) -> PathBuf {
        self.frontend.join("server.js")
    }

    pub fn database_url(&self) -> String {
        format!("sqlite:///{}", self.storage.join("app.db").display())
    }

    pub fn backend_log(&self) -> PathBuf {
        self.logs.join("backend.log")
    }

    pub fn frontend_log(&self) -> PathBuf {
        self.logs.join("frontend.log")
    }

    pub fn marker(&self, name: &str) -> PathBuf {
        self.markers.join(name)
    }

    /// True when the marker file exists and already records `value`. Markers
    /// are written only after a step fully succeeds, so an interrupted install
    /// re-runs its step instead of booting a half-provisioned runtime.
    pub fn marker_matches(&self, name: &str, value: &str) -> bool {
        std::fs::read_to_string(self.marker(name))
            .map(|current| current.trim() == value)
            .unwrap_or(false)
    }

    pub fn write_marker(&self, name: &str, value: &str) -> Result<()> {
        std::fs::create_dir_all(&self.markers)?;
        std::fs::write(self.marker(name), value)?;
        Ok(())
    }

}

/// Recursive directory copy. `std::fs` has no equivalent and pulling a crate in
/// for ~20 lines is not worth the dependency.
pub fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target = dst.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else if file_type.is_symlink() {
            // Node ships symlinked binaries (`bin/npm` -> `../lib/node_modules/...`).
            // Recreate the link rather than dereferencing it.
            let link = std::fs::read_link(entry.path())?;
            let _ = std::fs::remove_file(&target);
            std::os::unix::fs::symlink(link, &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// Replace `dst` with a fresh copy of `src`, so files deleted upstream do not
/// linger in the working copy across app upgrades.
pub fn replace_dir(src: &Path, dst: &Path) -> Result<()> {
    if dst.exists() {
        std::fs::remove_dir_all(dst)
            .map_err(|e| anyhow!("failed to clear {}: {e}", dst.display()))?;
    }
    copy_dir_all(src, dst)
        .map_err(|e| anyhow!("failed to copy {} to {}: {e}", src.display(), dst.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("onestep-paths-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn markers_only_match_exact_recorded_value() {
        let paths = AppPaths::rooted(temp_root("markers"));
        paths.ensure().unwrap();

        assert!(!paths.marker_matches("node", "v22.23.2"));

        paths.write_marker("node", "v22.23.2").unwrap();
        assert!(paths.marker_matches("node", "v22.23.2"));
        // A version bump must invalidate the short-circuit.
        assert!(!paths.marker_matches("node", "v22.24.0"));

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn database_url_is_absolute_and_outside_the_repo() {
        let paths = AppPaths::rooted(temp_root("db"));
        let url = paths.database_url();
        assert!(url.starts_with("sqlite:////"), "expected absolute path: {url}");
        assert!(url.ends_with("storage/app.db"), "unexpected db path: {url}");
    }

    #[test]
    fn replace_dir_drops_files_removed_upstream() {
        let base = temp_root("replace");
        let src = base.join("src");
        let dst = base.join("dst");
        std::fs::create_dir_all(src.join("nested")).unwrap();
        std::fs::write(src.join("nested").join("keep.txt"), "keep").unwrap();

        replace_dir(&src, &dst).unwrap();
        std::fs::write(dst.join("stale.txt"), "stale").unwrap();
        replace_dir(&src, &dst).unwrap();

        assert!(dst.join("nested").join("keep.txt").exists());
        assert!(!dst.join("stale.txt").exists());

        let _ = std::fs::remove_dir_all(&base);
    }
}
