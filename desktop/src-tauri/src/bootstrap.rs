//! First-run provisioning.
//!
//! The `.dmg` stays small by shipping only source and a pinned `uv`; the heavy
//! parts (CPython 3.11, ~2.7 GB of ML wheels, a Node runtime) are materialized
//! into the user's data root on first launch. Every step is idempotent and
//! guarded by a marker file that is written *only* after the step fully
//! succeeds, so a killed or offline first run resumes instead of booting a
//! half-provisioned environment.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};

use crate::download::{self, NODE_VERSION};
use crate::paths::{replace_dir, AppPaths};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    Running,
    Done,
    Skipped,
    Failed,
}

impl StepStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            StepStatus::Running => "running",
            StepStatus::Done => "done",
            StepStatus::Skipped => "skipped",
            StepStatus::Failed => "failed",
        }
    }
}

/// Progress sink. Kept free of Tauri types so the bootstrap logic stays
/// testable without an app handle.
pub trait Reporter: Send + Sync {
    fn step(&self, id: &str, label: &str, status: StepStatus, detail: Option<String>);
    fn log(&self, line: &str);
}

/// Ordered step list, shared with the setup UI so it can render the full
/// checklist up front rather than growing it one row at a time.
pub const STEPS: &[(&str, &str)] = &[
    ("dirs", "Preparing workspace"),
    ("sources", "Unpacking application"),
    ("node", "Installing Node runtime"),
    ("python", "Installing Python 3.11"),
    ("deps", "Installing ML dependencies"),
    ("backend", "Starting backend"),
    ("frontend", "Starting interface"),
];

/// Paths to the read-only payload inside the app bundle.
#[derive(Debug, Clone)]
pub struct BundleResources {
    pub backend: PathBuf,
    /// Tarball, not a directory — see [`sync_sources`].
    pub frontend_archive: PathBuf,
    pub uv: PathBuf,
}

impl BundleResources {
    pub fn from_resource_dir(resource_dir: &Path) -> Self {
        let base = resource_dir.join("resources");
        Self {
            backend: base.join("backend"),
            frontend_archive: base.join("frontend.tar.gz"),
            uv: base.join("bin").join("uv"),
        }
    }

    pub fn verify(&self) -> Result<()> {
        for (label, path) in [
            ("backend source", &self.backend),
            ("frontend build", &self.frontend_archive),
            ("uv binary", &self.uv),
        ] {
            if !path.exists() {
                bail!(
                    "bundled {label} is missing at {}. The app bundle is incomplete — \
                     rebuild it with `make desktop`.",
                    path.display()
                );
            }
        }
        Ok(())
    }

    /// Content digest of the shipped payload, used to decide whether the
    /// working copy is stale.
    pub fn fingerprint(&self) -> Result<String> {
        let backend = download::dir_sha256(&self.backend)
            .context("failed to fingerprint the bundled backend source")?;
        let frontend = download::file_sha256(&self.frontend_archive)
            .context("failed to fingerprint the bundled interface build")?;
        Ok(format!("{}-{backend}-{frontend}", app_version()))
    }
}

/// Run every provisioning step. Returns once the environment is ready to have
/// servers started against it.
pub fn provision(paths: &AppPaths, bundle: &BundleResources, reporter: &dyn Reporter) -> Result<()> {
    bundle.verify()?;

    run_step(reporter, "dirs", |_| {
        paths.ensure()?;
        Ok(StepStatus::Done)
    })?;

    run_step(reporter, "sources", |r| sync_sources(paths, bundle, r))?;
    run_step(reporter, "node", |r| install_node(paths, r))?;
    run_step(reporter, "python", |r| install_python(paths, bundle, r))?;
    run_step(reporter, "deps", |r| install_dependencies(paths, bundle, r))?;

    Ok(())
}

fn label_for(id: &str) -> &'static str {
    STEPS
        .iter()
        .find(|(step_id, _)| *step_id == id)
        .map(|(_, label)| *label)
        .unwrap_or("Working")
}

fn run_step(
    reporter: &dyn Reporter,
    id: &str,
    body: impl FnOnce(&dyn Reporter) -> Result<StepStatus>,
) -> Result<()> {
    let label = label_for(id);
    reporter.step(id, label, StepStatus::Running, None);
    match body(reporter) {
        Ok(status) => {
            reporter.step(id, label, status, None);
            Ok(())
        }
        Err(error) => {
            reporter.step(id, label, StepStatus::Failed, Some(format!("{error:#}")));
            Err(error)
        }
    }
}

/// Materialize the bundled backend source and Next build into the writable
/// runtime.
///
/// Neither can run from `Contents/Resources`: `uv sync` writes a `.venv` beside
/// `pyproject.toml`, and Next writes its fetch cache under `.next/`. Both are
/// replaced wholesale so files deleted in an app upgrade do not survive in the
/// working copy. `.venv` lives inside the backend copy, so it is preserved
/// across the replace by moving it aside.
///
/// The frontend arrives as a tarball and is unpacked with `tar` rather than
/// copied. pnpm's standalone output is a symlink farm whose structure Node's
/// module resolution depends on — `node_modules/next` must really live inside
/// `node_modules/.pnpm/next@…/node_modules/` for `next` to find `styled-jsx`.
/// Any copy that rewrites or flattens those links produces a server that dies
/// on boot with MODULE_NOT_FOUND, so the archive travels intact and `tar`
/// reconstructs it here.
fn sync_sources(
    paths: &AppPaths,
    bundle: &BundleResources,
    reporter: &dyn Reporter,
) -> Result<StepStatus> {
    // Keyed on payload content, not app version: a rebuilt bundle at an
    // unchanged version must still refresh the working copy, or a developer
    // iterating on the app silently keeps running yesterday's code.
    let fingerprint = bundle.fingerprint()?;
    if paths.marker_matches("sources", &fingerprint) {
        return Ok(StepStatus::Skipped);
    }

    reporter.log("Copying backend source…");
    let venv = paths.backend.join(".venv");
    let venv_backup = paths.runtime.join(".venv-preserved");
    let had_venv = venv.exists();
    if had_venv {
        let _ = std::fs::remove_dir_all(&venv_backup);
        std::fs::rename(&venv, &venv_backup).context("failed to preserve the existing venv")?;
    }

    let result = replace_dir(&bundle.backend, &paths.backend);

    if had_venv {
        // Restore the venv whether or not the copy succeeded — losing it would
        // force a 2.7 GB reinstall on the next launch.
        let _ = std::fs::remove_dir_all(&venv);
        std::fs::rename(&venv_backup, &venv).context("failed to restore the preserved venv")?;
    }
    result?;

    reporter.log("Unpacking interface build…");
    if paths.frontend.exists() {
        std::fs::remove_dir_all(&paths.frontend)
            .context("failed to clear the previous interface build")?;
    }
    download::extract_tar_gz(&bundle.frontend_archive, &paths.frontend)?;
    if !paths.frontend_server().exists() {
        bail!(
            "the interface archive unpacked without {}",
            paths.frontend_server().display()
        );
    }
    // The freshly unpacked build carries the placeholder origin again, so the
    // record of what to substitute has to go back with it.
    crate::origin::reset(paths)?;

    paths.write_marker("sources", &fingerprint)?;
    Ok(StepStatus::Done)
}

fn install_node(paths: &AppPaths, reporter: &dyn Reporter) -> Result<StepStatus> {
    if paths.marker_matches("node", NODE_VERSION) && paths.node_bin().exists() {
        return Ok(StepStatus::Skipped);
    }

    let asset = download::node_asset()?;
    reporter.log(&format!("Downloading Node {NODE_VERSION}…"));

    let mut last_percent = -1i64;
    download::install_asset(
        &asset,
        &paths.cache,
        &paths.node,
        |downloaded, total| {
            if let Some(total) = total {
                let percent = (downloaded as f64 / total as f64 * 100.0) as i64;
                if percent != last_percent && percent % 5 == 0 {
                    last_percent = percent;
                    reporter.log(&format!("Node runtime {percent}%"));
                }
            }
        },
    )?;

    if !paths.node_bin().exists() {
        bail!("Node install completed but {} is missing", paths.node_bin().display());
    }
    paths.write_marker("node", NODE_VERSION)?;
    Ok(StepStatus::Done)
}

fn install_python(
    paths: &AppPaths,
    bundle: &BundleResources,
    reporter: &dyn Reporter,
) -> Result<StepStatus> {
    if paths.marker_matches("python", "3.11") {
        return Ok(StepStatus::Skipped);
    }

    reporter.log("Installing a managed CPython 3.11…");
    let mut command = uv_command(paths, bundle);
    command.args(["python", "install", "3.11"]);
    run_streaming(command, reporter, "uv python install")?;

    paths.write_marker("python", "3.11")?;
    Ok(StepStatus::Done)
}

/// `uv sync` the backend project. This is the long pole: torch, TensorFlow,
/// Ultralytics and Transformers total roughly 2.7 GB.
fn install_dependencies(
    paths: &AppPaths,
    bundle: &BundleResources,
    reporter: &dyn Reporter,
) -> Result<StepStatus> {
    let lock = paths.backend.join("uv.lock");
    let manifest = paths.backend.join("pyproject.toml");
    let fingerprint = download::combined_sha256(&[manifest.as_path(), lock.as_path()])
        .context("failed to fingerprint the backend dependency manifest")?;

    if paths.marker_matches("deps", &fingerprint) && paths.venv_python().exists() {
        return Ok(StepStatus::Skipped);
    }

    reporter.log("Resolving and installing Python packages. This takes several minutes.");

    // `--frozen` installs exactly what `uv.lock` pins. If the bundled lock was
    // written by an older uv than the one we ship and cannot be read as-is,
    // fall back to a fresh resolve rather than failing the install outright.
    let mut locked = uv_command(paths, bundle);
    locked.args(["sync", "--frozen", "--no-dev"]);
    match run_streaming(locked, reporter, "uv sync --frozen") {
        Ok(()) => {}
        Err(error) => {
            reporter.log(&format!("Locked install failed ({error}); re-resolving dependencies…"));
            let mut resolved = uv_command(paths, bundle);
            resolved.args(["sync", "--no-dev"]);
            run_streaming(resolved, reporter, "uv sync")?;
        }
    }

    if !paths.venv_python().exists() {
        bail!(
            "dependency install finished but {} is missing",
            paths.venv_python().display()
        );
    }
    paths.write_marker("deps", &fingerprint)?;
    Ok(StepStatus::Done)
}

/// A `uv` invocation pinned entirely to this app's data root.
///
/// Every uv directory is redirected so the desktop app never shares state with
/// — or corrupts — a developer's own uv cache, managed pythons, or the repo
/// checkout's `.venv`. `UV_NO_CONFIG` keeps a stray user `uv.toml` from
/// changing index URLs or resolution behavior under us.
fn uv_command(paths: &AppPaths, bundle: &BundleResources) -> Command {
    let mut command = Command::new(&bundle.uv);
    command
        .current_dir(&paths.backend)
        .env("UV_PYTHON_INSTALL_DIR", &paths.python)
        .env("UV_PROJECT_ENVIRONMENT", paths.backend.join(".venv"))
        .env("UV_CACHE_DIR", paths.cache.join("uv"))
        .env("UV_PYTHON_PREFERENCE", "only-managed")
        .env("UV_NO_CONFIG", "1")
        .env("UV_NO_PROGRESS", "1");
    command
}

/// Run a command, streaming both stdout and stderr to the reporter line by
/// line so a multi-minute install shows continuous progress instead of a
/// frozen window.
fn run_streaming(mut command: Command, reporter: &dyn Reporter, what: &str) -> Result<()> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to start `{what}`"))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    std::thread::scope(|scope| {
        if let Some(stream) = stdout {
            scope.spawn(move || {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    reporter.log(&line);
                }
            });
        }
        if let Some(stream) = stderr {
            scope.spawn(move || {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    reporter.log(&line);
                }
            });
        }
    });

    let status = child
        .wait()
        .with_context(|| format!("failed to wait on `{what}`"))?;
    if !status.success() {
        return Err(anyhow!(
            "`{what}` exited with {}",
            status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "a signal".into())
        ));
    }
    Ok(())
}

pub fn app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingReporter {
        steps: Mutex<Vec<(String, String)>>,
        logs: Mutex<Vec<String>>,
    }

    impl Reporter for RecordingReporter {
        fn step(&self, id: &str, _label: &str, status: StepStatus, _detail: Option<String>) {
            self.steps
                .lock()
                .unwrap()
                .push((id.to_string(), status.as_str().to_string()));
        }
        fn log(&self, line: &str) {
            self.logs.lock().unwrap().push(line.to_string());
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("onestep-boot-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Build a stand-in for the bundle payload: backend source, a uv binary,
    /// and a frontend tarball that reproduces pnpm's symlink farm.
    fn fake_bundle(root: &Path) -> BundleResources {
        let resources = root.join("resources");
        std::fs::create_dir_all(resources.join("backend").join("app")).unwrap();
        std::fs::write(resources.join("backend").join("app").join("main.py"), b"x").unwrap();
        std::fs::create_dir_all(resources.join("bin")).unwrap();
        std::fs::write(resources.join("bin").join("uv"), b"").unwrap();

        // Mirror the real layout: server.js plus a package symlinked into a
        // .pnpm-style store, so a copy that flattens links is detectable.
        let staging = root.join("frontend-staging");
        let store = staging.join("node_modules").join(".pnpm").join("next").join("node_modules");
        std::fs::create_dir_all(store.join("next")).unwrap();
        std::fs::create_dir_all(store.join("styled-jsx")).unwrap();
        std::fs::write(staging.join("server.js"), b"// server").unwrap();
        std::os::unix::fs::symlink(
            ".pnpm/next/node_modules/next",
            staging.join("node_modules").join("next"),
        )
        .unwrap();

        let archive = resources.join("frontend.tar.gz");
        let status = std::process::Command::new("/usr/bin/tar")
            .arg("-czf")
            .arg(&archive)
            .arg("-C")
            .arg(&staging)
            .arg(".")
            .status()
            .unwrap();
        assert!(status.success());

        BundleResources::from_resource_dir(root)
    }

    #[test]
    fn every_step_id_has_a_label() {
        for (id, label) in STEPS {
            assert_eq!(label_for(id), *label);
            assert!(!label.is_empty());
        }
    }

    #[test]
    fn sync_sources_skips_unchanged_payloads_but_reruns_on_a_rebuild() {
        let paths = AppPaths::rooted(temp_root("sync-skip"));
        paths.ensure().unwrap();

        let bundle = fake_bundle(&paths.root.join("bundle"));
        bundle.verify().unwrap();

        let reporter = RecordingReporter::default();
        assert_eq!(
            sync_sources(&paths, &bundle, &reporter).unwrap(),
            StepStatus::Done
        );
        // Second run over an identical payload must not re-copy.
        assert_eq!(
            sync_sources(&paths, &bundle, &reporter).unwrap(),
            StepStatus::Skipped
        );

        // A rebuilt bundle at the same app version must still refresh: keying
        // the marker on the version alone would ship stale code here.
        std::fs::write(bundle.backend.join("app").join("main.py"), b"changed").unwrap();
        assert_eq!(
            sync_sources(&paths, &bundle, &reporter).unwrap(),
            StepStatus::Done
        );
        assert_eq!(
            std::fs::read_to_string(paths.backend.join("app").join("main.py")).unwrap(),
            "changed"
        );

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    /// Regression guard: pnpm's standalone output only resolves because
    /// `node_modules/next` is a symlink into the `.pnpm` store. A staging or
    /// unpack step that flattens it ships a frontend that cannot boot.
    #[test]
    fn unpacking_the_frontend_preserves_pnpm_symlinks() {
        let paths = AppPaths::rooted(temp_root("symlinks"));
        paths.ensure().unwrap();
        let bundle = fake_bundle(&paths.root.join("bundle"));

        let reporter = RecordingReporter::default();
        sync_sources(&paths, &bundle, &reporter).unwrap();

        let linked = paths.frontend.join("node_modules").join("next");
        let metadata = std::fs::symlink_metadata(&linked).unwrap();
        assert!(
            metadata.file_type().is_symlink(),
            "node_modules/next must stay a symlink, not be flattened into a directory"
        );
        // And the link must still resolve inside the unpacked tree.
        assert!(linked.join("..").join("styled-jsx").exists());

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn sync_sources_preserves_an_existing_venv() {
        let paths = AppPaths::rooted(temp_root("sync-venv"));
        paths.ensure().unwrap();

        let bundle = fake_bundle(&paths.root.join("bundle"));

        // Stand in for a provisioned 2.7 GB environment.
        let venv_marker = paths.backend.join(".venv").join("bin").join("python");
        std::fs::create_dir_all(venv_marker.parent().unwrap()).unwrap();
        std::fs::write(&venv_marker, b"python").unwrap();

        let reporter = RecordingReporter::default();
        sync_sources(&paths, &bundle, &reporter).unwrap();

        assert!(venv_marker.exists(), "the venv must survive a source refresh");
        assert!(paths.backend.join("app").join("main.py").exists());

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn verify_names_the_missing_resource() {
        let root = temp_root("verify");
        std::fs::create_dir_all(&root).unwrap();
        let bundle = BundleResources::from_resource_dir(&root);
        let error = bundle.verify().unwrap_err().to_string();
        assert!(error.contains("backend source"), "unhelpful error: {error}");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn failed_steps_report_failure_to_the_reporter() {
        let reporter = RecordingReporter::default();
        let result = run_step(&reporter, "node", |_| bail!("no network"));
        assert!(result.is_err());
        let steps = reporter.steps.lock().unwrap().clone();
        assert_eq!(steps.first().unwrap().1, "running");
        assert_eq!(steps.last().unwrap().1, "failed");
    }
}
