//! Child-process supervision for the backend and the Next server.
//!
//! Both servers are ordinary child processes bound to loopback. They are put in
//! their own process groups so shutdown can signal the whole tree — the backend
//! in particular spawns llama.cpp servers, and a bare `kill(pid)` would orphan
//! them.

use std::io::{Read, Seek, SeekFrom};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use crate::bootstrap::{Reporter, StepStatus};
use crate::origin;
use crate::paths::AppPaths;
use crate::ports;

/// Set on both children as a diagnostic marker only. It is deliberately *not*
/// used to prove ownership at reap time: macOS restricts `ps -E`, so a parent
/// cannot read a child's environment back.
const OWNER_TAG: &str = "orinth-desktop-child";

const HEALTH_TIMEOUT: Duration = Duration::from_secs(180);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(8);

pub struct Servers {
    pub frontend_port: u16,
    backend: Option<Child>,
    frontend: Option<Child>,
    paths: AppPaths,
}

/// Path of the PID file recording a child across runs.
fn pid_file(paths: &AppPaths, name: &str) -> PathBuf {
    paths.runtime.join(name)
}

impl Servers {
    /// Entry URL for the app window.
    ///
    /// The desktop app opens on `/signin` rather than the marketing landing
    /// page: someone who has already installed and launched the app does not
    /// need to be sold it. `/` remains reachable in the browser build.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/signin", self.frontend_port)
    }

    /// Signal both children and wait for them to exit. Safe to call twice.
    pub fn shutdown(&mut self) {
        // Frontend first: it proxies to the backend, and tearing it down first
        // avoids a burst of 502s in the webview during quit.
        for (child, name) in [
            (self.frontend.take(), "frontend.pid"),
            (self.backend.take(), "backend.pid"),
        ] {
            if let Some(mut child) = child {
                terminate(&mut child);
            }
            let _ = std::fs::remove_file(pid_file(&self.paths, name));
        }
    }
}

impl Drop for Servers {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Start both servers and wait until each answers HTTP.
pub fn start(paths: &AppPaths, reporter: &dyn Reporter) -> Result<Servers> {
    reap_stale_children(paths);

    let (backend_port, frontend_port) = ports::free_port_pair()?;

    reporter.step("backend", "Starting backend", StepStatus::Running, None);
    let backend = spawn_backend(paths, backend_port)
        .context("failed to start the backend process")
        .inspect_err(|e| {
            reporter.step(
                "backend",
                "Starting backend",
                StepStatus::Failed,
                Some(format!("{e:#}")),
            )
        })?;
    let mut servers = Servers {
        frontend_port,
        backend: Some(backend),
        frontend: None,
        paths: paths.clone(),
    };

    if let Err(error) = wait_for_http(
        &format!("http://127.0.0.1:{backend_port}/health"),
        servers.backend.as_mut().unwrap(),
        &paths.backend_log(),
    ) {
        reporter.step(
            "backend",
            "Starting backend",
            StepStatus::Failed,
            Some(format!("{error:#}")),
        );
        return Err(error);
    }
    reporter.step("backend", "Starting backend", StepStatus::Done, None);

    reporter.step("frontend", "Starting interface", StepStatus::Running, None);
    // Must happen after the backend's port is known and before Next boots: the
    // `/api` and `/media` rewrites are baked into the build and cannot be
    // steered by an environment variable.
    match origin::retarget(paths, &format!("http://127.0.0.1:{backend_port}")) {
        Ok(count) => reporter.log(&format!("Pointed {count} manifest(s) at the backend port")),
        Err(error) => {
            reporter.step(
                "frontend",
                "Starting interface",
                StepStatus::Failed,
                Some(format!("{error:#}")),
            );
            return Err(error);
        }
    }

    let frontend = spawn_frontend(paths, frontend_port, backend_port)
        .context("failed to start the interface process")
        .inspect_err(|e| {
            reporter.step(
                "frontend",
                "Starting interface",
                StepStatus::Failed,
                Some(format!("{e:#}")),
            )
        })?;
    servers.frontend = Some(frontend);

    if let Err(error) = wait_for_http(
        &format!("http://127.0.0.1:{frontend_port}/"),
        servers.frontend.as_mut().unwrap(),
        &paths.frontend_log(),
    ) {
        reporter.step(
            "frontend",
            "Starting interface",
            StepStatus::Failed,
            Some(format!("{error:#}")),
        );
        return Err(error);
    }
    reporter.step("frontend", "Starting interface", StepStatus::Done, None);

    Ok(servers)
}

fn spawn_backend(paths: &AppPaths, port: u16) -> Result<Child> {
    let python = paths.venv_python();
    if !python.exists() {
        bail!("the Python environment is missing at {}", python.display());
    }

    let mut command = Command::new(&python);
    command
        .current_dir(&paths.backend)
        .args([
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
        ])
        .arg(port.to_string())
        // `python -m` already puts the cwd on sys.path; this makes the import
        // root explicit rather than incidental.
        .args(["--app-dir", "."])
        .env("ORINTH_DESKTOP", OWNER_TAG)
        // Absolute data paths keep every artifact in Application Support. These
        // win over the backend's `.env` lookup: pydantic-settings reads the
        // process environment first.
        .env("STORAGE_DIR", &paths.storage)
        .env("MODELS_DIR", &paths.models)
        .env("DATASETS_DIR", &paths.datasets)
        // Without this the backend looks for starter datasets relative to a
        // repo root that does not exist in a packaged build, and silently
        // presents an empty catalog.
        .env("SAMPLE_DATA_DIR", &paths.sample_data)
        .env("DATABASE_URL", paths.database_url())
        .env("PYTHONUNBUFFERED", "1");

    spawn_logged(command, &paths.backend_log(), &pid_file(paths, "backend.pid"))
}

fn spawn_frontend(paths: &AppPaths, port: u16, backend_port: u16) -> Result<Child> {
    let node = paths.node_bin();
    let server = paths.frontend_server();
    if !node.exists() {
        bail!("the Node runtime is missing at {}", node.display());
    }
    if !server.exists() {
        bail!("the interface build is missing at {}", server.display());
    }

    let mut command = Command::new(&node);
    command
        .current_dir(&paths.frontend)
        .arg(&server)
        .env("ORINTH_DESKTOP", OWNER_TAG)
        .env("NODE_ENV", "production")
        .env("HOSTNAME", "127.0.0.1")
        .env("PORT", port.to_string())
        // Drives both the `/api` + `/media` rewrites and the chat SSE proxy
        // route, which otherwise default to port 8000.
        .env("BACKEND_PROXY_ORIGIN", format!("http://127.0.0.1:{backend_port}"));

    spawn_logged(command, &paths.frontend_log(), &pid_file(paths, "frontend.pid"))
}

fn spawn_logged(mut command: Command, log: &Path, pid_file: &Path) -> Result<Child> {
    if let Some(parent) = log.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let out = std::fs::File::create(log)
        .with_context(|| format!("failed to open {}", log.display()))?;
    let err = out.try_clone()?;

    command.stdin(Stdio::null()).stdout(out).stderr(err);
    // Own process group, so shutdown can signal the child *and* anything it
    // spawned (llama.cpp servers, training workers) in one call.
    command.process_group(0);

    let child = command.spawn()?;
    // PID plus the process's start time. The PID alone is not proof of
    // identity — PIDs are recycled — and the start time makes recycling
    // detectable without needing to inspect the process's command or
    // environment, neither of which is reliable here (see `owns_process`).
    let record = match process_start_token(child.id() as i32) {
        Some(token) => format!("{}\n{token}", child.id()),
        None => child.id().to_string(),
    };
    let _ = std::fs::write(pid_file, record);
    Ok(child)
}

/// An opaque per-process start-time string from `ps`.
///
/// The value is never parsed, only compared byte-for-byte against what was
/// recorded at spawn, so its locale-dependent formatting does not matter. If
/// the format ever shifts between a spawn and a reap the comparison simply
/// fails, and the safe branch is taken: leave the process alone.
fn process_start_token(pid: i32) -> Option<String> {
    let output = Command::new("/bin/ps")
        .args(["-o", "lstart=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!token.is_empty()).then_some(token)
}

/// Poll `url` until it answers, failing fast if the child dies first.
fn wait_for_http(url: &str, child: &mut Child, log: &Path) -> Result<()> {
    let deadline = Instant::now() + HEALTH_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait()? {
            bail!(
                "the process exited with {} before it started serving.\n\n{}",
                status.code().map(|c| c.to_string()).unwrap_or_else(|| "a signal".into()),
                log_tail(log, 50)
            );
        }

        let responded = ureq::get(url)
            .timeout(Duration::from_secs(3))
            .call()
            .map(|_| true)
            // A 4xx/5xx still proves the server is listening and routing.
            .unwrap_or_else(|error| matches!(error, ureq::Error::Status(_, _)));
        if responded {
            return Ok(());
        }

        if Instant::now() >= deadline {
            bail!(
                "timed out after {}s waiting for {url}.\n\n{}",
                HEALTH_TIMEOUT.as_secs(),
                log_tail(log, 50)
            );
        }
        std::thread::sleep(Duration::from_millis(300));
    }
}

/// Last `lines` lines of a log file, for surfacing a startup failure in the UI
/// instead of leaving the user with a blank window.
pub fn log_tail(path: &Path, lines: usize) -> String {
    const MAX_BYTES: u64 = 64 * 1024;
    let Ok(mut file) = std::fs::File::open(path) else {
        return format!("(no log at {})", path.display());
    };
    let Ok(size) = file.metadata().map(|m| m.len()) else {
        return String::new();
    };
    let start = size.saturating_sub(MAX_BYTES);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return String::new();
    }
    let mut buffer = String::new();
    if file.read_to_string(&mut buffer).is_err() {
        // Logs are usually UTF-8; a truncated multibyte read is not worth
        // failing over.
        return String::new();
    }
    let collected: Vec<&str> = buffer.lines().rev().take(lines).collect();
    collected.into_iter().rev().collect::<Vec<_>>().join("\n")
}

/// Terminate a child's process group: SIGTERM, then SIGKILL after a grace
/// period so a wedged server cannot block app quit.
fn terminate(child: &mut Child) {
    let pid = child.id() as i32;
    unsafe {
        libc::kill(-pid, libc::SIGTERM);
    }

    let deadline = Instant::now() + SHUTDOWN_GRACE;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) => {}
            Err(_) => return,
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    unsafe {
        libc::kill(-pid, libc::SIGKILL);
    }
    let _ = child.wait();
}

/// Kill servers left behind by a previous run that died without unwinding
/// (a crash, or a `kill` that bypassed the app's own shutdown path).
///
/// Only a PID whose recorded start time still matches is signalled. PIDs are
/// recycled, and killing a stranger's process would be far worse than leaving
/// a stale server running, so anything unverifiable is left alone.
fn reap_stale_children(paths: &AppPaths) {
    for name in ["backend.pid", "frontend.pid"] {
        let pid_file = pid_file(paths, name);
        let Ok(contents) = std::fs::read_to_string(&pid_file) else {
            continue;
        };
        if let Some(pid) = verified_pid(&contents) {
            unsafe {
                libc::kill(-pid, libc::SIGTERM);
            }
        }
        let _ = std::fs::remove_file(&pid_file);
    }
}

/// Parse a PID file and confirm the process is still the one we spawned.
///
/// Identity is proven by start time rather than by the process's command,
/// because neither is available in the general case: Next renames its own
/// process title to `next-server` the moment it boots, so a command match
/// would never fire for the frontend, and macOS restricts `ps -E` so the
/// environment cannot be read back either.
fn verified_pid(contents: &str) -> Option<i32> {
    let mut lines = contents.lines();
    let pid: i32 = lines.next()?.trim().parse().ok()?;
    if pid <= 1 {
        return None;
    }
    // A record without a start token predates this format (or `ps` failed at
    // spawn); unverifiable, so decline to signal it.
    let recorded = lines.next()?.trim();
    let current = process_start_token(pid)?;
    (current == recorded).then_some(pid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("orinth-sup-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn log_tail_returns_the_last_lines_in_order() {
        let dir = temp_root("tail");
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("backend.log");
        let body: String = (1..=200).map(|i| format!("line {i}\n")).collect();
        std::fs::write(&log, body).unwrap();

        let tail = log_tail(&log, 3);
        assert_eq!(tail, "line 198\nline 199\nline 200");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn log_tail_is_explicit_when_the_log_is_absent() {
        let missing = temp_root("missing").join("nope.log");
        assert!(log_tail(&missing, 10).contains("no log at"));
    }

    #[test]
    fn spawn_backend_fails_clearly_without_a_python_environment() {
        let paths = AppPaths::rooted(temp_root("nopython"));
        paths.ensure().unwrap();
        let error = spawn_backend(&paths, 12345).unwrap_err().to_string();
        assert!(error.contains("Python environment is missing"), "got: {error}");

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn spawn_frontend_fails_clearly_without_a_node_runtime() {
        let paths = AppPaths::rooted(temp_root("nonode"));
        paths.ensure().unwrap();
        let error = spawn_frontend(&paths, 12345, 12346).unwrap_err().to_string();
        assert!(error.contains("Node runtime is missing"), "got: {error}");

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn wait_for_http_reports_the_log_tail_when_the_child_dies() {
        let paths = AppPaths::rooted(temp_root("died"));
        paths.ensure().unwrap();
        let log = paths.backend_log();
        std::fs::write(&log, "Traceback (most recent call last):\nImportError: boom\n").unwrap();

        let mut child = Command::new("/usr/bin/false").spawn().unwrap();
        let error = wait_for_http("http://127.0.0.1:1/health", &mut child, &log)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ImportError: boom"), "got: {error}");

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn reap_stale_children_removes_pid_files_it_cannot_attribute() {
        let paths = AppPaths::rooted(temp_root("reap"));
        paths.ensure().unwrap();
        let pid_file = pid_file(&paths, "backend.pid");
        // PID 1 (launchd) is alive but is emphatically not ours.
        std::fs::write(&pid_file, "1").unwrap();

        reap_stale_children(&paths);
        assert!(!pid_file.exists());

        let _ = std::fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn verified_pid_accepts_a_live_process_with_a_matching_start_time() {
        // `ps` against our own process is the one identity we can assert.
        let pid = std::process::id() as i32;
        let token = process_start_token(pid).expect("ps must report our own start time");

        assert_eq!(verified_pid(&format!("{pid}\n{token}")), Some(pid));
    }

    #[test]
    fn verified_pid_rejects_a_recycled_pid() {
        let pid = std::process::id() as i32;
        // Same PID, different start time: a recycled PID must not be signalled.
        assert_eq!(verified_pid(&format!("{pid}\nWed Jan  1 00:00:00 2020")), None);
    }

    #[test]
    fn verified_pid_rejects_records_without_a_start_token() {
        let pid = std::process::id() as i32;
        // Bare-PID records are unverifiable and must be declined, not trusted.
        assert_eq!(verified_pid(&pid.to_string()), None);
        assert_eq!(verified_pid("1\nWed Jan  1 00:00:00 2020"), None);
        assert_eq!(verified_pid("not-a-pid"), None);
    }

    /// Regression guard for the real defect: Next renames its process title to
    /// `next-server`, so any ownership check based on the command string can
    /// never match the frontend child and would leak an orphan every crash.
    #[test]
    fn ownership_does_not_depend_on_the_process_command() {
        let pid = std::process::id() as i32;
        let command = Command::new("/bin/ps")
            .args(["-o", "command=", "-p", &pid.to_string()])
            .output()
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
            .unwrap_or_default();
        let token = process_start_token(pid).unwrap();

        // Verification succeeds regardless of what the command line says.
        assert_eq!(verified_pid(&format!("{pid}\n{token}")), Some(pid));
        assert!(!token.contains(&command) || command.is_empty());
    }
}
