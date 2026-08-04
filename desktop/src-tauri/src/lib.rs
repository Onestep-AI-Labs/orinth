//! Onestep AI Platform desktop shell.
//!
//! The app owns three things: provisioning a Python/Node runtime on first
//! launch, supervising the FastAPI and Next servers, and pointing a webview at
//! the Next server. The web app itself is unchanged — this is a shell around
//! the same servers `make dev` runs.

mod bootstrap;
mod download;
mod origin;
mod paths;
mod ports;
mod supervisor;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

use bootstrap::{BundleResources, Reporter, StepStatus};
use paths::AppPaths;
use supervisor::Servers;

const SETUP_WINDOW: &str = "setup";
const MAIN_WINDOW: &str = "main";

#[derive(Default)]
struct AppState {
    servers: Mutex<Option<Servers>>,
    launching: AtomicBool,
}

#[derive(Clone, Serialize)]
struct StepEvent {
    id: String,
    label: String,
    status: String,
    detail: Option<String>,
}

#[derive(Clone, Serialize)]
struct StepDefinition {
    id: String,
    label: String,
}

#[derive(Clone, Serialize)]
struct ErrorEvent {
    message: String,
}

/// Bridges bootstrap progress onto the Tauri event bus.
struct EventReporter {
    app: AppHandle,
}

impl Reporter for EventReporter {
    fn step(&self, id: &str, label: &str, status: StepStatus, detail: Option<String>) {
        let _ = self.app.emit(
            "bootstrap:step",
            StepEvent {
                id: id.to_string(),
                label: label.to_string(),
                status: status.as_str().to_string(),
                detail,
            },
        );
    }

    fn log(&self, line: &str) {
        let trimmed = line.trim_end();
        if !trimmed.is_empty() {
            let _ = self.app.emit("bootstrap:log", trimmed.to_string());
        }
    }
}

/// The full checklist, so the setup window can render every row up front
/// instead of growing the list one step at a time.
#[tauri::command]
fn bootstrap_steps() -> Vec<StepDefinition> {
    bootstrap::STEPS
        .iter()
        .map(|(id, label)| StepDefinition {
            id: (*id).to_string(),
            label: (*label).to_string(),
        })
        .collect()
}

/// Begin (or resume) provisioning.
///
/// The setup window calls this once it has subscribed to the event bus, which
/// is why bootstrap is not kicked off from `RunEvent::Ready`: early progress
/// events fired before the webview subscribed would be dropped, leaving the
/// checklist stuck on the first row. It doubles as the Retry handler — every
/// step is idempotent, so a retry resumes at whichever step failed.
#[tauri::command]
fn start_bootstrap(app: AppHandle) {
    launch(app);
}

/// Open the log directory in Finder so a user can attach logs to a bug report.
#[tauri::command]
fn reveal_logs() -> Result<(), String> {
    let paths = AppPaths::resolve().map_err(|e| e.to_string())?;
    std::process::Command::new("/usr/bin/open")
        .arg(&paths.logs)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Provision (if needed), start both servers, then swap the setup window for
/// the real app window.
fn launch(app: AppHandle) {
    let state = app.state::<AppState>();
    // Guard against a double-click on Retry kicking off two 2.7 GB installs.
    if state.launching.swap(true, Ordering::SeqCst) {
        return;
    }

    std::thread::spawn(move || {
        let result = provision_and_start(&app);
        let state = app.state::<AppState>();
        state.launching.store(false, Ordering::SeqCst);

        match result {
            Ok(servers) => {
                let url = servers.url();
                *state.servers.lock().unwrap() = Some(servers);
                if let Err(error) = show_main_window(&app, &url) {
                    let _ = app.emit(
                        "bootstrap:error",
                        ErrorEvent {
                            message: format!("Failed to open the app window: {error:#}"),
                        },
                    );
                }
            }
            Err(error) => {
                let _ = app.emit(
                    "bootstrap:error",
                    ErrorEvent {
                        message: format!("{error:#}"),
                    },
                );
            }
        }
    });
}

fn provision_and_start(app: &AppHandle) -> anyhow::Result<Servers> {
    let paths = AppPaths::resolve()?;
    let resource_dir = app.path().resource_dir()?;
    let bundle = BundleResources::from_resource_dir(&resource_dir);
    let reporter = EventReporter { app: app.clone() };

    bootstrap::provision(&paths, &bundle, &reporter)?;
    supervisor::start(&paths, &reporter)
}

fn show_main_window(app: &AppHandle, url: &str) -> anyhow::Result<()> {
    let url = tauri::Url::parse(url)?;
    let app = app.clone();
    // Window construction belongs on the main thread on macOS.
    app.clone().run_on_main_thread(move || {
        let built = WebviewWindowBuilder::new(&app, MAIN_WINDOW, WebviewUrl::External(url))
            .title("Onestep AI Platform")
            .inner_size(1440.0, 900.0)
            .min_inner_size(1024.0, 700.0)
            .center()
            .resizable(true)
            .build();

        match built {
            Ok(_) => {
                if let Some(setup) = app.get_webview_window(SETUP_WINDOW) {
                    let _ = setup.close();
                }
            }
            Err(error) => {
                let _ = app.emit(
                    "bootstrap:error",
                    ErrorEvent {
                        message: format!("Failed to open the app window: {error}"),
                    },
                );
            }
        }
    })?;
    Ok(())
}

pub fn run() {
    let app = tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            bootstrap_steps,
            start_bootstrap,
            reveal_logs
        ])
        .setup(|app| {
            WebviewWindowBuilder::new(app, SETUP_WINDOW, WebviewUrl::App("index.html".into()))
                .title("Onestep AI Platform")
                .inner_size(680.0, 560.0)
                .resizable(false)
                .center()
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to start the Onestep AI Platform desktop app");

    // Bootstrap is kicked off by the setup window (see `start_bootstrap`), not
    // from here, so no progress event can fire before the webview subscribes.
    app.run(|app, event| match event {
        RunEvent::ExitRequested { .. } | RunEvent::Exit => {
            if let Some(mut servers) = app.state::<AppState>().servers.lock().unwrap().take() {
                servers.shutdown();
            }
        }
        _ => {}
    });
}
