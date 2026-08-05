# Spec: macOS Desktop App (Tauri)

## Status

Implemented

## Goal

Ship Onestep AI Platform as a signed-optional macOS `.dmg`. A user drags
**Onestep AI Platform.app** to `/Applications`, double-clicks it, and gets the
full platform — landing, projects, datasets, training, inference — in a native
window. No terminal, no `make dev`, no manually started servers.

The app is a supervisor, not a rewrite: it starts the existing FastAPI backend
and the existing Next.js server as child processes on loopback ports and points
a WKWebView at the Next server. Application code is unchanged apart from one
env-gated build flag.

## Scope

In:

- `desktop/` — a Tauri v2 application (Rust) that owns bootstrap, process
  supervision, and window lifecycle.
- A first-run bootstrap that provisions the Python 3.11 ML environment and a
  Node runtime into `~/Library/Application Support/`, with progress UI.
- A resource-prepare script that bakes the backend source, the Next standalone
  build, and a pinned `uv` binary into the app bundle.
- `.dmg` packaging via the Tauri bundler.
- `output: "standalone"` and `images.unoptimized` in `frontend/next.config.mjs`,
  both gated on `DESKTOP_BUILD=1` so `pnpm build` / `make build` behavior is
  unchanged.
- The desktop window opens on `/signin`, not the marketing landing page.

Out:

- Windows and Linux bundles. This is not a build-target flag: the data root
  (`~/Library/Application Support`), the Node and uv asset names and archive
  formats, `.venv/bin/python` vs `Scripts\python.exe`, `/usr/bin/tar`, `/bin/ps`,
  and process-group teardown via `libc::kill` are all Unix- or macOS-specific.
  A Tauri Windows bundle also cannot be cross-compiled from macOS — it needs the
  MSVC toolchain, the Windows SDK, and WebView2 — so it must be built on Windows
  or a `windows-latest` CI runner.
- Apple Developer ID signing and notarization. `bundle.macOS.signingIdentity`
  is `"-"`, which produces a *valid* ad-hoc signature — without it the bundle
  ships with a broken one, and a quarantined copy then reports "is damaged and
  can't be opened", which reads like a corrupt download rather than an unsigned
  app. Ad-hoc is not notarization: first launch elsewhere still needs
  `xattr -dr com.apple.quarantine` or Privacy & Security → Open Anyway
  (`desktop/INSTALL.md`). Notarization is a credentials task, not a code task.
- Shipping model weights. `models/` stays a read-only local research asset per
  `AGENTS.md`; the app provisions an empty models directory and lets the user
  point at or import weights.
- Auto-update. No update server is configured.
- The `llm` / `llm-cuda` optional dependency groups. `llama-cpp-python` builds
  from source on macOS and needs Xcode Command Line Tools; the bootstrap
  installs base dependencies only, and LLM fine-tuning/serving remains a
  developer-environment feature.

## Interfaces

- API endpoints: none added. The backend is started unmodified.
- Schemas: unchanged.
- Frontend surfaces: none added to the web app. The desktop app contributes one
  local setup screen (`desktop/ui/index.html`) shown only while bootstrapping.
- Storage/DB changes: none to the schema. The desktop app relocates the runtime
  data root by passing absolute paths as environment variables to the backend
  process (`STORAGE_DIR`, `MODELS_DIR`, `DATASETS_DIR`, `DATABASE_URL`).

### Entry surface

The app window opens directly on `/signin`. Someone who has installed and
launched a desktop app does not need the marketing landing page, so `/` is
skipped in this shell; it remains the entry route for the browser build, and
`frontend/app/(marketing)/page.tsx` is unchanged.

### Runtime layout

Bundle (read-only), under `Onestep AI Platform.app/Contents/Resources/`:

```
resources/backend/          backend source: app/, migrations/, alembic.ini,
                            pyproject.toml, uv.lock
resources/frontend.tar.gz   Next standalone output: server.js, .next/, public/
resources/sample_data/      starter datasets (trash classification + NLP)
resources/bin/uv            pinned uv binary (universal: arm64 + x86_64)
```

The frontend ships as a **tarball, not a directory**. pnpm's standalone output
is a symlink farm — `node_modules/next` points into
`node_modules/.pnpm/next@…/node_modules/next` — and Node resolves a package's
dependencies from its *real* path, so `next` only finds `styled-jsx` because it
really lives inside the `.pnpm` store. Every copy step between the build and the
user's disk destroys that: `fs.cpSync` rewrites relative symlinks to absolute
build-machine paths, and the Tauri bundler dereferences them into real
directories. Either way the server dies on boot with
`Cannot find module 'styled-jsx/package.json'`. `tar` reproduces the farm
exactly, and a single opaque file survives the bundler untouched.

User data (writable), under
`~/Library/Application Support/ai.onestep.platform/`:

```
runtime/backend/       working copy of resources/backend + its .venv
runtime/node/          extracted Node runtime
runtime/python/        uv-managed CPython 3.11
storage/               uploads, overlays, checkpoints, app.db
models/                user-supplied weights
datasets/              user-supplied datasets
logs/backend.log       backend stdout/stderr
logs/frontend.log      Next server stdout/stderr
```

### Pinned assets

| Asset | Version | Source |
| --- | --- | --- |
| Node | v22.23.2 (LTS) | `nodejs.org/dist`, SHA-256 verified, downloaded per arch at first launch |
| uv | 0.12.1 | GitHub release tarballs for both arches, SHA-256 verified, `lipo`-merged into a universal binary at build time |
| CPython | 3.11 | `uv python install`, into `runtime/python` |

## Data Flow

**Build** (`desktop/scripts/prepare-resources.mjs`, then `pnpm tauri build`):

1. `DESKTOP_BUILD=1 pnpm build` in `frontend/` emits `.next/standalone`.
2. Standalone `server.js`, `.next/static`, and `public/` are assembled and
   tarred to `desktop/src-tauri/resources/frontend.tar.gz`.
3. `backend/app`, `backend/migrations`, `alembic.ini`, `pyproject.toml`, and
   `uv.lock` are copied to `desktop/src-tauri/resources/backend/`, excluding
   `__pycache__`, `.venv`, and tests.
4. `sample_data/` is copied to `desktop/src-tauri/resources/sample_data/`.
5. Both pinned `uv` builds are downloaded, checksum-verified, and welded with
   `lipo` into one universal `desktop/src-tauri/resources/bin/uv`.
6. The Tauri bundler, targeting `universal-apple-darwin`, produces the `.app`
   and `.dmg` under
   `desktop/src-tauri/target/universal-apple-darwin/release/bundle/`.

**First launch**:

1. The setup window opens immediately and subscribes to `bootstrap://progress`.
2. Bootstrap thread: create data dirs → sync backend source from Resources into
   `runtime/backend` → download + verify + extract Node into `runtime/node` →
   `uv python install 3.11` → `uv sync --frozen --no-dev` (~2.7 GB, the long
   step) → done.
3. Supervisor: pick two free loopback ports; start
   `runtime/backend/.venv/bin/python -m uvicorn app.main:app` with the data-root
   env vars; poll `/health` until 200. Retarget the baked backend origin (see
   below), then start `runtime/node/bin/node runtime/frontend/server.js` with
   `PORT`, `HOSTNAME`, and `BACKEND_PROXY_ORIGIN`; poll `/` until 200.
4. The main window opens at `http://127.0.0.1:<frontend port>/signin`; the setup
   window closes.

### Retargeting the backend origin

Next resolves `next.config.mjs` rewrite destinations **at build time** and
serializes them into `routes-manifest.json` and `required-server-files.json`.
The standalone server reads those manifests rather than re-evaluating the
config, so setting `BACKEND_PROXY_ORIGIN` at launch does nothing for the `/api`
and `/media` rewrites — the compiled-in default (`127.0.0.1:8000`) wins.

That is not a theoretical problem. Any other service on port 8000 silently
receives the app's API traffic; a second FastAPI project there answers with its
own `{"detail":"Not Found"}`, so the UI reports "Not Found" on every action
while both of this app's servers are perfectly healthy.

The desktop build therefore bakes the unroutable placeholder
`http://onestep-backend.invalid` (RFC 2606 guarantees it can never resolve, so a
missed substitution fails loudly rather than reaching a real host). Before
starting Next, `src-tauri/src/origin.rs` rewrites that string to the real
loopback origin across `.next/**/*.{json,js}`, and records what it wrote in a
marker so the next launch — which gets a different ephemeral port — knows what
to replace. A fresh unpack resets the marker to the placeholder. Finding zero
occurrences is a hard error, not a warning.

The runtime `BACKEND_PROXY_ORIGIN` env var is still set: the chat SSE route
handler reads it at request time and is unaffected by the manifest issue.

**Subsequent launches**: every bootstrap step is idempotent and short-circuits
on a marker file recording the provisioned versions, so startup is step 3
onward — a few seconds.

**Shutdown**: on window close / app exit, both children get `SIGTERM` to their
process group, then `SIGKILL` after a grace period. The backend's FastAPI
lifespan already reaps orphaned llama.cpp servers and drains its executors.

## Edge Cases

- **No network on first run.** Node/uv/PyPI downloads fail; the setup screen
  shows the failing step, the underlying error, and a Retry button. Nothing is
  half-written: each step writes to a temp path and renames on success.
- **Interrupted first run.** Version-marker files are only written after a step
  fully succeeds, so a killed install re-runs that step rather than booting a
  partial environment.
- **Port already in use.** Ports are chosen by binding `127.0.0.1:0` and
  reading back the assigned port, so a running `make dev` on 3000/8000 does not
  collide.
- **Backend fails to start.** The health poll times out (120 s), and the setup
  screen surfaces the last 50 lines of `logs/backend.log` instead of a blank
  window.
- **Stale child processes.** Each spawn records the child's PID *and its
  process start time*. On startup, a recorded PID is signalled only when its
  current start time still matches, which detects PID reuse without needing to
  identify the process any other way. Command-line matching does not work here:
  Next renames its own process title to `next-server` as it boots, so a
  command check would never fire for the frontend and would leak an orphaned
  server after every crash. Reading the child's environment is not an option
  either — macOS restricts `ps -E`. Anything unverifiable is left alone, since
  signalling a stranger's process is worse than leaving a stale server.
- **A second app instance.** The reap is keyed to the data root, so launching a
  second copy terminates the first copy's servers (last launch wins) rather
  than running two backends against one SQLite file.
- **Gatekeeper quarantine.** Documented in `README.md`; an unsigned bundle
  needs one right-click → Open.
- **Intel vs Apple silicon.** Node and uv asset names and checksums are
  selected per `std::env::consts::ARCH`; the bundle itself is built for the
  host arch.
- **Missing model weights.** `models/` starts empty. Inference surfaces the
  backend's existing "no model" errors; no desktop-specific handling.

## Acceptance Criteria

- Behavior:
  - `.dmg` mounts, shows the app plus an `/Applications` alias, and installs by
    drag-and-drop.
  - First launch provisions the environment with visible per-step progress and
    reaches the working app without any terminal interaction.
  - Second launch reaches the working app in under ~10 s.
  - Quitting the app leaves no `uvicorn`, `node`, or `llama` process behind.
  - Data written in the app appears under
    `~/Library/Application Support/ai.onestep.platform/storage/`, never in the
    repo checkout.
- Tests:
  - `cd desktop/src-tauri && cargo test` covers port selection, checksum
    verification, marker short-circuiting, symlink preservation through the
    frontend archive, origin retargeting across relaunches, and PID
    verification by start time.
  - Existing backend and frontend suites still pass — the desktop app must not
    change their behavior.
- Manual checks:
  - `DESKTOP_BUILD=1 pnpm build` produces `.next/standalone/server.js`.
  - `pnpm build` without the flag produces the unchanged non-standalone output.
  - Creating a project from the app window succeeds (the end-to-end proof that
    `/api` reaches *this* app's backend and not port 8000).

## Build Environment

`bundle_dmg.sh` calls `head -1` internally. libwww-perl ships a `head` that
takes URLs instead of line counts, and anything bundling libwww-perl (XAMPP,
for one) can shadow `/usr/bin/head` on `PATH`. When it does, DMG creation fails
with `unable to proceed with final disk image creation because the interstitial
disk image was not found` and leaves read-write disk images mounted under
`/Volumes`. `make desktop` prepends `/usr/bin:/bin` to `PATH` so the build is
reproducible regardless of what else is installed.

Rust ≥ 1.88 is required by Tauri's dependency tree (`rustup update stable`).

## Validation

```bash
cd backend && uv run pytest
cd frontend && pnpm typecheck
cd frontend && pnpm lint
cd frontend && pnpm build
cd desktop/src-tauri && cargo test
make desktop           # prepare resources + build the .dmg
```
