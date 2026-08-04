# Desktop app (macOS)

Packages the platform as `Onestep AI Platform.app` and an installable `.dmg`.
Drag it to `/Applications`, double-click, and the full app opens in a native
window — no terminal, no `make dev`.

Design and rationale: [`specs/phase-18-macos-desktop-app.md`](../specs/phase-18-macos-desktop-app.md).

## What it actually is

A supervisor, not a rewrite. The app starts the same two servers `make dev`
does — FastAPI and the Next.js server — on ephemeral loopback ports, then
points a WKWebView at the Next server. The web app is unchanged; the only
frontend edit is an env-gated `output: "standalone"` in `next.config.mjs`.

```
Onestep AI Platform.app
└── Contents/Resources/resources/
    ├── backend/    FastAPI source + pyproject.toml + uv.lock
    ├── frontend/   Next standalone server
    └── bin/uv      pinned uv binary
```

The `.dmg` is ~35 MB because it ships source, not runtimes. On first launch the
app provisions the heavy parts into `~/Library/Application Support/ai.onestep.platform/`:

```
runtime/backend/.venv   ~2.7 GB of ML wheels (torch, TensorFlow, Ultralytics, Transformers)
runtime/node/           Node v22.23.2
runtime/python/         uv-managed CPython 3.11
storage/  models/  datasets/  logs/
```

First launch therefore needs a network connection and takes roughly 10–20
minutes, with per-step progress on screen. Later launches skip every completed
step and open in a few seconds.

**All user data lives under Application Support, never in the repo checkout.**
The backend is started with absolute `STORAGE_DIR` / `MODELS_DIR` /
`DATASETS_DIR` / `DATABASE_URL` values, which take precedence over its `.env`.

## Build

```bash
make desktop     # stage resources + build the .dmg
make desktop-dev # run the shell against a freshly staged bundle
make desktop-test
```

Output: `desktop/src-tauri/target/release/bundle/dmg/Onestep AI Platform_<version>_<arch>.dmg`

Requires Rust ≥ 1.88 (`rustup update stable`), Xcode Command Line Tools, Node,
and pnpm.

### Why the Makefile rewrites PATH

`make desktop` prepends `/usr/bin:/bin` to `PATH`. Tauri's `bundle_dmg.sh` calls
`head -1` internally; libwww-perl installs a `head` that takes URLs instead of
line counts, and anything shipping libwww-perl (XAMPP, for one) can shadow the
system binary. When that happens the DMG step fails with the thoroughly
unhelpful `unable to proceed with final disk image creation because the
interstitial disk image was not found`, and leaves read-write `.dmg` images
mounted under `/Volumes`. If you hit that, `hdiutil detach` them and build via
`make desktop`.

## Signing

The bundle is **ad-hoc signed**. On a machine that did not build it, Gatekeeper
will refuse the first launch; open it once with right-click → Open, or:

```bash
xattr -dr com.apple.quarantine "/Applications/Onestep AI Platform.app"
```

Distributing without that step needs a Developer ID certificate and
notarization — see the spec's Scope section for what that would involve.

## Layout

```
desktop/
├── package.json                    tauri CLI + build scripts
├── scripts/prepare-resources.mjs   stages backend, frontend, and uv
├── ui/index.html                   first-run setup screen
└── src-tauri/
    ├── tauri.conf.json
    ├── capabilities/default.json
    └── src/
        ├── lib.rs         app wiring, windows, Tauri commands
        ├── bootstrap.rs   provisioning steps + marker short-circuits
        ├── supervisor.rs  child processes, health waits, teardown
        ├── download.rs    pinned assets, checksum verification
        ├── paths.rs       bundle vs. data-root layout
        └── ports.rs       ephemeral loopback ports
```

## Troubleshooting

Logs are at `~/Library/Application Support/ai.onestep.platform/logs/`
(`backend.log`, `frontend.log`); the setup window's **Open logs** button reveals
them in Finder. A failed step shows the error plus the last 50 log lines, and
**Retry** resumes from the step that failed rather than starting over.

To force a clean reprovision, delete the marker for the step you want re-run
(or all of `runtime/markers/`). Deleting `runtime/` entirely also works but
re-downloads the full 2.7 GB.

## Known limitations

- macOS only, built for the host architecture.
- The `llm` extra (`llama-cpp-python`, PEFT/TRL) is **not** installed: it builds
  from source and needs Xcode Command Line Tools. LLM fine-tuning and serving
  remain developer-environment features.
- `models/` starts empty. The repo's trained weights are gitignored research
  assets and are deliberately not bundled.
- No auto-update.
