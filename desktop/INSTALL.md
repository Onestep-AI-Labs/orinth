# Installing Onestep AI Platform (macOS)

The `.dmg` is **universal** — the same file runs on Apple silicon (M1–M4) and
on Intel Macs.

## Before you start

- macOS 11 (Big Sur) or newer.
- **An internet connection for the first launch.** The download is small
  (~56 MB) because the app installs its Python + machine-learning environment
  on first run — about **2.7 GB**, which takes roughly 10–20 minutes. After
  that it starts in a few seconds and works offline.
- ~5 GB of free disk space.

## Install

1. Double-click **`Onestep AI Platform_0.1.0_universal.dmg`**.
2. Drag **Onestep AI Platform** onto the **Applications** shortcut in the
   window that opens.
3. Eject the disk image (click ⏏ next to it in Finder's sidebar).

## First launch — the security prompt

The app is **not signed with an Apple Developer ID**, so macOS blocks it the
first time. This is expected for an internally distributed app, and it only
happens once.

Pick whichever route you prefer.

### Route A — Terminal (one command, most reliable)

Open **Terminal** (⌘-Space, type "Terminal") and paste:

```bash
xattr -dr com.apple.quarantine "/Applications/Onestep AI Platform.app"
```

Then open the app normally from Applications. Nothing else is needed.

This removes the "downloaded from the internet" flag that macOS attaches to
files arriving by browser, email, AirDrop, or Slack. It is the flag — not the
app — that triggers the block.

### Route B — System Settings

1. Open the app from Applications. macOS refuses and shows a warning.
2. Open **System Settings → Privacy & Security**.
3. Scroll to the **Security** section. You will see a line saying
   *"Onestep AI Platform was blocked to protect your Mac."*
4. Click **Open Anyway**, then confirm with Touch ID or your password.
5. Open the app again.

> On macOS 15 (Sequoia) and newer, right-click → **Open** no longer bypasses
> this — Apple removed that shortcut. Use Route A or B.

## First launch — what you should see

A small window appears with a checklist:

```
Preparing workspace
Unpacking application
Installing Node runtime
Installing Python 3.11
Installing ML dependencies     ← the long one, ~10–20 min
Starting backend
Starting interface
```

When it finishes, the setup window closes and the app opens on the sign-in
screen. **Leave it running during "Installing ML dependencies"** — it is
downloading PyTorch, TensorFlow, and friends.

If a step fails, the window shows the error and a **Retry** button. Retry
resumes at the failed step rather than starting over, so a dropped connection
costs you only the remainder.

## Where your data lives

Everything the app creates — datasets, models, training runs, the database —
lives in:

```
~/Library/Application Support/ai.onestep.platform/
```

Logs, if you ever need to send them:

```
~/Library/Application Support/ai.onestep.platform/logs/
```

The setup window also has an **Open logs** button.

## Uninstall

```bash
rm -rf "/Applications/Onestep AI Platform.app"
rm -rf ~/Library/Application\ Support/ai.onestep.platform
```

The second command deletes your projects and trained models too — skip it if
you want to keep them for a reinstall.

## Troubleshooting

**"The application ... can't be opened."**
This means the app's architecture does not match the Mac. Make sure the file
is the **`_universal.dmg`**, not `_aarch64.dmg` (Apple silicon only) or
`_x64.dmg` (Intel only). Check which Mac you have with  → About This Mac.

**"...is damaged and can't be opened."**
The download was truncated or the quarantine flag is set. Re-download, then
use Route A above.

**Setup fails at "Installing ML dependencies"**
Almost always the network. Click **Retry**. On a metered or filtered
connection, note that it downloads from `pypi.org` and `nodejs.org`.

**The app opens but pages do not load**
Send `backend.log` and `frontend.log` from the logs folder above.
