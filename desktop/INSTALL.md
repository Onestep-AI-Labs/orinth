# Installing Orinth (macOS)

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

1. Double-click **`Orinth_0.1.0_universal.dmg`**.
2. Drag **Orinth** onto the **Applications** shortcut in the
   window that opens.
3. Eject the disk image (click ⏏ next to it in Finder's sidebar).

## First launch — the security prompt

The app is **not signed with an Apple Developer ID**, so macOS blocks it the
first time. This is expected for an internally distributed app, and it only
happens once.

### Route A — System Settings (no typing, recommended)

1. Open the app from Applications. macOS refuses and shows a warning. Click
   **Done** / **OK**.
2. Open **System Settings → Privacy & Security**.
3. Scroll down to the **Security** section. You will see a line saying
   *"Orinth was blocked to protect your Mac."*
4. Click **Open Anyway**, then confirm with Touch ID or your password.
5. Open the app again from Applications. It starts.

### Route B — Terminal

Open **Terminal** (⌘-Space, type "Terminal", press Return), then **type this
by hand**:

```
xattr -dr com.apple.quarantine
```

Type one space after `quarantine`, then **drag the app from the Applications
folder onto the Terminal window** — that inserts its path correctly — and press
Return. Open the app normally afterwards.

> **Do not copy this command out of a chat message, email, or web page.**
> Formatted text carries invisible characters, and pasting them produces:
>
> ```
> zsh: command not found:  xattr
> ```
>
> That is not a missing tool — `xattr` is part of macOS and always present at
> `/usr/bin/xattr`. The shell is reading an invisible character glued to the
> word. Typing the command by hand avoids it entirely. If you already pasted
> it and got that error, just retype it.

This removes the "downloaded from the internet" flag that macOS attaches to
files arriving by browser, email, AirDrop, or Slack. It is the flag — not the
app — that triggers the block.

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
~/Library/Application Support/orinth.ai.studio/
```

Logs, if you ever need to send them:

```
~/Library/Application Support/orinth.ai.studio/logs/
```

The setup window also has an **Open logs** button.

## Uninstall

```bash
rm -rf "/Applications/Orinth.app"
rm -rf ~/Library/Application\ Support/orinth.ai.studio
```

The second command deletes your projects and trained models too — skip it if
you want to keep them for a reinstall.

## Intel Macs

Supported, on older versions of the ML libraries. TensorFlow and PyTorch both
stopped publishing Intel-Mac builds in 2024, so on an Intel Mac the app installs
the last releases that have them — TensorFlow 2.16.2 and PyTorch 2.2.2 — while
Apple silicon gets current versions. This is handled by platform markers in the
dependency manifest; nothing to configure.

One consequence worth knowing: `transformers` declares its own Torch code paths
against torch 2.4 or newer, so NLP and LLM features run on an older Torch than
upstream targets on Intel. Image workflows are unaffected.

## Troubleshooting

**"The application ... can't be opened."**
The app's architecture does not match the Mac. Make sure the file is the
**`_universal.dmg`**, not `_aarch64.dmg` (Apple silicon only).

**"...is damaged and can't be opened."**
The download was truncated or the quarantine flag is set. Re-download, then
use Route A above.

**Setup fails at "Installing ML dependencies"**
Almost always the network. Click **Retry**. On a metered or filtered
connection, note that it downloads from `pypi.org` and `nodejs.org`.

**The app opens but pages do not load**
Send `backend.log` and `frontend.log` from the logs folder above.
