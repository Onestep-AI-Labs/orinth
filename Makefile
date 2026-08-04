SHELL := /bin/bash

.PHONY: backend frontend dev test lint lint-backend typecheck build check doctor \
	desktop desktop-dev desktop-test

# --- Run -------------------------------------------------------------------

backend:
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && pnpm dev

# Fail fast if a dev port is already occupied. Uses bash's /dev/tcp so no
# extra tool (lsof/nc) is required.
define check_port
	@bash -c "exec 3<>/dev/tcp/127.0.0.1/$(1)" 2>/dev/null && { echo "Port $(1) is already in use. Stop whatever is using it, then re-run 'make dev'." >&2; exit 1; } || true
endef

# Run backend and frontend together in one terminal with interleaved logs.
# Ctrl-C (or any exit) tears down both process trees so nothing is left
# orphaned: `set -m` puts each job in its own process group, and the trap
# kills those groups (not just the top-level pid) on the way out.
#
# Waiting for "whichever server exits first" is done by polling both pids
# rather than `wait -n`: that builtin needs bash >=4.3, but macOS ships
# bash 3.2 as /bin/bash, where `wait -n` is a hard error. A 1s poll adds
# negligible latency to noticing a crashed server and is portable everywhere.
dev:
	$(call check_port,8000)
	$(call check_port,3000)
	@set -m; \
	trap 'echo "Stopping dev servers..."; kill -TERM -$$backend_pid -$$frontend_pid 2>/dev/null; wait 2>/dev/null' INT TERM EXIT; \
	(cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000) & backend_pid=$$!; \
	(cd frontend && pnpm dev) & frontend_pid=$$!; \
	while kill -0 $$backend_pid 2>/dev/null && kill -0 $$frontend_pid 2>/dev/null; do \
		sleep 1; \
	done

# --- Validation --------------------------------------------------------

test:
	cd backend && uv run pytest

# NOTE: once a `slow` pytest marker lands (see the CI quality gates spec),
# switch this (and the `check` target below) to `uv run pytest -m "not slow"`
# so `make check` stays fast. Plain `uv run pytest` works today with no
# flags since dev-only deps live in [dependency-groups].

lint-backend:
	cd backend && uv run ruff check .

lint:
	cd frontend && pnpm lint

typecheck:
	cd frontend && pnpm typecheck

build:
	cd frontend && pnpm build

# Aggregate gate mirroring docs/ai/workflow.md's validation commands:
# backend lint + fast backend tests, then frontend typecheck + lint + build.
# Stops at the first failing gate.
check: lint-backend test typecheck lint build
	@echo "All checks passed."

# --- macOS desktop app -------------------------------------------------

# See specs/phase-18-macos-desktop-app.md.
#
# The bundler's bundle_dmg.sh calls `head -1` internally, so a `head` earlier
# on PATH than /usr/bin's breaks DMG creation with a misleading "interstitial
# disk image was not found". libwww-perl ships such a `head`, and XAMPP bundles
# libwww-perl — putting the system paths first makes the build reproducible
# regardless of what else is installed.
DESKTOP_PATH := /usr/bin:/bin:/usr/sbin:/sbin:$(PATH)

# Build the installable .dmg. Output lands in
# desktop/src-tauri/target/release/bundle/dmg/.
desktop:
	cd desktop && PATH="$(DESKTOP_PATH)" pnpm build

# Run the desktop shell against a freshly staged bundle, with devtools.
desktop-dev:
	cd desktop && PATH="$(DESKTOP_PATH)" pnpm dev

desktop-test:
	cd desktop/src-tauri && cargo test

# --- Environment -------------------------------------------------------

# Verify the prerequisite tools/versions this project depends on are
# installed: uv (backend deps + venvs), pnpm (frontend deps), a Python 3.11
# interpreter (via uv, since that's what `uv venv --python 3.11` needs),
# and Node (for pnpm/next). Exits non-zero if anything required is missing.
doctor:
	@echo "Checking required tools..."; \
	ok=1; \
	if command -v uv >/dev/null 2>&1; then \
		echo "  [OK]      uv       $$(uv --version)"; \
	else \
		echo "  [MISSING] uv       install from https://docs.astral.sh/uv/getting-started/installation/"; \
		ok=0; \
	fi; \
	if command -v pnpm >/dev/null 2>&1; then \
		echo "  [OK]      pnpm     $$(pnpm --version)"; \
	else \
		echo "  [MISSING] pnpm     install from https://pnpm.io/installation"; \
		ok=0; \
	fi; \
	if command -v uv >/dev/null 2>&1 && py311=$$(uv python find 3.11 2>/dev/null); then \
		echo "  [OK]      python   3.11 -> $$py311"; \
	else \
		echo "  [MISSING] python   3.11 not found; run 'uv python install 3.11'"; \
		ok=0; \
	fi; \
	if command -v node >/dev/null 2>&1; then \
		echo "  [OK]      node     $$(node --version)"; \
	else \
		echo "  [MISSING] node     install from https://nodejs.org"; \
		ok=0; \
	fi; \
	if [ "$$ok" -eq 1 ]; then \
		echo "All required tools are present."; \
	else \
		echo "One or more required tools are missing. See above." >&2; \
		exit 1; \
	fi
