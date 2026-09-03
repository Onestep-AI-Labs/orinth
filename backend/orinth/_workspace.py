"""Where the workspace is, and who this notebook belongs to.

Resolved once per kernel. The kernel is spawned by the notebook runtime with
`STORAGE_DIR`, `MODELS_DIR`, `DATASETS_DIR`, `DATABASE_URL`, and
`ORINTH_API_BASE` already in its environment, so `Settings()` here produces
exactly the paths the server is using — the SDK never guesses a location and
never needs a config file of its own.

Which notebook this is, though, is **not** in the environment. The runtime spawns
one `jupyter-server` for the whole workspace and it spawns every kernel, so a
per-notebook variable has nowhere to be set. It is read from the working
directory instead: `jupyter-server` starts a kernel in the directory of the
session path, which for `<notebook_id>/notebook.ipynb` is the notebook's own
directory. `ORINTH_NOTEBOOK_ID` still wins when something sets it, which is what
tests and a headless runner use.
"""

import functools
import json
import os
from pathlib import Path


@functools.lru_cache(maxsize=1)
def settings():
    """The server's own `Settings`, built from the kernel's environment."""
    from app.core.config import Settings

    return Settings()


@functools.lru_cache(maxsize=1)
def storage():
    from app.core.storage import Storage

    return Storage(settings())


def api_base() -> str:
    return (os.environ.get("ORINTH_API_BASE") or settings().api_base_url).rstrip("/")


def notebook_id() -> str | None:
    """This kernel's notebook, or None when running outside one.

    None is a normal state — the SDK is importable from a plain `python -c` for
    testing — and every caller that needs an id says so with its own error
    rather than assuming.

    The working-directory branch is what makes `orinth.runs` work at all. The
    first cut read only `ORINTH_NOTEBOOK_ID`, and nothing anywhere set it: one
    `jupyter-server` serves every notebook in the workspace, so there is no
    point in the lifecycle where a per-notebook variable could be written. Every
    `orinth.runs.log(...)` in a real kernel therefore raised, and the Runs rail
    could never fill. A kernel's cwd, by contrast, *is* per notebook —
    `jupyter-server` starts it in the session path's directory.

    The identity is confirmed against the workspace rather than trusted: the
    directory has to sit directly under `storage/notebooks/` and hold a
    `manifest.json`, so a kernel started somewhere else reports None instead of
    inventing a notebook out of whatever `cwd` happened to be.
    """
    value = (os.environ.get("ORINTH_NOTEBOOK_ID") or "").strip()
    if value:
        return value
    try:
        directory = Path.cwd().resolve()
    except OSError:
        # A deleted cwd is possible and is not this function's problem.
        return None
    root = storage().notebooks.resolve()
    if directory.parent == root and (directory / "manifest.json").is_file():
        return directory.name
    return None


def notebook_dir() -> Path | None:
    identifier = notebook_id()
    if not identifier:
        return None
    return storage().notebooks / identifier


def notebook_manifest() -> dict:
    directory = notebook_dir()
    if directory is None:
        return {}
    path = directory / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def default_project_id() -> str:
    """The notebook's project, falling back to the platform default."""
    from app.core.defaults import DEFAULT_PROJECT_ID

    return notebook_manifest().get("project_id") or DEFAULT_PROJECT_ID
