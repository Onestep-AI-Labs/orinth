"""Where the workspace is, and who this notebook belongs to.

Resolved once per kernel. The kernel is spawned by the notebook runtime with
`STORAGE_DIR`, `MODELS_DIR`, `DATASETS_DIR`, `DATABASE_URL`, and
`ORINTH_API_BASE` already in its environment, so `Settings()` here produces
exactly the paths the server is using — the SDK never guesses a location and
never needs a config file of its own.

`ORINTH_NOTEBOOK_ID` is set the same way, which is how `orinth.runs` knows where
to write and `orinth.project()` knows which project it is in without being told.
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
    """
    value = (os.environ.get("ORINTH_NOTEBOOK_ID") or "").strip()
    return value or None


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
