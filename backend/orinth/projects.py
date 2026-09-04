"""Projects, and which one this notebook is in.

Read over HTTP rather than off the database: projects are DB rows, and
`app.core.database` builds an engine and mkdirs its directory as an import side
effect. A kernel opening a second SQLite connection to the file uvicorn is
writing is the "two owners of the same mutable state" problem the CLI's
transport decision spells out — and it buys nothing here, because a project list
is three fields and one HTTP call.
"""

import builtins

from orinth import _http, _workspace
from orinth.errors import OrinthError
from orinth.types import ProjectRef


def _ref(payload: dict) -> ProjectRef:
    return ProjectRef(
        id=payload.get("id", ""),
        name=payload.get("name", ""),
        task_types=builtins.list(payload.get("task_types") or []),
    )


def list() -> builtins.list[ProjectRef]:  # noqa: A001
    return [_ref(entry) for entry in _http.get("/api/projects") or []]


def get(project_id: str) -> ProjectRef:
    for ref in list():
        if ref.id == project_id:
            return ref
    raise OrinthError(f"No project '{project_id}'.")


def current() -> ProjectRef:
    """The project this notebook belongs to, from its manifest."""
    return get(_workspace.default_project_id())
