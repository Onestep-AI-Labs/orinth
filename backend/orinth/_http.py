"""The kernel's client for the write API.

Small on purpose. Reads go direct to the filesystem; the only things that come
through here are the operations that must run the project gate, write a manifest
atomically, and compute readiness exactly once — which is the API's job and not
something to reimplement in a notebook.

Failures name the base URL, because the most likely cause of one is that the
backend the kernel was told about is not the one running.
"""

import json
from typing import Any

from orinth import _workspace
from orinth.errors import OrinthError, ProjectTaskNotAllowed

TIMEOUT = 120.0
#: Ingest carries the whole dataset, so it is bounded by upload size rather than
#: by how quickly the server thinks.
UPLOAD_TIMEOUT = 3600.0


def _url(path: str) -> str:
    return f"{_workspace.api_base()}{path if path.startswith('/') else '/' + path}"


def request(method: str, path: str, **kwargs: Any) -> Any:
    import httpx

    url = _url(path)
    try:
        response = httpx.request(method, url, timeout=kwargs.pop("timeout", TIMEOUT), **kwargs)
    except httpx.HTTPError as error:
        raise OrinthError(
            f"Could not reach the Orinth API at {_workspace.api_base()}: {error}. "
            "Is the backend running? Set ORINTH_API_BASE if it is on another port."
        ) from error
    return _decode(response)


def _decode(response) -> Any:
    if response.status_code >= 400:
        raise OrinthError(_message(response))
    if not response.content:
        return None
    try:
        return response.json()
    except json.JSONDecodeError as error:
        raise OrinthError(f"API returned non-JSON ({response.status_code})") from error


def _message(response) -> str:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str) and detail:
        return detail
    return f"API returned {response.status_code} for {response.request.url.path}"


def get(path: str, **kwargs: Any) -> Any:
    return request("GET", path, **kwargs)


def post(path: str, **kwargs: Any) -> Any:
    return request("POST", path, **kwargs)


def raise_for_project(error: OrinthError, project_id: str, task_type: str) -> None:
    """Re-raise a project-gate 409 as the SDK's own error.

    The gate lives in the API (`_require_project_task`), so the SDK never
    pre-checks it — a second copy of that rule is a second thing to keep in sync.
    What it does do is recognise the answer and say what to do about it.
    """
    message = str(error)
    if "task" in message.lower() and "project" in message.lower():
        raise ProjectTaskNotAllowed(project_id, task_type) from error
    raise error
