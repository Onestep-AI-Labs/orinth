"""Notebook CRUD, the runtime supervisor, and the `jupyter-server` proxy.

Literal paths (`/templates`, `/runtime/*`, `/proxy/*`) are declared **ahead of**
`/{notebook_id}` for the reason already documented at the top of
`routers/datasets.py`: FastAPI matches in declaration order, and a path
parameter declared first swallows every literal after it.

No kernel restart or interrupt endpoint is added. The client reaches
`jupyter-server`'s own `/api/kernels/{id}/restart` and `/interrupt` through the
proxy; duplicating them here would create a second definition of the same
operation that can disagree with the first.
"""

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from starlette.websockets import WebSocket

from app.container import notebook_runtime, notebook_service
from app.core.defaults import DEFAULT_PROJECT_ID
from app.schemas import (
    DeleteResponse,
    NotebookComputeTarget,
    NotebookCreate,
    NotebookRun,
    NotebookRunSeries,
    NotebookRuntimeStart,
    NotebookRuntimeStatus,
    NotebookSession,
    NotebookSummary,
    NotebookTemplate,
    NotebookUpdate,
)
from app.services.notebooks import proxy as proxy_module
from app.services.notebooks import runs as runs_module
from app.services.notebooks import runtime as runtime_module
from app.services.notebooks.runtime import KERNEL_NAME, NotebookRuntimeError

router = APIRouter(prefix="/notebooks", tags=["notebooks"])


# ---- literals first --------------------------------------------------------


@router.get("/templates", response_model=list[NotebookTemplate])
def list_notebook_templates() -> list[NotebookTemplate]:
    return notebook_service.templates()


@router.get("/runtime", response_model=NotebookRuntimeStatus)
def get_notebook_runtime() -> NotebookRuntimeStatus:
    return notebook_runtime.status()


@router.get("/runtime/targets", response_model=list[NotebookComputeTarget])
def list_notebook_compute_targets() -> list[NotebookComputeTarget]:
    """What the runtime's machine dropdown offers.

    Declared **above** `/runtime/start` for the reason at the top of this file,
    and built in the service so the roadmap (`available: false` providers) has
    one owner rather than one per client.
    """
    return runtime_module.compute_targets()


@router.post("/runtime/start", response_model=NotebookRuntimeStatus)
def start_notebook_runtime(payload: NotebookRuntimeStart | None = None) -> NotebookRuntimeStatus:
    try:
        return notebook_runtime.start((payload or NotebookRuntimeStart()).device)
    except NotebookRuntimeError as error:
        # 503, not 500: a missing optional extra is an expected state with an
        # obvious remedy, and the UI turns this into an install hint.
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/runtime/restart", response_model=NotebookRuntimeStatus)
def restart_notebook_runtime(payload: NotebookRuntimeStart | None = None) -> NotebookRuntimeStatus:
    """Stop and start in one call — the only way to change the device.

    A kernel inherits its environment at spawn, so moving from CPU to GPU means
    a new server. Doing it here rather than as two client calls closes the gap
    another tab could start the old one in.
    """
    try:
        return notebook_runtime.restart((payload or NotebookRuntimeStart()).device)
    except NotebookRuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/runtime/stop", response_model=NotebookRuntimeStatus)
def stop_notebook_runtime() -> NotebookRuntimeStatus:
    return notebook_runtime.stop()


@router.websocket("/proxy/{path:path}")
async def proxy_notebook_websocket(websocket: WebSocket, path: str) -> None:
    await proxy_module.forward_websocket(
        notebook_runtime, websocket, path, websocket.url.query
    )


@router.api_route(
    "/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    # Kept out of the schema: it is a verbatim pass-through to another server's
    # API, not part of this one's contract, and a `{path}` catch-all across seven
    # methods generates duplicate operation ids and junk in the generated client.
    include_in_schema=False,
)
async def proxy_notebook_http(path: str, request: Request) -> Response:
    return await proxy_module.forward_http(
        notebook_runtime,
        method=request.method,
        path=path,
        query=str(request.url.query),
        headers=dict(request.headers),
        body=await request.body(),
    )


# ---- collection ------------------------------------------------------------


@router.get("", response_model=list[NotebookSummary])
def list_notebooks(project_id: str = Query(default=DEFAULT_PROJECT_ID)) -> list[NotebookSummary]:
    return notebook_service.list_notebooks(project_id)


@router.post("", response_model=NotebookSummary)
def create_notebook(payload: NotebookCreate) -> NotebookSummary:
    return notebook_service.create(payload)


# ---- one notebook ----------------------------------------------------------


@router.get("/{notebook_id}", response_model=NotebookSummary)
def get_notebook(notebook_id: str) -> NotebookSummary:
    return notebook_service.summary(notebook_id)


@router.patch("/{notebook_id}", response_model=NotebookSummary)
def update_notebook(notebook_id: str, payload: NotebookUpdate) -> NotebookSummary:
    return notebook_service.update(notebook_id, payload)


@router.post("/{notebook_id}/duplicate", response_model=NotebookSummary)
def duplicate_notebook(notebook_id: str) -> NotebookSummary:
    return notebook_service.duplicate(notebook_id)


@router.delete("/{notebook_id}", response_model=DeleteResponse)
def delete_notebook(notebook_id: str) -> DeleteResponse:
    notebook_service.delete(notebook_id)
    return DeleteResponse(deleted=1)


@router.get("/{notebook_id}/session", response_model=NotebookSession)
def get_notebook_session(notebook_id: str) -> NotebookSession:
    """What the browser needs to open a kernel channel.

    Same-origin paths, not the loopback address: the client talks to this
    process, which forwards. Handing out `127.0.0.1:<port>` would leak the
    runtime's location and would not work through the dev proxy.
    """
    notebook = notebook_service.summary(notebook_id)
    return NotebookSession(
        base_url="/api/notebooks/proxy",
        ws_url="/api/notebooks/proxy",
        kernel_name=KERNEL_NAME,
        notebook_path=notebook.path,
    )


@router.get("/{notebook_id}/runs", response_model=list[NotebookRun])
def list_notebook_runs(notebook_id: str) -> list[NotebookRun]:
    return runs_module.list_runs(notebook_service.runs_dir(notebook_id), notebook_id)


@router.get("/{notebook_id}/runs/{run_id}", response_model=NotebookRunSeries)
def get_notebook_run(notebook_id: str, run_id: str) -> NotebookRunSeries:
    series = runs_module.get_run(notebook_service.runs_dir(notebook_id), notebook_id, run_id)
    if series is None:
        raise HTTPException(status_code=404, detail=f"No run '{run_id}'")
    return series


@router.get("/{notebook_id}/runs/{run_id}/artifacts/{name}")
def download_notebook_artifact(notebook_id: str, run_id: str, name: str) -> FileResponse:
    path = runs_module.artifact_path(notebook_service.runs_dir(notebook_id), run_id, name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No artifact '{name}'")
    return FileResponse(path, filename=path.name)
