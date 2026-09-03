"""Pass-through to the managed `jupyter-server`, and nothing more.

**No URL rewriting.** `jupyter-server` runs with
`ServerApp.base_url = "/api/notebooks/proxy/"`, so every URL it generates for
itself is already correct on the outside. A proxy that rewrote bodies would have
to understand every response shape the Jupyter API has, and would break the
first time one changed.

Its only mutation is injecting the token outbound, so the credential stays
server-side and never reaches the browser. The browser therefore holds nothing
worth stealing from it, which is also why `disable_check_xsrf` is safe here.

The one thing this refuses is a destination: requests go to the loopback port
*this process allocated* and nowhere else. Without that check a crafted path
would turn the backend into an open relay.
"""

from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import Response
from starlette.websockets import WebSocket, WebSocketDisconnect

from app.services.notebooks.runtime import NotebookRuntime, NotebookRuntimeError

#: Hop-by-hop headers a proxy must not forward. `content-length` is dropped
#: because the body is re-framed; `connection` and friends belong to the hop.
_STRIP_REQUEST = {"host", "connection", "content-length", "authorization", "cookie"}
_STRIP_RESPONSE = {
    "content-encoding",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
}

#: Long enough for a slow kernel start; short enough that a wedged request does
#: not hold a worker forever. Kernel *execution* rides the WebSocket, not this.
HTTP_TIMEOUT = 120.0


def _require_running(runtime: NotebookRuntime) -> str:
    try:
        return runtime.base_url()
    except NotebookRuntimeError as error:
        # 503 rather than 500: the runtime being stopped is an expected state
        # with an obvious remedy, and the UI turns this into a Start button.
        raise HTTPException(status_code=503, detail=str(error)) from error


async def forward_http(
    runtime: NotebookRuntime,
    *,
    method: str,
    path: str,
    query: str,
    headers: dict[str, str],
    body: bytes,
) -> Response:
    base = _require_running(runtime)
    url = f"{base}/api/notebooks/proxy/{path.lstrip('/')}"
    outbound = {
        key: value for key, value in headers.items() if key.lower() not in _STRIP_REQUEST
    }
    outbound.update(runtime.headers())

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            upstream = await client.request(
                method,
                url,
                params=query or None,
                headers=outbound,
                content=body or None,
            )
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502, detail=f"The notebook server did not answer: {error}"
        ) from error

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers={
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in _STRIP_RESPONSE
        },
        media_type=upstream.headers.get("content-type"),
    )


async def forward_websocket(
    runtime: NotebookRuntime, websocket: WebSocket, path: str, query: str
) -> None:
    """Bridge the browser's kernel channel to `jupyter-server`'s.

    Two independent pumps rather than a request/response loop: kernel traffic is
    genuinely bidirectional and unsolicited in both directions — a cell's output
    arrives without the client having asked for it — so anything that alternates
    would deadlock on the first busy kernel.
    """
    import asyncio

    from websockets.exceptions import ConnectionClosed

    base = _require_running(runtime).replace("http://", "ws://")
    url = f"{base}/api/notebooks/proxy/{path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"

    await websocket.accept(subprotocol=_negotiated_subprotocol(websocket))

    try:
        import websockets
    except ImportError:
        await websocket.close(code=1011, reason="websockets is not installed")
        return

    try:
        async with websockets.connect(
            url,
            additional_headers=runtime.headers(),
            # Jupyter's kernel protocol negotiates this; declining it silently
            # makes message framing subtly wrong rather than failing loudly.
            subprotocols=["v1.kernel.websocket.jupyter.org"],
            max_size=None,
            open_timeout=30,
        ) as upstream:
            await asyncio.gather(
                _pump_to_upstream(websocket, upstream),
                _pump_to_client(websocket, upstream),
            )
    except (ConnectionClosed, WebSocketDisconnect):
        return
    except Exception as error:  # noqa: BLE001 - a proxy failure must close cleanly
        try:
            await websocket.close(code=1011, reason=str(error)[:120])
        except RuntimeError:
            return


def _negotiated_subprotocol(websocket: WebSocket) -> str | None:
    requested = websocket.headers.get("sec-websocket-protocol", "")
    offered = [entry.strip() for entry in requested.split(",") if entry.strip()]
    return offered[0] if offered else None


async def _pump_to_upstream(websocket: WebSocket, upstream: Any) -> None:
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            await upstream.close()
            return
        if (text := message.get("text")) is not None:
            await upstream.send(text)
        elif (data := message.get("bytes")) is not None:
            await upstream.send(data)


async def _pump_to_client(websocket: WebSocket, upstream: Any) -> None:
    async for message in upstream:
        if isinstance(message, bytes):
            await websocket.send_bytes(message)
        else:
            await websocket.send_text(message)
