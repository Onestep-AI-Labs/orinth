"""Serving endpoints and the SSE chat proxy (phase 15).

The chat proxy exists so the browser only ever talks to the API origin — the
llama.cpp port stays an internal implementation detail. SSE-over-fetch (not
websockets) follows the platform's no-websockets convention: chat needs only
one-directional streaming.
"""

import json
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.container import serving_service
from app.schemas import (
    ChatRequest,
    HubDownloadRequest,
    HubDownloadResult,
    HubFilesResponse,
    HubModelSearchResponse,
    HubRecommendationsResponse,
    ServingBrowseResult,
    ServingConfig,
    ServingConfigUpdate,
    ServingPickRequest,
    ServingPickResult,
    ServingScanResult,
    ServingStartRequest,
    ServingStatus,
    WebSearchRequest,
    WebSearchResponse,
)
from app.services.serving import ServingError, ServingService

router = APIRouter()


@router.post("/serving/start", response_model=ServingStatus)
def serving_start(payload: ServingStartRequest) -> ServingStatus:
    try:
        return serving_service.start(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Model not found") from exc
    except ServingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/serving/stop", response_model=ServingStatus)
def serving_stop() -> ServingStatus:
    return serving_service.stop()


@router.get("/serving/status", response_model=ServingStatus)
def serving_status() -> ServingStatus:
    return serving_service.status()


@router.get("/serving/scan", response_model=ServingScanResult)
def serving_scan(path: str = Query(..., min_length=1)) -> ServingScanResult:
    try:
        return serving_service.scan_models(path)
    except ServingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/serving/browse", response_model=ServingBrowseResult)
def serving_browse(path: str | None = Query(default=None)) -> ServingBrowseResult:
    try:
        return serving_service.browse(path)
    except ServingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/serving/config", response_model=ServingConfig)
def serving_config() -> ServingConfig:
    return serving_service.load_config()


@router.put("/serving/config", response_model=ServingConfig)
def serving_config_update(payload: ServingConfigUpdate) -> ServingConfig:
    try:
        return serving_service.save_config(payload)
    except ServingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/serving/pick", response_model=ServingPickResult)
def serving_pick(payload: ServingPickRequest) -> ServingPickResult:
    return serving_service.pick_path(payload.kind)


@router.get("/serving/hf/search", response_model=HubModelSearchResponse)
def serving_hf_search(
    query: str = Query(default=""),
    format: str = Query(default="gguf"),
) -> HubModelSearchResponse:
    return serving_service.search_hub(query, format)


@router.get("/serving/hf/recommended", response_model=HubRecommendationsResponse)
def serving_hf_recommended() -> HubRecommendationsResponse:
    return serving_service.recommended_hub_models()


@router.get("/serving/hf/files", response_model=HubFilesResponse)
def serving_hf_files(
    repo_id: str = Query(..., min_length=1),
    format: str = Query(default="gguf"),
) -> HubFilesResponse:
    return serving_service.hub_files(repo_id, format)


@router.post("/serving/hf/download", response_model=HubDownloadResult)
def serving_hf_download(payload: HubDownloadRequest) -> HubDownloadResult:
    try:
        return serving_service.download_hub(payload.repo_id, payload.filename)
    except ServingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/serving/web-search", response_model=WebSearchResponse)
def serving_web_search(payload: WebSearchRequest) -> WebSearchResponse:
    return serving_service.web_search(payload.query, payload.max_results)


@router.post("/serving/chat")
async def serving_chat(payload: ChatRequest) -> StreamingResponse:
    try:
        port = serving_service.running_session_port()
    except ServingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    upstream = _upstream_lines(port, upstream_chat_body(payload))
    return StreamingResponse(
        chat_event_stream(upstream, serving_service),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class UpstreamError(RuntimeError):
    """Raised by the upstream line source; surfaces as an SSE error event."""


def upstream_chat_body(payload: ChatRequest) -> dict[str, Any]:
    """OpenAI-compatible request body; the GGUF's own chat template applies upstream."""
    messages = [{"role": message.role, "content": message.content} for message in payload.messages]
    if payload.system:
        messages = [{"role": "system", "content": payload.system}, *messages]
    body: dict[str, Any] = {"messages": messages, "stream": True}
    if payload.temperature is not None:
        body["temperature"] = payload.temperature
    if payload.top_p is not None:
        body["top_p"] = payload.top_p
    if payload.max_tokens is not None:
        body["max_tokens"] = payload.max_tokens
    return body


async def _upstream_lines(port: int, body: dict[str, Any]) -> AsyncIterator[str]:
    """Stream raw SSE lines from the llama.cpp server.

    Closing this generator (client abort) exits the httpx stream context, which
    cancels the upstream request so generation stops server-side too.
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=body) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode("utf-8", errors="replace")[:400]
                raise UpstreamError(
                    f"llama.cpp server rejected the request ({response.status_code}): {detail}"
                )
            async for line in response.aiter_lines():
                yield line


async def chat_event_stream(
    upstream: AsyncIterator[str], service: ServingService
) -> AsyncIterator[str]:
    """Re-emit upstream OpenAI-style SSE as this API's chat event frames.

    Frames: ``{"type": "delta", "content": ...}`` per token, one final
    ``{"type": "usage", ...}`` with tokens/s and time-to-first-token, an
    ``{"type": "error", ...}`` frame when the upstream dies mid-stream, and a
    closing ``data: [DONE]``.
    """
    started = perf_counter()
    first_token_at: float | None = None
    completion_tokens = 0
    upstream_usage: dict[str, Any] | None = None
    try:
        async for line in upstream:
            data = _sse_data(line)
            if data is None or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk.get("usage"), dict):
                upstream_usage = chunk["usage"]
            choices = chunk.get("choices") or []
            delta = ""
            if choices and isinstance(choices[0], dict):
                delta = str((choices[0].get("delta") or {}).get("content") or "")
            if delta:
                completion_tokens += 1
                if first_token_at is None:
                    first_token_at = perf_counter()
                # Per-token activity: a long generation never counts as idle.
                service.touch_activity()
                yield _sse_frame({"type": "delta", "content": delta})
    except (UpstreamError, httpx.HTTPError, OSError) as exc:
        detail = f"Generation stream failed: {exc}"
        service.record_crash(detail)
        yield _sse_frame({"type": "error", "message": detail})
        yield "data: [DONE]\n\n"
        return

    now = perf_counter()
    generation_seconds = max(1e-6, now - (first_token_at or started))
    usage: dict[str, Any] = {
        "type": "usage",
        "completion_tokens": (upstream_usage or {}).get("completion_tokens", completion_tokens),
        "prompt_tokens": (upstream_usage or {}).get("prompt_tokens"),
        "tokens_per_second": round(completion_tokens / generation_seconds, 2),
        "time_to_first_token_seconds": (
            round(first_token_at - started, 3) if first_token_at is not None else None
        ),
    }
    yield _sse_frame(usage)
    yield "data: [DONE]\n\n"


def _sse_data(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


def _sse_frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
