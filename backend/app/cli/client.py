"""The HTTP client, and the one place a server error becomes a `CliError`.

Thin on purpose. It carries the base URL, turns transport failures into the two
exit codes that distinguish "the server is down" from "the server said no", and
does nothing else — no response models, no caching, no retries. Retrying is
specifically avoided: every write this CLI makes creates a job or a dataset, and
a retried `POST /prep` on a slow response is a second run over the same files.
"""

import json
from pathlib import Path
from typing import Any

import httpx

from app.cli.errors import BackendUnreachable, CliError

#: Job following does its own polling with its own budget, so this bounds one
#: request rather than one operation.
DEFAULT_TIMEOUT = 60.0
#: An upload is bounded by the size of what is being uploaded, not by how quickly
#: the server thinks. A 5 GB ingest at 60s would fail on principle.
UPLOAD_TIMEOUT = 3600.0


class Client:
    def __init__(self, base_url: str, source: str = "default", timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.source = source
        self.timeout = timeout

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}{path if path.startswith('/') else '/' + path}"

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        timeout = kwargs.pop("timeout", self.timeout)
        try:
            response = httpx.request(method, self._url(path), timeout=timeout, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            raise BackendUnreachable(self.base_url, self.source, type(error).__name__) from error
        except httpx.HTTPError as error:
            raise CliError(f"Request to {self._url(path)} failed: {error}") from error
        return self._decode(response)

    @staticmethod
    def _decode(response: httpx.Response) -> Any:
        if response.status_code >= 400:
            raise CliError(_server_message(response))
        if not response.content:
            return None
        try:
            return response.json()
        except json.JSONDecodeError as error:
            raise CliError(f"Server returned non-JSON ({response.status_code})") from error

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self.request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self.request("DELETE", path, **kwargs)

    def download(self, path: str, destination: Path, **kwargs: Any) -> Path:
        """Stream a response to a file rather than buffering it in memory.

        A dataset export is the whole dataset; reading it into a bytes object to
        write it back out doubles the peak memory for no benefit.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.stream(
                "GET", self._url(path), timeout=UPLOAD_TIMEOUT, **kwargs
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise CliError(_server_message(response))
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        handle.write(chunk)
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            raise BackendUnreachable(self.base_url, self.source, type(error).__name__) from error
        except httpx.HTTPError as error:
            raise CliError(f"Download failed: {error}") from error
        return destination

    def reachable(self) -> bool:
        try:
            httpx.get(self._url("/health"), timeout=2.0)
        except httpx.HTTPError:
            return False
        return True


def _server_message(response: httpx.Response) -> str:
    """Prefer FastAPI's `detail` over a bare status line.

    The server already writes an actionable sentence for the cases the user can
    fix — a gated Hub dataset, a prep run already in flight, a validation
    failure. Replacing that with "HTTP 409" throws away the only useful part.
    """
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        if isinstance(detail, list) and detail:
            # Pydantic validation errors: `[{"loc": [...], "msg": "..."}]`.
            parts = []
            for entry in detail:
                if not isinstance(entry, dict):
                    continue
                where = ".".join(str(item) for item in entry.get("loc", [])[1:])
                parts.append(f"{where}: {entry.get('msg', '')}" if where else str(entry.get("msg")))
            if parts:
                return "; ".join(parts)
    return f"Server returned {response.status_code} for {response.request.url.path}"
