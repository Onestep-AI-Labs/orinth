"""OpenRouter chat client.

`DEFAULT_OPENROUTER_MODEL` was verified against the live catalog at
`GET https://openrouter.ai/api/v1/models` on **2026-07-19**: `openai/gpt-5.4-mini`
is present, advertises both `response_format` and `structured_outputs`, and
carries a 400k context at $0.75/$4.50 per million prompt/completion tokens.
`openai/gpt-5.4-nano` is the same family roughly four times cheaper if generation
volume matters more than quality. Re-verify this slug before changing it —
model ids are retired without notice, and a default that 404s makes every job
fail at the first chunk.
"""

import json
import random
import re
import time
from typing import Any

from app.services.providers.errors import (
    LlmAuthError,
    LlmError,
    LlmRateLimitError,
    LlmRequestError,
    LlmResponseError,
)
from app.services.providers.types import ChatMessage, ChatResult, LlmUsage, ModelInfo

#: Fallback only. The model actually used comes from `SettingsService.openrouter_model()`,
#: which is the field the user already configures in Settings for phase 11 recipes —
#: so there is one place a model is chosen, not two that can disagree.
DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.4-mini"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_MAX_ATTEMPTS = 4
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 30.0
_CATALOG_TTL_SECONDS = 600.0
# A hung socket with a 2-worker pool permanently consumes half the pool, so both
# halves of the timeout are mandatory rather than left to httpx's defaults.
_CONNECT_TIMEOUT_SECONDS = 10.0

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_json_fence(text: str) -> str:
    """Unwrap ```json fences. Models emit them even when asked not to."""
    match = _FENCE_RE.match(text or "")
    return match.group(1) if match else (text or "").strip()


class OpenRouterClient:
    """Thin OpenRouter wrapper.

    Constructor-injected into `DataPrepService` specifically so tests can pass a
    fake with a scripted `chat_json` — no HTTP, no monkeypatching.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        timeout_seconds: float = 120.0,
        transport: Any = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        # Only `test_openrouter_client.py` passes one (an `httpx.MockTransport`).
        self._transport = transport
        self._catalog: list[ModelInfo] | None = None
        self._catalog_fetched_at = 0.0

    @property
    def configured(self) -> bool:
        return bool((self.api_key or "").strip())

    def _client(self):
        import httpx  # lazy: keeps httpx off the import path of every request

        return httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS),
            transport=self._transport,
            headers={
                "Authorization": f"Bearer {(self.api_key or '').strip()}",
                "Content-Type": "application/json",
                # OpenRouter attributes traffic with these; harmless if absent.
                "HTTP-Referer": "https://github.com/Onestep-AI-Labs/orinth",
                "X-Title": "Orinth",
            },
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        if not self.configured:
            raise LlmAuthError("OpenRouter API key is not configured")

        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_schema is not None:
            # Requested, but never trusted — support is uneven across models, so
            # every response still goes through the parse/validate path below.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": json_schema},
            }

        body = self._post_with_retry("/chat/completions", payload)
        choices = body.get("choices") or []
        if not choices:
            raise LlmResponseError("OpenRouter returned no choices")
        content = (choices[0].get("message") or {}).get("content") or ""
        if not content.strip():
            raise LlmResponseError("OpenRouter returned an empty message")

        usage_body = body.get("usage") or {}
        return ChatResult(
            content=content,
            usage=LlmUsage(
                prompt_tokens=int(usage_body.get("prompt_tokens") or 0),
                completion_tokens=int(usage_body.get("completion_tokens") or 0),
                cost_usd=_optional_float(usage_body.get("cost")),
            ),
            model=str(body.get("model") or model),
        )

    def chat_json(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], LlmUsage]:
        """Call the model and parse a JSON object, with exactly one repair turn.

        One repair, not more: a model that cannot emit valid JSON twice will not
        on the fifth attempt, and every attempt costs money.
        """
        result = self.chat(
            messages, model=model, temperature=temperature, json_schema=json_schema, max_tokens=max_tokens
        )
        usage = result.usage
        try:
            return _loads_object(result.content), usage
        except LlmResponseError as first_error:
            repair = [
                *messages,
                ChatMessage(role="assistant", content=result.content),
                ChatMessage(
                    role="user",
                    content=(
                        "That was not valid JSON. Reply with the JSON object only — "
                        "no prose, no markdown fences, no trailing commas."
                    ),
                ),
            ]
            retry = self.chat(
                repair, model=model, temperature=0.0, json_schema=json_schema, max_tokens=max_tokens
            )
            usage = usage + retry.usage
            try:
                return _loads_object(retry.content), usage
            except LlmResponseError as second_error:
                raise LlmResponseError(
                    f"model did not return valid JSON after one repair turn: {second_error}"
                ) from first_error

    def models(self, force: bool = False) -> list[ModelInfo]:
        """The provider catalog, cached ~10 minutes. Returns `[]` when unconfigured."""
        if not self.configured:
            return []
        fresh = time.monotonic() - self._catalog_fetched_at < _CATALOG_TTL_SECONDS
        if self._catalog is not None and fresh and not force:
            return self._catalog

        try:
            with self._client() as client:
                response = client.get("/models")
                response.raise_for_status()
                rows = response.json().get("data") or []
        except LlmError:
            raise
        except Exception as error:  # network/parse failures must not break Settings
            if self._catalog is not None:
                return self._catalog
            raise LlmError(f"could not load the OpenRouter model catalog: {error}") from error

        self._catalog = [_model_info(row) for row in rows if row.get("id")]
        self._catalog_fetched_at = time.monotonic()
        return self._catalog

    def _post_with_retry(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                with self._client() as client:
                    response = client.post(path, json=payload)
            except httpx.TimeoutException as error:
                last_error = LlmError(f"OpenRouter request timed out: {error}")
            except httpx.HTTPError as error:
                last_error = LlmError(f"OpenRouter request failed: {error}")
            else:
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as error:
                        raise LlmResponseError(f"OpenRouter returned non-JSON body: {error}") from error

                detail = _error_detail(response)
                if response.status_code in (401, 403):
                    raise LlmAuthError(f"OpenRouter rejected the API key: {detail}")
                if response.status_code == 429:
                    last_error = LlmRateLimitError(detail, _retry_after(response))
                elif response.status_code < 500:
                    raise LlmRequestError(f"OpenRouter rejected the request: {detail}")
                else:
                    last_error = LlmError(f"OpenRouter server error {response.status_code}: {detail}")

            if attempt == _MAX_ATTEMPTS - 1:
                break
            time.sleep(_backoff_seconds(attempt, getattr(last_error, "retry_after", None)))

        raise last_error if last_error else LlmError("OpenRouter request failed")


def _backoff_seconds(attempt: int, retry_after: float | None) -> float:
    """Full-jitter exponential backoff, honoring `Retry-After` when present."""
    if retry_after is not None:
        return min(max(retry_after, 0.0), _BACKOFF_CAP_SECONDS)
    ceiling = min(_BACKOFF_BASE_SECONDS * (2**attempt), _BACKOFF_CAP_SECONDS)
    return random.uniform(0.0, ceiling)


def _retry_after(response: Any) -> float | None:
    raw = response.headers.get("Retry-After") if getattr(response, "headers", None) else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _error_detail(response: Any) -> str:
    try:
        body = response.json()
    except Exception:
        return (getattr(response, "text", "") or "").strip()[:400] or f"HTTP {response.status_code}"
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error)[:400]
    return str(error or body)[:400]


def _loads_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(strip_json_fence(content))
    except json.JSONDecodeError as error:
        raise LlmResponseError(f"response was not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise LlmResponseError("response JSON was not an object")
    return parsed


def _model_info(row: dict[str, Any]) -> ModelInfo:
    pricing = row.get("pricing") or {}
    return ModelInfo(
        id=str(row["id"]),
        name=str(row.get("name") or row["id"]),
        context_length=_optional_int(row.get("context_length")),
        prompt_price_usd_per_token=_optional_float(pricing.get("prompt")),
        completion_price_usd_per_token=_optional_float(pricing.get("completion")),
    )


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
