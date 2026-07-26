"""Thin OpenRouter client over ``httpx``: model listing and chat completion.

The API key is passed in per call (injected server-side from settings) and is
never logged or stored. Failures are surfaced as typed exceptions so the
generation loop can decide between a per-chunk retry and flipping the run to
rule-based.
"""

import httpx

from app.services.recipes.constants import FALLBACK_OPENROUTER_MODELS

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_TIMEOUT = 60.0


class OpenRouterError(Exception):
    """Generic, non-retryable OpenRouter failure."""


class OpenRouterAuthError(OpenRouterError):
    """401/403 — the key was rejected."""


class OpenRouterRateLimitError(OpenRouterError):
    """429 — rate limited."""


def curated_models() -> list[tuple[str, str]]:
    return list(FALLBACK_OPENROUTER_MODELS)


def list_models(api_key: str) -> list[tuple[str, str]]:
    """Live model list from OpenRouter. Raises on failure; caller falls back."""
    response = httpx.get(
        f"{OPENROUTER_BASE}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    if response.status_code in (401, 403):
        raise OpenRouterAuthError("OpenRouter key was rejected")
    if response.status_code >= 400:
        raise OpenRouterError(f"OpenRouter models error ({response.status_code})")
    payload = response.json()
    models: list[tuple[str, str]] = []
    for entry in payload.get("data", []) if isinstance(payload, dict) else []:
        model_id = entry.get("id")
        if not model_id:
            continue
        models.append((model_id, entry.get("name") or model_id))
    return models or curated_models()


def chat_completion(api_key: str, model: str, system: str, user: str) -> str:
    """One chat completion. Returns the assistant message content string.

    Raises :class:`OpenRouterAuthError` on 401/403, :class:`OpenRouterRateLimitError`
    on 429, and :class:`OpenRouterError` on timeout / transport / other errors.
    """
    try:
        response = httpx.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.4,
            },
            timeout=_TIMEOUT,
        )
    except httpx.TimeoutException as exc:
        raise OpenRouterError(f"OpenRouter request timed out: {exc}") from exc
    except httpx.HTTPError as exc:
        raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
    if response.status_code in (401, 403):
        raise OpenRouterAuthError("OpenRouter key was rejected")
    if response.status_code == 429:
        raise OpenRouterRateLimitError("OpenRouter rate limit reached")
    if response.status_code >= 400:
        raise OpenRouterError(f"OpenRouter error ({response.status_code})")
    payload = response.json()
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError("OpenRouter returned no completion") from exc
