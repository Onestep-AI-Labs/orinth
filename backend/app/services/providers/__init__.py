"""LLM provider clients.

Deliberately imports nothing from `app.services`: `services/recipes/__init__`
imports `DatasetService`, so a provider that reached back into the dataset layer
would close an import cycle and force function-local imports at every call site.
"""

from app.services.providers.errors import (
    LlmAuthError,
    LlmError,
    LlmRateLimitError,
    LlmRequestError,
    LlmResponseError,
)
from app.services.providers.openrouter import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterClient,
)
from app.services.providers.types import ChatMessage, ChatResult, LlmUsage, ModelInfo

__all__ = [
    "ChatMessage",
    "ChatResult",
    "DEFAULT_OPENROUTER_MODEL",
    "LlmAuthError",
    "LlmError",
    "LlmRateLimitError",
    "LlmRequestError",
    "LlmResponseError",
    "LlmUsage",
    "ModelInfo",
    "OpenRouterClient",
]
