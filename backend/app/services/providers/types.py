"""Provider-neutral request/response shapes.

Only OpenRouter ships, but nothing outside `openrouter.py` should import an
OpenRouter-specific type, so a second provider stays a new module rather than a
refactor of every caller.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LlmUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # `None` unless the provider reports it. A hardcoded price table would rot.
    cost_usd: float | None = None

    def __add__(self, other: "LlmUsage") -> "LlmUsage":
        costs = [value for value in (self.cost_usd, other.cost_usd) if value is not None]
        return LlmUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=sum(costs) if costs else None,
        )


@dataclass(frozen=True)
class ChatResult:
    content: str
    usage: LlmUsage = field(default_factory=LlmUsage)
    model: str = ""


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str = ""
    context_length: int | None = None
    prompt_price_usd_per_token: float | None = None
    completion_price_usd_per_token: float | None = None
