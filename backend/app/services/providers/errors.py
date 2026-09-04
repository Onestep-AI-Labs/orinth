"""LLM provider error taxonomy.

The split exists because the retry policy differs per class: rate limits are
retried with backoff, auth and bad-request failures never are (retrying a bad
key just spends time), and a malformed response gets exactly one repair turn.
"""


class LlmError(RuntimeError):
    """Base class for every provider failure."""


class LlmAuthError(LlmError):
    """401/403. Never retried — the key is wrong and will stay wrong."""


class LlmRateLimitError(LlmError):
    """429. Retried with full-jitter backoff, honoring `Retry-After`."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LlmRequestError(LlmError):
    """4xx other than 401/403/429. Never retried."""


class LlmResponseError(LlmError):
    """The call succeeded but the body was not usable (bad JSON, empty choices)."""
