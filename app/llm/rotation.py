"""
Multi-key rotation for the Gemini client.

Free-tier Gemini keys carry per-key quota. When a key is exhausted, the SDK
raises `ClientError` with HTTP 429. Rather than falling back to the local
template on the very first quota hit, rotate to the next configured key and
retry the same call — only fall back once every key is exhausted.

Any other error (bad request, server error, etc.) is not a quota problem and
is re-raised immediately so the existing `call_with_policy` retry/fallback
path in `GeminiService`/`WellnessService` handles it unchanged.
"""

from __future__ import annotations

from typing import Any

from app.observability import log_event

genai: Any
ClientError: Any
try:
    from google import genai
    from google.genai.errors import ClientError
except ImportError:  # pragma: no cover - runtime guard for missing dependency
    genai = None
    ClientError = None

QUOTA_EXCEEDED_STATUS = 429


class _RotatingModels:
    def __init__(self, parent: RotatingGeminiClient) -> None:
        self._parent = parent

    def generate_content(self, **kwargs: Any) -> Any:
        parent = self._parent
        last_exc: Exception | None = None
        for _ in range(len(parent._clients)):
            key_index = parent._index
            client = parent._clients[key_index]
            try:
                return client.models.generate_content(**kwargs)
            except ClientError as exc:
                if exc.code != QUOTA_EXCEEDED_STATUS:
                    raise
                last_exc = exc
                log_event("gemini_key_quota_exhausted", key_index=key_index)
                parent._index = (key_index + 1) % len(parent._clients)
        assert last_exc is not None
        raise last_exc


class RotatingGeminiClient:
    """Drop-in replacement for `genai.Client` backed by multiple API keys.

    Exposes `.models.generate_content(...)` like a real client so callers
    (`GeminiService`, `WellnessService`) don't need to change their call sites.
    """

    def __init__(self, api_keys: list[str]) -> None:
        if not api_keys:
            raise ValueError("RotatingGeminiClient requires at least one API key.")
        self._clients = [genai.Client(api_key=key) for key in api_keys]
        self._index = 0
        self.models = _RotatingModels(self)
