"""
Shared timeout/retry policy for outbound Gemini `generate_content` calls.

Used by both `GeminiService` and `WellnessService` so failure semantics
(timeout, malformed response, transient error) are identical everywhere:
bound the wait, retry once on a transient error, and record the *actual*
reason a call fell back to the local template instead of silently
swallowing it (see runtime-safety-contracts spec: "Bounded generated-content
behavior").
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from typing import TypeVar

from app.observability import log_event

T = TypeVar("T")

GENERATION_TIMEOUT_SECONDS = 8.0
GENERATION_MAX_ATTEMPTS = 2
# Defense-in-depth cap on outbound prompt size, independent of request schema
# limits (schemas already bound individual fields; this bounds the assembled
# prompt as a whole, e.g. a long multi-turn chat transcript).
MAX_PROMPT_CHARS = 8000


class GenerationTimeoutError(Exception):
    """Raised when a generate_content call exceeds GENERATION_TIMEOUT_SECONDS."""


def bound_prompt(text: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    return text if len(text) <= max_chars else text[:max_chars]


def call_with_policy(
    fn: Callable[[], T],
    *,
    timeout: float = GENERATION_TIMEOUT_SECONDS,
    max_attempts: int = GENERATION_MAX_ATTEMPTS,
) -> T:
    """Run `fn` (a blocking `generate_content` call) with a hard timeout and a
    single retry on transient errors.

    A timeout is not retried — it already represents the worst-case wait, and
    retrying would double the request latency for no benefit.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn)
        try:
            result = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise GenerationTimeoutError(f"generate_content exceeded {timeout}s") from exc
        except Exception as exc:  # transient network/API error
            pool.shutdown(wait=True)
            last_exc = exc
            if attempt >= max_attempts:
                raise
        else:
            pool.shutdown(wait=True)
            return result
    assert last_exc is not None  # loop either records an exception or returns
    raise last_exc  # pragma: no cover - unreachable, loop above always returns or raises


def log_fallback(endpoint: str, reason: str, exc: Exception | None = None) -> None:
    """Record the actual reason a generator fell back to its local template."""
    log_event(
        "generator_fallback",
        endpoint=endpoint,
        reason=reason,
        error=str(exc) if exc else None,
    )
