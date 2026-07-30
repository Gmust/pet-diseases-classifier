"""
Lightweight, dependency-free observability for the microservice.

- `configure_logging()` installs a JSON log formatter. On Lambda/CloudWatch,
  one JSON object per line is directly queryable in CloudWatch Logs Insights
  (e.g. filter by event="prediction" and stats avg(confidence)).
- `RequestLoggingMiddleware` logs one structured line per HTTP request with
  method, path, status and latency.
- `log_event(...)` emits an arbitrary structured metric/event line.

Stdlib only — no extra runtime dependency, no impact on image size or cold start.
If you later want first-class CloudWatch metrics, these JSON lines map cleanly
onto AWS Lambda Powertools or EMF.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-Id"

logger = logging.getLogger("petcare")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any structured fields attached via `extra={"fields": {...}}`.
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    log_level = (level or "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)


def log_event(event: str, **fields: Any) -> None:
    """Emit a structured event line, e.g. log_event('prediction', condition=..., confidence=...)."""
    logger.info(event, extra={"fields": {"event": event, **fields}})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status code and latency for every request.

    Also assigns a per-request id (`request.state.request_id`), echoed back as
    the `X-Request-Id` response header, so a stable error response can point
    the caller/operator at the matching server-side log line without exposing
    the underlying exception."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            logger.info(
                "request",
                extra={
                    "fields": {
                        "event": "request",
                        "method": request.method,
                        "path": request.url.path,
                        "status": status_code,
                        "duration_ms": duration_ms,
                        "request_id": request_id,
                    }
                },
            )
