"""
Exception handlers.

Use-case errors are mapped to status codes here, once, instead of in a
`try/except` inside every route — a new route inherits the mapping for free.
Every response carries the request id from RequestLoggingMiddleware so a caller
can correlate a public error with the matching server-side log line.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.errors import InferenceUnavailableError, InvalidInputError

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str | None:
    request_id: str | None = getattr(request.state, "request_id", None)
    return request_id


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        # Preserve FastAPI's default `{"detail": ...}` shape — existing clients rely on it.
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "requestId": _request_id(request)},
            headers=exc.headers,
        )

    @app.exception_handler(InvalidInputError)
    async def invalid_input_handler(request: Request, exc: InvalidInputError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "requestId": _request_id(request)},
        )

    @app.exception_handler(InferenceUnavailableError)
    async def inference_unavailable_handler(
        request: Request, exc: InferenceUnavailableError
    ) -> JSONResponse:
        logger.warning(
            "Inference unavailable",
            extra={
                "fields": {
                    "event": "inference_unavailable",
                    "path": request.url.path,
                    "request_id": _request_id(request),
                }
            },
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Prediction failed.", "requestId": _request_id(request)},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Safety net for anything the handlers above do not cover. Never expose
        # exc's text to the caller; log it server-side instead.
        logger.exception(
            "Unhandled exception",
            extra={
                "fields": {
                    "event": "unhandled_exception",
                    "path": request.url.path,
                    "request_id": _request_id(request),
                }
            },
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "requestId": _request_id(request)},
        )
