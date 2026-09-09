"""Shared route dependencies: authentication and the services accessor."""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.app_services import AppServices
from app.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def api_key_auth(api_key: str | None = Security(api_key_header)) -> None:
    """No-op when API_KEY is unset, so local dev needs no header."""
    expected_api_key = get_settings().api_key
    if not expected_api_key:
        return
    if not api_key or not hmac.compare_digest(api_key, expected_api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")


def get_services(request: Request) -> AppServices:
    """Services built once at startup and cached on app.state."""
    services: AppServices = request.app.state.services
    return services
