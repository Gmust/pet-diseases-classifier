"""
FastAPI application assembly.

Deliberately import-light: this runs during Lambda INIT, so anything heavy
belongs behind `build_services` in app.bootstrap instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.errors import register_error_handlers
from app.api.routes import chat, feeding, health, predict, wellness
from app.app_services import AppServices
from app.bootstrap import build_services
from app.config import get_settings
from app.observability import RequestLoggingMiddleware, configure_logging

load_dotenv()
configure_logging(get_settings().log_level)


def ensure_services() -> AppServices:
    """Idempotently load services onto app.state (safe to call multiple times)."""
    if getattr(app.state, "services", None) is None:
        app.state.services = build_services()
    services: AppServices = app.state.services
    return services


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ensure_services()  # no-op if already loaded at INIT (Lambda) — loads on first run (uvicorn)
    yield


app = FastAPI(
    title="Pet Care AI Microservice",
    description="Classifier-based pet condition pre-assessment with Gemini-generated explanation.",
    version="1.0.0",
    lifespan=lifespan,
    # ROOT_PATH tells FastAPI it is mounted behind a proxy at this prefix.
    # Set to "/Prod" on Lambda (API Gateway stage), leave empty for local dev.
    root_path=get_settings().root_path,
)
app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

for module in (health, predict, chat, wellness, feeding):
    app.include_router(module.router)
