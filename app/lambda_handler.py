"""
AWS Lambda entry point.

Wraps the FastAPI app with Mangum so API Gateway HTTP events are translated
into ASGI requests. The FastAPI lifespan (model loading) runs on the first
cold-start and is then cached for the lifetime of the container.

EventBridge keep-warm pings (scheduled events) are swallowed here so Mangum
doesn't raise an error on non-HTTP payloads.
"""
from __future__ import annotations

from typing import Any

from mangum import Mangum

from app.main import app

_mangum_handler = Mangum(app, lifespan="auto")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    if "httpMethod" not in event and "requestContext" not in event:
        return {"statusCode": 200, "body": "warm"}

    return _mangum_handler(event, context)
