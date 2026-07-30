"""
AWS Lambda entry point.

Wraps the FastAPI app with Mangum so API Gateway HTTP events are translated
into ASGI requests.

Model loading happens at INIT (module import) via ensure_services(), so the model
is in memory before the first request. Non-HTTP invocations are handled safely,
but no scheduled keep-warm event is deployed.
"""

from __future__ import annotations

from typing import Any

from mangum import Mangum

from app.main import app, ensure_services

# Load the model during Lambda INIT (full-power CPU, before any request). Without
# this the model only loads on the first real HTTP request, making it slow even
# on a "warm" container.
ensure_services()

_mangum_handler = Mangum(app, lifespan="auto")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    # Ignore non-HTTP invocations rather than passing them to Mangum.
    if "httpMethod" not in event and "requestContext" not in event:
        ensure_services()
        return {"statusCode": 200, "body": "warm"}

    return _mangum_handler(event, context)
