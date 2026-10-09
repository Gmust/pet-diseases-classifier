"""Liveness and readiness probes. Always unauthenticated."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/live")
def liveness() -> dict[str, str]:
    """Process is up and able to serve HTTP. Does not check the model."""
    return {"status": "ok"}


@router.get("/health/ready")
def readiness(request: Request, response: Response) -> dict[str, object]:
    """Model bundle is loaded and validated. No filesystem paths or secrets exposed."""
    services = getattr(request.app.state, "services", None)
    if services is None:
        response.status_code = 503
        return {"status": "not_ready", "reason": "model_not_loaded"}
    meta = services.predictor.metadata
    return {
        "status": "ready",
        "backend": meta.backend,
        "modelVersion": meta.model_version,
        "labelCount": len(meta.labels),
    }
