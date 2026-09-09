"""
Service construction.

Kept out of the API module so it can run at Lambda INIT (before any request)
as well as from the FastAPI lifespan locally. `ensure_services` is idempotent,
so a warm container skips the model load entirely.
"""

from __future__ import annotations

import logging

from app.app_services import AppServices
from app.config import Settings, get_settings
from app.inference.predictor import Predictor
from app.inference.protocols import Classifier
from app.llm.gemini_service import GeminiService
from app.observability import log_event
from app.wellness.service import WellnessService

logger = logging.getLogger(__name__)


def build_services(settings: Settings | None = None) -> AppServices:
    """Load the model + services. Called at Lambda INIT (and by the keep-warm
    ping) so a warm container has the model already loaded, instead of paying
    the load on the first real request."""
    settings = settings or get_settings()

    # Backend selection: torch (default) or quantized ONNX (cheaper on Lambda).
    predictor: Classifier
    if settings.model_backend == "onnx":
        from app.inference.onnx_predictor import OnnxPredictor

        predictor = OnnxPredictor.from_paths(model_path=settings.model_path)
    else:
        predictor = Predictor.from_paths(model_path=settings.model_path)

    gemini_api_keys = settings.resolved_gemini_api_keys()
    services = AppServices(
        predictor=predictor,
        gemini_service=GeminiService(api_keys=gemini_api_keys, model_name=settings.gemini_model),
        wellness_service=WellnessService(
            api_keys=gemini_api_keys, model_name=settings.gemini_model
        ),
        low_confidence_threshold=settings.low_confidence_threshold,
        # When true, /predict serves a cautious templated explanation instead of
        # calling Gemini — zero per-request API cost. See README / cost notes.
        use_static_explanations=settings.use_static_explanations,
    )
    metadata = services.predictor.metadata
    log_event(
        "model_loaded",
        backend=metadata.backend,
        model_version=metadata.model_version,
        label_count=len(metadata.labels),
    )
    return services
