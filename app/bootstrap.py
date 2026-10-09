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
from app.inference.model_validation import ModelValidationError
from app.inference.predictor import Predictor
from app.inference.protocols import Classifier
from app.llm.gemini_service import GeminiService
from app.observability import log_event
from app.wellness.service import WellnessService

logger = logging.getLogger(__name__)

# Unambiguous owner texts any acceptable model classifies correctly. A model can
# load cleanly and still collapse on the runtime CPU (ADR 0004: per-channel int8
# saturating on the Lambda AVX2 hosts sent every input to one class), so a missed
# canary fails startup: every route then errors and the Errors alarm fires, instead
# of /predict and /chat silently serving one class.
MODEL_CANARY: tuple[tuple[str, str], ...] = (
    ("My dog has been vomiting and has had diarrhea since yesterday.", "Digestive Issues"),
    ("My cat keeps shaking her head and scratching at her ears.", "Ear Conditions"),
    (
        "My dog is limping on his back leg after jumping off the couch.",
        "Musculoskeletal Conditions",
    ),
)


def failed_canaries(predictor: Classifier) -> list[str]:
    """Expected labels of the canary texts the predictor gets wrong."""
    return [
        expected
        for text, expected in MODEL_CANARY
        if predictor.predict(text).predicted_condition != expected
    ]


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

    metadata = predictor.metadata
    canary_failures = failed_canaries(predictor)
    if canary_failures:
        logger.error(
            "model_canary_failed",
            extra={
                "fields": {
                    "event": "model_canary_failed",
                    "model_version": metadata.model_version,
                    "missed": canary_failures,
                }
            },
        )
        raise ModelValidationError(
            f"Model {metadata.model_version} misclassified canary texts: {canary_failures}"
        )

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
    log_event(
        "model_loaded",
        backend=metadata.backend,
        model_version=metadata.model_version,
        label_count=len(metadata.labels),
        canary_passed=True,
    )
    return services
