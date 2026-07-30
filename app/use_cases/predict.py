"""Application use case for POST /predict — orchestrates the classifier,
deterministic safety layer, and explanation generation. No FastAPI/HTTP
concerns here; `app.main` maps `UseCaseError` subclasses to HTTP responses."""

from __future__ import annotations

import logging

from app.app_services import AppServices
from app.ml.condition_metadata import build_static_explanation, get_condition_metadata
from app.observability import log_event
from app.schemas import PredictRequest, PredictResponse
from app.services.advice_safety import sanitize_advice
from app.services.gemini_service import DEFAULT_DISCLAIMER
from app.services.triage_safety import (
    EMERGENCY_HOME_ADVICE,
    apply_red_flag_urgency,
    emergency_explanation,
)
from app.use_cases.errors import InferenceUnavailableError, InvalidInputError

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_NOTE = (
    "Model confidence is limited for this prediction. Monitor your pet closely and seek "
    "veterinary advice."
)


def run_predict(payload: PredictRequest, services: AppServices) -> PredictResponse:
    try:
        prediction = services.predictor.predict(payload.text)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Prediction failed",
            extra={"fields": {"event": "prediction_error", "endpoint": "predict"}},
        )
        raise InferenceUnavailableError("predict") from exc

    meta = get_condition_metadata(prediction.predicted_condition)
    urgency, red_flag = apply_red_flag_urgency(payload.text, meta.urgency)

    if red_flag.triggered:
        # Emergency: override condition-specific content (the predicted class may be
        # wrong) and skip Gemini — lead with emergency message + first-aid advice.
        explanation = emergency_explanation(red_flag.reason)
        disclaimer = DEFAULT_DISCLAIMER
        home_advice = list(EMERGENCY_HOME_ADVICE)
        gemini_used = False
    elif services.use_static_explanations:
        # Cost path: skip Gemini and serve a cautious templated explanation.
        explanation = build_static_explanation(prediction.predicted_condition, meta)
        disclaimer = DEFAULT_DISCLAIMER
        home_advice = list(meta.home_advice)
        gemini_used = False
    else:
        explanation_payload = services.gemini_service.generate_explanation(
            user_text=payload.text,
            predicted_condition=prediction.predicted_condition,
            default_home_advice=list(meta.home_advice),
        )
        explanation = explanation_payload.explanation
        disclaimer = explanation_payload.disclaimer or DEFAULT_DISCLAIMER
        home_advice = sanitize_advice(
            explanation_payload.home_advice,
            fallback=list(meta.home_advice),
            source="predict.explanation",
        )
        gemini_used = True

    # Low-confidence note — only when this is NOT an emergency override.
    if (
        not red_flag.triggered
        and prediction.confidence < services.low_confidence_threshold
        and LOW_CONFIDENCE_NOTE not in explanation
    ):
        explanation = f"{explanation} {LOW_CONFIDENCE_NOTE}"

    log_event(
        "prediction",
        endpoint="predict",
        condition=prediction.predicted_condition,
        confidence=round(prediction.confidence, 4),
        low_confidence=prediction.confidence < services.low_confidence_threshold,
        red_flag=red_flag.reason,
        gemini_used=gemini_used,
    )

    return PredictResponse(
        predicted_condition=prediction.predicted_condition,
        confidence=round(prediction.confidence, 4),
        explanation=explanation,
        disclaimer=disclaimer,
        urgency=urgency,
        specialist=meta.specialist,
        disease_category=meta.disease_category,
        home_advice=home_advice,
    )
