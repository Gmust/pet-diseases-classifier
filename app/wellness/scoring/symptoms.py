"""Symptom dimension: current symptoms routed through the classifier."""

from __future__ import annotations

import logging

from app.domain.conditions import get_condition_metadata
from app.domain.enums import UrgencyLevel
from app.inference.protocols import Classifier
from app.wellness.norms import (
    _URGENCY_BASE_SCORE,
    _clamp,
)
from app.wellness.responses import (
    WellnessBreakdownItem,
)
from app.wellness.schemas import (
    WellnessDimensionAvailability,
    WellnessReasonCode,
)
from app.wellness.scoring.common import _breakdown_item

logger = logging.getLogger(__name__)


def _score_symptoms(
    symptoms_text: str | None,
    predictor: Classifier | None,
) -> tuple[WellnessBreakdownItem, str | None, UrgencyLevel | None]:
    """Return the symptom score, detected condition, and its clinical urgency."""
    MAX = 25.0
    if not symptoms_text:
        return (
            _breakdown_item(
                score=0,
                max_score=MAX,
                availability=WellnessDimensionAvailability.NOT_APPLICABLE,
                reason_codes=[WellnessReasonCode.SYMPTOMS_NOT_REPORTED],
            ),
            None,
            None,
        )
    if predictor is None:
        return (
            _breakdown_item(
                score=0,
                max_score=MAX,
                availability=WellnessDimensionAvailability.MISSING,
                reason_codes=[WellnessReasonCode.SYMPTOM_CLASSIFIER_UNAVAILABLE],
            ),
            None,
            None,
        )

    try:
        prediction = predictor.predict(symptoms_text)
    except Exception as exc:
        logger.warning("Wellness symptom classifier failed: %s", exc)
        return (
            _breakdown_item(
                score=0,
                max_score=MAX,
                availability=WellnessDimensionAvailability.MISSING,
                reason_codes=[WellnessReasonCode.SYMPTOM_CLASSIFIER_FAILED],
            ),
            None,
            None,
        )

    meta = get_condition_metadata(prediction.predicted_condition)
    base = _URGENCY_BASE_SCORE[meta.urgency]

    # High confidence of a mild condition → slight bonus; bad condition → stays low
    if meta.urgency == UrgencyLevel.MONITOR:
        score = base + prediction.confidence * 4  # up to 25
    elif meta.urgency == UrgencyLevel.EMERGENCY:
        score = base + (1 - prediction.confidence) * 3  # stays near 0-5
    else:
        score = base + (1 - prediction.confidence) * 3  # slight leniency for uncertainty

    return (
        _breakdown_item(
            score=round(_clamp(score, 0, MAX), 1),
            max_score=MAX,
            availability=WellnessDimensionAvailability.AVAILABLE,
            reason_codes=[WellnessReasonCode.SYMPTOM_RESULT_AVAILABLE],
            evidence={
                "confidence": round(prediction.confidence, 4),
                "urgency": meta.urgency.value,
            },
        ),
        prediction.predicted_condition,
        meta.urgency,
    )
