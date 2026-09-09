"""
Wellness scoring orchestrator.

Sequence per request: score each dimension -> aggregate and apply the condition
cap -> check the reliability gate -> build reminders -> generate the narrative.
Every rule lives in the module that owns it; this file only sequences them, so
no threshold, species norm, or prompt text belongs here.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from app.inference.protocols import Classifier, GeneratorMetadata
from app.llm.rotation import RotatingGeminiClient
from app.wellness.narrative import generate_narrative
from app.wellness.norms import (
    _BAND_LABELS,
    CALCULATION_VERSION,
    INSUFFICIENT_DATA_NARRATIVE,
    WELLNESS_DISCLAIMER,
)
from app.wellness.reminders import (
    _add_weight_recommendation,
    _filter_recommendations,
    _get_reminders,
    _get_safety_reminders,
)
from app.wellness.responses import WellnessBreakdown, WellnessResponse
from app.wellness.schemas import WellnessBand, WellnessRequest, WellnessScoreStatus
from app.wellness.scoring.activity import _score_activity
from app.wellness.scoring.aggregate import (
    _compute_reliability,
    _compute_score,
    _condition_cap,
    _get_band,
    _get_trend,
)
from app.wellness.scoring.baseline import (
    _medication_adherence,
    _score_baseline,
    _weight_stability,
)
from app.wellness.scoring.diet import _score_diet
from app.wellness.scoring.preventive import _score_preventive
from app.wellness.scoring.sleep import _score_sleep
from app.wellness.scoring.symptoms import _score_symptoms
from app.wellness.tracking import (
    _get_maintenance_tracking_recommendations,
    _get_routine_care_recommendations,
    _get_tracking_recommendations,
)

genai: Any
try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

logger = logging.getLogger(__name__)


class WellnessService:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        api_keys: list[str] | None = None,
    ) -> None:
        """`api_keys` (if given) takes priority over the single `api_key` and
        enables quota rotation: a 429 on one key retries on the next before
        falling back to the local template."""
        self.model_name = model_name
        self.client: Any = None
        keys = api_keys or ([api_key] if api_key else [])
        if keys and genai is not None:
            self.client = RotatingGeminiClient(keys)
        else:
            logger.warning("Gemini not available — /wellness will use fallback narratives.")

    @property
    def metadata(self) -> GeneratorMetadata:
        return GeneratorMetadata(
            backend="gemini" if self.client is not None else "fallback",
            model_name=self.model_name,
            available=self.client is not None,
        )

    def score(
        self, request: WellnessRequest, predictor: Classifier | None = None
    ) -> WellnessResponse:
        # Derived once here — several scorers and the prompt all need them.
        stability = _weight_stability(request.weight_history)
        adherence = _medication_adherence(request.active_medications)

        # 1. Score each dimension
        symptoms_item, detected_condition, detected_urgency = _score_symptoms(
            request.current_symptoms,
            predictor,
        )
        breakdown = WellnessBreakdown(
            activity=_score_activity(request.activity, request.pet.species),
            sleep=_score_sleep(request.activity, request.pet.species),
            diet=_score_diet(request.feeding, request.pet),
            symptoms=symptoms_item,
            preventive_care=_score_preventive(request.preventive_care, adherence),
            baseline=_score_baseline(request.pet, request.weight_history, stability),
        )
        data_coverage, score_status = _compute_reliability(breakdown)
        cap = _condition_cap(request.active_conditions)

        # Fields that do not depend on whether a score could be produced.
        common = {
            "score_status": score_status,
            "data_coverage": data_coverage,
            "calculation_version": CALCULATION_VERSION,
            "evaluated_at": datetime.now(UTC),
            "evaluation_window": request.evaluation_window,
            "breakdown": breakdown,
            "condition_cap": cap,
            "classifier_condition": detected_condition,
            "disclaimer": WELLNESS_DISCLAIMER,
        }

        if score_status == WellnessScoreStatus.INSUFFICIENT_DATA:
            return WellnessResponse(
                **common,
                wellness_score=None,
                band=None,
                band_label=None,
                trend=None,
                narrative=INSUFFICIENT_DATA_NARRATIVE,
                recommendations=[],
                reminders=_get_safety_reminders(detected_urgency),
                tracking_recommendations=[
                    *_get_tracking_recommendations(breakdown),
                    *_get_routine_care_recommendations(request),
                ],
            )

        # 2. Raw score (0-100), scaled for missing dimensions
        raw_score = _compute_score(breakdown)

        # 3. Apply condition cap
        final_score = int(min(raw_score, cap) if cap is not None else raw_score)
        final_score = max(0, min(100, final_score))

        band = _get_band(final_score)
        reminders = _get_reminders(request, breakdown, detected_urgency, adherence)
        if score_status == WellnessScoreStatus.COMPLETE and band in {
            WellnessBand.GOOD,
            WellnessBand.EXCELLENT,
        }:
            tracking_recommendations = _get_maintenance_tracking_recommendations(
                breakdown,
                reminders,
            )
        else:
            tracking_recommendations = _get_tracking_recommendations(breakdown)
        tracking_recommendations += _get_routine_care_recommendations(request)

        # 4. Gemini narrative + recommendations
        narrative, recommendations = generate_narrative(
            self.client,
            self.model_name,
            request=request,
            breakdown=breakdown,
            score=final_score,
            band=band,
            score_status=score_status,
            data_coverage=data_coverage,
            condition_cap=cap,
            detected_condition=detected_condition,
            reminders=reminders,
            stability=stability,
        )
        recommendations = _filter_recommendations(
            recommendations,
            reminders,
            breakdown,
        )
        recommendations = _add_weight_recommendation(recommendations, stability)

        return WellnessResponse(
            **common,
            wellness_score=final_score,
            band=band,
            band_label=_BAND_LABELS[band],
            trend=_get_trend(final_score, request.previous_score),
            narrative=narrative,
            recommendations=recommendations,
            reminders=reminders,
            tracking_recommendations=tracking_recommendations,
        )
