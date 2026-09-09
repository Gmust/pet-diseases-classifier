"""Narrative generation: the Gemini call and its deterministic fallback.

Score, breakdown and reminders are all computed before this runs, so any
failure here degrades prose only — never the numbers.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from pydantic import ValidationError

from app.llm.generation_policy import (
    GenerationTimeoutError,
    bound_prompt,
    call_with_policy,
    log_fallback,
)
from app.wellness.prompt import (
    _NARRATIVE_SYSTEM,
    _build_narrative_prompt,
    _shorten_narrative,
    _WellnessNarrative,
)
from app.wellness.responses import (
    WellnessBreakdown,
    WellnessReminder,
)
from app.wellness.schemas import (
    WellnessBand,
    WellnessRequest,
    WellnessScoreStatus,
)

logger = logging.getLogger(__name__)


_FALLBACK_NARRATIVES: dict[WellnessBand, str] = {
    WellnessBand.EXCELLENT: "Your pet is in excellent shape based on this week's tracked data. Keep up the great routine!",
    WellnessBand.GOOD: "Your pet is doing well overall. There are a few small areas worth improving.",
    WellnessBand.FAIR: "Your pet's wellness is fair. Some dimensions need attention — check the breakdown above.",
    WellnessBand.CONCERNING: "Your pet's wellness is concerning this week. Consider reviewing diet, activity, and scheduling a vet check.",
    WellnessBand.CRITICAL: "Your pet's tracked data indicates a critical wellness level. Please consult a veterinarian promptly.",
}

_FALLBACK_RECOMMENDATIONS: dict[WellnessBand, tuple[str, ...]] = {
    WellnessBand.EXCELLENT: (
        "Maintain the current routine.",
        "Schedule a routine annual vet check.",
    ),
    WellnessBand.GOOD: (
        "Review the dimension with the lowest sub-score.",
        "Ensure consistent meal timing.",
    ),
    WellnessBand.FAIR: (
        "Increase daily active time.",
        "Log feeding more consistently.",
        "Book a vet appointment if symptoms persist.",
    ),
    WellnessBand.CONCERNING: (
        "Schedule a veterinary check-up soon.",
        "Improve feeding consistency.",
        "Increase monitored exercise.",
    ),
    WellnessBand.CRITICAL: (
        "Contact a veterinarian as soon as possible.",
        "Monitor symptoms closely.",
        "Avoid strenuous activity until assessed.",
    ),
}


def _fallback_narrative(
    band: WellnessBand,
    score_status: WellnessScoreStatus,
    data_coverage: float,
) -> tuple[str, list[str]]:
    narrative = _FALLBACK_NARRATIVES[band]
    if score_status == WellnessScoreStatus.PARTIAL:
        narrative = (
            f"This is a partial wellness assessment based on "
            f"{data_coverage:.0%} weighted data coverage. {narrative}"
        )
    return narrative, list(_FALLBACK_RECOMMENDATIONS[band])


def generate_narrative(
    client: Any,
    model_name: str,
    *,
    request: WellnessRequest,
    breakdown: WellnessBreakdown,
    score: int,
    band: WellnessBand,
    score_status: WellnessScoreStatus,
    data_coverage: float,
    condition_cap: int | None,
    detected_condition: str | None,
    reminders: list[WellnessReminder],
    stability: float | None,
) -> tuple[str, list[str]]:
    """Ask Gemini for the narrative + recommendations.

    Takes the client rather than owning it, so the scoring path stays free of
    LLM concerns and this is callable in isolation. Any failure falls back to
    the deterministic band-based narrative.
    """
    fallback = partial(_fallback_narrative, band, score_status, data_coverage)
    if client is None:
        return fallback()

    prompt = _build_narrative_prompt(
        request,
        breakdown,
        score,
        band,
        score_status,
        data_coverage,
        condition_cap,
        detected_condition,
        reminders,
        stability,
    )
    try:
        response = call_with_policy(
            lambda: client.models.generate_content(
                model=model_name,
                contents=bound_prompt(prompt),
                config={
                    "temperature": 0.35,
                    "system_instruction": _NARRATIVE_SYSTEM,
                    "response_mime_type": "application/json",
                    "response_json_schema": _WellnessNarrative.model_json_schema(),
                },
            )
        )
        if not response.text:
            raise ValueError("Empty Gemini response.")
        parsed = _WellnessNarrative.model_validate_json(response.text)
        return _shorten_narrative(parsed.narrative), parsed.recommendations
    except GenerationTimeoutError as exc:
        log_fallback("wellness.narrative", "timeout", exc)
        return fallback()
    except (ValidationError, ValueError) as exc:
        log_fallback("wellness.narrative", "invalid_response", exc)
        return fallback()
    except Exception as exc:  # pragma: no cover
        log_fallback("wellness.narrative", "request_error", exc)
        return fallback()
