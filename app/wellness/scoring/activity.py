"""Activity dimension: steps and active minutes against species targets."""

from __future__ import annotations

import logging

from app.wellness.norms import (
    _ACTIVITY_TARGETS,
    _DEFAULT_ACTIVITY,
    _clamp,
    _norm,
)
from app.wellness.responses import (
    WellnessBreakdownItem,
)
from app.wellness.schemas import (
    WellnessActivity,
    WellnessDimensionAvailability,
    WellnessReasonCode,
)
from app.wellness.scoring.common import _breakdown_item

logger = logging.getLogger(__name__)


def _score_activity(
    activity: WellnessActivity | None,
    species: str,
) -> WellnessBreakdownItem:
    """Score activity while reporting missing and inapplicable data explicitly."""
    MAX = 20.0
    target = _ACTIVITY_TARGETS.get(_norm(species), _DEFAULT_ACTIVITY)
    if target["steps"] == 0 and target["active_min"] == 0:
        return _breakdown_item(
            score=0,
            max_score=MAX,
            availability=WellnessDimensionAvailability.NOT_APPLICABLE,
            reason_codes=[WellnessReasonCode.ACTIVITY_NOT_APPLICABLE],
        )
    if activity is None:
        return _breakdown_item(
            score=0,
            max_score=MAX,
            availability=WellnessDimensionAvailability.MISSING,
            reason_codes=[WellnessReasonCode.ACTIVITY_DATA_MISSING],
        )

    earned = 0.0
    possible = 0.0
    evidence: dict[str, bool | int | float | str] = {}

    # Steps sub-score (10 pts) — only when applicable and tracked.
    if target["steps"] > 0 and activity.avg_steps_per_day is not None:
        possible += 10
        ratio = _clamp(activity.avg_steps_per_day / target["steps"])
        earned += ratio * 10
        evidence["avgStepsPerDay"] = activity.avg_steps_per_day
        evidence["stepTarget"] = target["steps"]

    # Active minutes sub-score (10 pts) — only when applicable and tracked.
    if target["active_min"] > 0 and activity.avg_active_minutes_per_day is not None:
        possible += 10
        ratio = _clamp(activity.avg_active_minutes_per_day / target["active_min"])
        earned += ratio * 10
        evidence["avgActiveMinutesPerDay"] = activity.avg_active_minutes_per_day
        evidence["activeMinuteTarget"] = target["active_min"]

    if possible == 0:
        return _breakdown_item(
            score=0,
            max_score=MAX,
            availability=WellnessDimensionAvailability.MISSING,
            reason_codes=[WellnessReasonCode.ACTIVITY_DATA_MISSING],
        )

    # Scale earned to MAX
    scaled = (earned / possible) * MAX
    reason = (
        WellnessReasonCode.ACTIVITY_TARGET_MET
        if scaled / MAX >= 0.75
        else WellnessReasonCode.ACTIVITY_BELOW_TARGET
    )
    return _breakdown_item(
        score=round(scaled, 1),
        max_score=MAX,
        availability=WellnessDimensionAvailability.AVAILABLE,
        reason_codes=[reason],
        evidence=evidence,
    )
