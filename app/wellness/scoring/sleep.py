"""Sleep dimension: average sleep hours against species norms."""

from __future__ import annotations

import logging

from app.wellness.norms import (
    _DEFAULT_SLEEP,
    _SLEEP_NORMS,
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


def _score_sleep(
    activity: WellnessActivity | None,
    species: str,
) -> WellnessBreakdownItem:
    MAX = 15.0
    norms = _SLEEP_NORMS.get(_norm(species), _DEFAULT_SLEEP)
    if norms is None:
        return _breakdown_item(
            score=0,
            max_score=MAX,
            availability=WellnessDimensionAvailability.NOT_APPLICABLE,
            reason_codes=[WellnessReasonCode.SLEEP_NOT_APPLICABLE],
        )
    if activity is None or activity.avg_sleep_hours_per_day is None:
        return _breakdown_item(
            score=0,
            max_score=MAX,
            availability=WellnessDimensionAvailability.MISSING,
            reason_codes=[WellnessReasonCode.SLEEP_DATA_MISSING],
        )

    lo, hi = norms
    hours = activity.avg_sleep_hours_per_day
    mid = (lo + hi) / 2
    spread = (hi - lo) / 2

    deviation = abs(hours - mid) / spread  # 0 = perfect, 1 = at boundary, >1 = outside
    if deviation <= 0.2:
        score = MAX
    elif deviation <= 0.6:
        score = MAX * 0.8
    elif deviation <= 1.0:
        score = MAX * 0.55
    elif deviation <= 1.5:
        score = MAX * 0.3
    else:
        score = MAX * 0.1

    reason = (
        WellnessReasonCode.SLEEP_WITHIN_RANGE
        if lo <= hours <= hi
        else WellnessReasonCode.SLEEP_OUTSIDE_RANGE
    )
    return _breakdown_item(
        score=round(score, 1),
        max_score=MAX,
        availability=WellnessDimensionAvailability.AVAILABLE,
        reason_codes=[reason],
        evidence={
            "avgSleepHoursPerDay": hours,
            "healthyMinimumHours": lo,
            "healthyMaximumHours": hi,
        },
    )
