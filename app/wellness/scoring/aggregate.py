"""Combine dimension items into a score: condition caps, weighted total,
reliability/coverage, band, and trend."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from app.wellness.norms import (
    _CONDITION_CAP_KEYWORDS,
    _FOUNDATIONAL_DIMENSIONS,
    MIN_SCORE_DATA_COVERAGE,
)
from app.wellness.responses import (
    WellnessBreakdown,
    WellnessBreakdownItem,
)
from app.wellness.schemas import (
    TrendDirection,
    WellnessBand,
    WellnessCondition,
    WellnessDimension,
    WellnessDimensionAvailability,
    WellnessScoreStatus,
)

logger = logging.getLogger(__name__)


def _condition_cap(conditions: list[WellnessCondition]) -> int | None:
    """Return the lowest cap imposed by any active condition, or None."""
    lowest_cap: int | None = None
    for condition in conditions:
        name_lower = condition.name.lower()
        for keywords, cap in _CONDITION_CAP_KEYWORDS:
            if any(kw in name_lower for kw in keywords):
                if lowest_cap is None or cap < lowest_cap:
                    lowest_cap = cap
                break
    return lowest_cap


def _compute_score(breakdown: WellnessBreakdown) -> float:
    """Scale the raw earned points against the max of present dimensions."""
    included_items = [item for _, item in _dimension_items(breakdown) if item.included]
    total_earned = sum(item.score for item in included_items)
    total_max = sum(item.max_score for item in included_items)
    if total_max == 0:
        return 0.0
    return (total_earned / total_max) * 100


def _dimension_items(
    breakdown: WellnessBreakdown,
) -> Iterator[tuple[WellnessDimension, WellnessBreakdownItem]]:
    yield WellnessDimension.ACTIVITY, breakdown.activity
    yield WellnessDimension.SLEEP, breakdown.sleep
    yield WellnessDimension.DIET, breakdown.diet
    yield WellnessDimension.SYMPTOMS, breakdown.symptoms
    yield WellnessDimension.PREVENTIVE_CARE, breakdown.preventive_care
    yield WellnessDimension.BASELINE, breakdown.baseline


def _compute_reliability(
    breakdown: WellnessBreakdown,
) -> tuple[float, WellnessScoreStatus]:
    by_dimension = dict(_dimension_items(breakdown))
    items = by_dimension.values()

    denominator = sum(item.max_score for item in items if item.applicable)
    numerator = sum(item.max_score for item in items if item.included)
    coverage = round(numerator / denominator, 4) if denominator else 0.0

    def included(*dimensions: WellnessDimension) -> int:
        return sum(by_dimension[dimension].included for dimension in dimensions)

    behavior = (
        by_dimension[WellnessDimension.ACTIVITY],
        by_dimension[WellnessDimension.SLEEP],
    )
    reliability_gate_met = (
        coverage >= MIN_SCORE_DATA_COVERAGE
        and included(*_FOUNDATIONAL_DIMENSIONS) >= 3
        and by_dimension[WellnessDimension.DIET].included
        and (
            any(item.included for item in behavior) or not any(item.applicable for item in behavior)
        )
        and included(
            WellnessDimension.PREVENTIVE_CARE,
            WellnessDimension.BASELINE,
        )
        > 0
    )

    if not reliability_gate_met:
        return coverage, WellnessScoreStatus.INSUFFICIENT_DATA
    if any(item.availability == WellnessDimensionAvailability.MISSING for item in items):
        return coverage, WellnessScoreStatus.PARTIAL
    return coverage, WellnessScoreStatus.COMPLETE


def _get_band(score: int) -> WellnessBand:
    if score >= 90:
        return WellnessBand.EXCELLENT
    if score >= 75:
        return WellnessBand.GOOD
    if score >= 60:
        return WellnessBand.FAIR
    if score >= 40:
        return WellnessBand.CONCERNING
    return WellnessBand.CRITICAL


def _get_trend(current: int, previous: int | None) -> TrendDirection | None:
    if previous is None:
        return None
    diff = current - previous
    if diff > 3:
        return TrendDirection.IMPROVING
    if diff < -3:
        return TrendDirection.DECLINING
    return TrendDirection.STABLE
