"""Baseline dimension: medication adherence and weight stability."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.wellness.norms import (
    _clamp,
)
from app.wellness.responses import (
    WellnessBreakdownItem,
)
from app.wellness.schemas import (
    WellnessDimensionAvailability,
    WellnessMedication,
    WellnessPet,
    WellnessReasonCode,
    WellnessWeightMeasurement,
)
from app.wellness.scoring.common import _breakdown_item

logger = logging.getLogger(__name__)


def _medication_adherence(medications: list[WellnessMedication]) -> float | None:
    scheduled = 0
    completed = 0
    for medication in medications:
        doses = medication.scheduled_doses
        if doses is None or doses <= 0:
            continue
        scheduled += doses
        completed += min(medication.completed_doses or 0, doses)

    if scheduled == 0:
        return None
    return _clamp(completed / scheduled)


def _weight_stability(history: list[WellnessWeightMeasurement]) -> float | None:
    if len(history) < 2:
        return None

    def normalized_timestamp(measurement: WellnessWeightMeasurement) -> datetime:
        measured_at = measurement.measured_at
        if measured_at.tzinfo is None or measured_at.utcoffset() is None:
            return measured_at.replace(tzinfo=UTC)
        return measured_at.astimezone(UTC)

    ordered = sorted(history, key=normalized_timestamp)
    oldest = ordered[0].weight_kg
    newest = ordered[-1].weight_kg
    change_ratio = abs(newest - oldest) / oldest

    if change_ratio <= 0.03:
        return 1.0
    if change_ratio <= 0.05:
        return 0.8
    if change_ratio <= 0.10:
        return 0.4
    return 0.0


def _score_baseline(
    pet: WellnessPet,
    weight_history: list[WellnessWeightMeasurement],
    stability: float | None,
) -> WellnessBreakdownItem:
    MAX = 10.0
    earned = 0.0
    possible = 0.0

    # Age factor (5 pts): tracked age earns full points through 10 years,
    # with slight leniency for seniors.
    if pet.age_months is not None:
        possible += 5
        if pet.age_months <= 120:
            earned += 5
        else:
            earned += 4

    if stability is not None:
        possible += 5
        earned += stability * 5

    if possible == 0:
        return _breakdown_item(
            score=0,
            max_score=MAX,
            availability=WellnessDimensionAvailability.MISSING,
            reason_codes=[WellnessReasonCode.BASELINE_DATA_MISSING],
        )

    scaled = (earned / possible) * MAX
    reason = (
        WellnessReasonCode.BASELINE_STABLE
        if scaled / MAX >= 0.8
        else WellnessReasonCode.BASELINE_NEEDS_ATTENTION
    )
    evidence: dict[str, bool | int | float | str] = {
        "weightMeasurementCount": len(weight_history),
    }
    if pet.age_months is not None:
        evidence["ageMonths"] = pet.age_months
    if stability is not None:
        evidence["weightStability"] = round(stability, 4)
    return _breakdown_item(
        score=round(scaled, 1),
        max_score=MAX,
        availability=WellnessDimensionAvailability.AVAILABLE,
        reason_codes=[reason],
        evidence=evidence,
    )
