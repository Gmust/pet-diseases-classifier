"""Diet dimension: calorie intake against computed need, plus consistency."""

from __future__ import annotations

import logging

from app.wellness.norms import (
    _DEFAULT_KCAL_PER_KG,
    _KCAL_PER_KG,
    _clamp,
    _norm,
)
from app.wellness.responses import (
    WellnessBreakdownItem,
)
from app.wellness.schemas import (
    WellnessDimensionAvailability,
    WellnessFeeding,
    WellnessPet,
    WellnessReasonCode,
)
from app.wellness.scoring.common import _breakdown_item

logger = logging.getLogger(__name__)


def _score_diet(
    feeding: WellnessFeeding | None,
    pet: WellnessPet,
) -> WellnessBreakdownItem:
    MAX = 20.0
    if feeding is None or (
        feeding.avg_meals_per_day is None
        and feeding.avg_calories_per_day is None
        and not feeding.food_types
        and feeding.consistency_days == 0
    ):
        return _breakdown_item(
            score=0,
            max_score=MAX,
            availability=WellnessDimensionAvailability.MISSING,
            reason_codes=[WellnessReasonCode.DIET_DATA_MISSING],
        )

    earned = 0.0
    possible = 0.0
    calorie_weight: float | None = None
    calorie_target: float | None = None
    calorie_ratio: float | None = None

    # Consistency (6 pts): how many of last 7 days had feeding logs
    possible += 6
    if feeding.consistency_days > 0:
        earned += _clamp(feeding.consistency_days / 7) * 6

    # Meal frequency (6 pts): only counts when provided
    if feeding.avg_meals_per_day is not None:
        possible += 6
        mpd = feeding.avg_meals_per_day
        if 1.8 <= mpd <= 3.2:
            earned += 6
        elif 1.0 <= mpd < 1.8 or 3.2 < mpd <= 4.0:
            earned += 4
        else:
            earned += 2

    # Variety (2 pts): only counts when food types are known
    if feeding.food_types:
        possible += 2
        earned += 2 if len(feeding.food_types) >= 2 else 1

    # Calorie fit (6 pts): only if weight and calories are known
    if pet.weight_kg and feeding.avg_calories_per_day:
        possible += 6
        kcal_per_kg = _KCAL_PER_KG.get(_norm(pet.species), _DEFAULT_KCAL_PER_KG)
        calorie_weight = pet.weight_kg
        calorie_target = kcal_per_kg * calorie_weight
        if calorie_target > 0:
            calorie_ratio = feeding.avg_calories_per_day / calorie_target
            # ratio=1.0 is perfect; penalise deviation
            deviation = abs(calorie_ratio - 1.0)
            if deviation <= 0.10:
                earned += 6
            elif deviation <= 0.25:
                earned += 4
            elif deviation <= 0.40:
                earned += 2

    scaled = (earned / possible) * MAX
    reason = (
        WellnessReasonCode.DIET_TRACKING_STRONG
        if scaled / MAX >= 0.75
        else WellnessReasonCode.DIET_TRACKING_NEEDS_ATTENTION
    )
    evidence: dict[str, bool | int | float | str] = {
        "consistencyDays": feeding.consistency_days,
        "foodTypeCount": len(feeding.food_types),
    }
    if feeding.avg_meals_per_day is not None:
        evidence["avgMealsPerDay"] = feeding.avg_meals_per_day
    if feeding.avg_calories_per_day is not None:
        evidence["avgCaloriesPerDay"] = feeding.avg_calories_per_day
    if calorie_weight is not None and calorie_target is not None and calorie_ratio is not None:
        evidence["weightKg"] = calorie_weight
        evidence["calorieTargetPerDay"] = round(calorie_target, 4)
        evidence["calorieRatio"] = round(calorie_ratio, 4)
    return _breakdown_item(
        score=round(scaled, 1),
        max_score=MAX,
        availability=WellnessDimensionAvailability.AVAILABLE,
        reason_codes=[reason],
        evidence=evidence,
    )
