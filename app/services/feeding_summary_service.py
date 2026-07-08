"""
Deterministic daily feeding-summary service.

No Gemini call — this endpoint is meant to run once per day for every pet, so
prompt cost/latency would scale with the pet count. Everything here is a pure
function of (species, weight, logged products), matching the "no LLM where a
formula suffices" pattern used by the wellness score's rule-based dimensions
(see `_score_diet` in wellness_service.py).

Target daily calories use the standard veterinary-nutrition RER/MER formula:
    RER (kcal/day) = 70 * weight_kg^0.75
    MER (kcal/day) = RER * factor
`_ACTIVITY_FACTORS` holds a conservative "average adult" factor per species.
For juveniles (ageMonths < `_JUVENILE_MONTHS_THRESHOLD`) `_JUVENILE_FACTOR` is
used instead — a growing kitten/puppy needs far more kcal/kg than an adult at
the same weight, so a low weight alone (e.g. a 1kg puppy) is not implausible
and must not be treated as a data error. Good enough for a daily nudge
notification, not a substitute for a vet-prescribed diet plan.
"""

from __future__ import annotations

from app.schemas import (
    FeedingSummaryPet,
    FeedingSummaryRequest,
    FeedingSummaryResponse,
    FeedingSummaryResult,
    FeedingSummaryStatus,
    PetType,
)

FEEDING_SUMMARY_DISCLAIMER = (
    "This feeding summary is an estimate based on logged food and standard calorie "
    "formulas. It is not a substitute for a vet-prescribed diet plan."
)

# Average-adult maintenance-energy multiplier over RER, by species.
_ACTIVITY_FACTORS: dict[PetType, float] = {
    PetType.DOG: 1.6,
    PetType.CAT: 1.2,
    PetType.RABBIT: 1.4,
    PetType.HAMSTER: 1.4,
    PetType.GUINEA_PIG: 1.4,
    PetType.BIRD: 1.4,
    PetType.FISH: 1.0,
    PetType.TURTLE: 1.0,
    PetType.OTHER: 1.4,
}

_JUVENILE_MONTHS_THRESHOLD = 12
_JUVENILE_FACTOR = 2.5  # growth energy requirement, applied regardless of species

_ON_TARGET_BAND = 10.0  # ±10% of target counts as "on target"
_EXTREME_OVER_BAND = 100.0  # > +100% of target counts as "extreme" overfeeding
_EXTREME_UNDER_BAND = -50.0  # < -50% of target counts as "extreme" underfeeding


def _activity_factor(pet: FeedingSummaryPet) -> float:
    if pet.age_months is not None and pet.age_months < _JUVENILE_MONTHS_THRESHOLD:
        return _JUVENILE_FACTOR
    return _ACTIVITY_FACTORS.get(pet.species, 1.4)


def _target_calories(pet: FeedingSummaryPet) -> float:
    rer = 70.0 * (pet.weight_kg**0.75)
    return rer * _activity_factor(pet)


def _status_and_deviation(actual: float, target: float) -> tuple[FeedingSummaryStatus, float]:
    if target <= 0:
        return FeedingSummaryStatus.ON_TARGET, 0.0
    deviation_pct = ((actual - target) / target) * 100.0
    if deviation_pct > _EXTREME_OVER_BAND:
        return FeedingSummaryStatus.EXTREME_OVER_TARGET, deviation_pct
    if deviation_pct > _ON_TARGET_BAND:
        return FeedingSummaryStatus.OVER_TARGET, deviation_pct
    if deviation_pct < _EXTREME_UNDER_BAND:
        return FeedingSummaryStatus.EXTREME_UNDER_TARGET, deviation_pct
    if deviation_pct < -_ON_TARGET_BAND:
        return FeedingSummaryStatus.UNDER_TARGET, deviation_pct
    return FeedingSummaryStatus.ON_TARGET, deviation_pct


def summarize_pet(pet: FeedingSummaryPet) -> FeedingSummaryResult:
    target = _target_calories(pet)
    actual = sum(product.calories for product in pet.products)
    status, deviation_pct = _status_and_deviation(actual, target)
    return FeedingSummaryResult(
        pet_id=pet.pet_id,
        status=status,
        target_calories=round(target, 1),
        actual_calories=round(actual, 1),
        deviation_pct=round(deviation_pct, 1),
    )


def summarize(request: FeedingSummaryRequest) -> FeedingSummaryResponse:
    results = [summarize_pet(pet) for pet in request.pets]
    return FeedingSummaryResponse(results=results, disclaimer=FEEDING_SUMMARY_DISCLAIMER)
