"""
Wellness scoring service.

Architecture:
  1. Rule-based scoring across 6 dimensions → raw score (0-100)
  2. Condition cap applied if active chronic/serious conditions present
  3. Gemini generates narrative + recommendations from the final scores
  4. Trend computed by comparing to previousScore

Scoring dimensions and max points:
  Activity      20  (steps + active minutes vs species norms)
  Sleep         15  (sleep hours vs species norms)
  Diet          20  (meal consistency, food variety, calorie fit)
  Symptoms      25  (classifier output if currentSymptoms provided)
  Preventive    10  (vet visit + vaccinations + medication adherence)
  Baseline      10  (weight stability + age appropriateness)
  ─────────────────
  Total        100

Missing dimensions are marked included=false and omitted from score normalization.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import partial
from typing import Any, NamedTuple

from pydantic import BaseModel, Field, ValidationError

from app.ml.condition_metadata import get_condition_metadata
from app.ml.protocols import Classifier, GeneratorMetadata
from app.schemas import (
    ReminderType,
    TrendDirection,
    UrgencyLevel,
    WellnessActivity,
    WellnessBand,
    WellnessBreakdown,
    WellnessBreakdownItem,
    WellnessCondition,
    WellnessDimension,
    WellnessDimensionAvailability,
    WellnessFeeding,
    WellnessMedication,
    WellnessPet,
    WellnessPreventiveCare,
    WellnessReasonCode,
    WellnessReminder,
    WellnessRequest,
    WellnessResponse,
    WellnessScoreStatus,
    WellnessTrackingRecommendation,
    WellnessWeightMeasurement,
)
from app.services.gemini_rotation import RotatingGeminiClient
from app.services.generation_policy import (
    GenerationTimeoutError,
    bound_prompt,
    call_with_policy,
    log_fallback,
)

genai: Any
try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

logger = logging.getLogger(__name__)

WELLNESS_DISCLAIMER = (
    "This wellness indicator is based on tracked activity, feeding, and care data. "
    "It is not a clinical assessment and does not replace a veterinary examination."
)
CALCULATION_VERSION = "2.0.0"
MIN_SCORE_DATA_COVERAGE = 0.60
# Dimensions the reliability gate counts before a score is considered trustworthy.
_FOUNDATIONAL_DIMENSIONS = (
    WellnessDimension.ACTIVITY,
    WellnessDimension.SLEEP,
    WellnessDimension.DIET,
    WellnessDimension.PREVENTIVE_CARE,
    WellnessDimension.BASELINE,
)
INSUFFICIENT_DATA_NARRATIVE = (
    "There is not enough tracked data to calculate a wellness score yet. "
    "Record the suggested activity, feeding, preventive-care, or baseline details "
    "and request a new assessment."
)

# ── Species-specific norms ─────────────────────────────────────────────────

_ACTIVITY_TARGETS: dict[str, dict] = {
    "dog": {"steps": 8000, "active_min": 45},
    "cat": {"steps": 1500, "active_min": 20},
    "rabbit": {"steps": 0, "active_min": 30},
    "hamster": {"steps": 0, "active_min": 20},
    "guinea_pig": {"steps": 0, "active_min": 25},
    "bird": {"steps": 0, "active_min": 15},
    "fish": {"steps": 0, "active_min": 0},
    "turtle": {"steps": 0, "active_min": 10},
}
_DEFAULT_ACTIVITY = {"steps": 5000, "active_min": 30}

# (min_hours, max_hours) of healthy sleep per day; None = dimension not applicable
_SLEEP_NORMS: dict[str, tuple[float, float] | None] = {
    "dog": (12.0, 14.0),
    "cat": (13.0, 16.0),
    "rabbit": (8.0, 10.0),
    "hamster": (12.0, 14.0),
    "guinea_pig": (10.0, 12.0),
    "bird": (10.0, 12.0),
    "fish": None,
    "turtle": (12.0, 16.0),
}
_DEFAULT_SLEEP = (11.0, 14.0)

# Rough daily calorie target per kg of body weight (adult)
_KCAL_PER_KG: dict[str, float] = {
    "dog": 35.0,
    "cat": 45.0,
    "rabbit": 50.0,
    "hamster": 120.0,
    "guinea_pig": 60.0,
    "bird": 80.0,
    "fish": 0.0,
    "turtle": 20.0,
}
_DEFAULT_KCAL_PER_KG = 40.0

# Urgency → base symptom score (out of 25)
_URGENCY_BASE_SCORE: dict[UrgencyLevel, float] = {
    UrgencyLevel.EMERGENCY: 2.0,
    UrgencyLevel.URGENT: 9.0,
    UrgencyLevel.CONSULT_SOON: 15.0,
    UrgencyLevel.MONITOR: 21.0,
}

# Condition severity → maximum possible wellness score
_CONDITION_CAP_KEYWORDS: list[tuple[list[str], int]] = [
    # (keywords_to_match_in_name, cap)
    (
        [
            "cancer",
            "tumor",
            "tumour",
            "lymphoma",
            "leukemia",
            "carcinoma",
            "sarcoma",
            "heart failure",
            "congestive",
        ],
        65,
    ),
    (
        [
            "diabetes",
            "kidney",
            "renal",
            "liver",
            "hepatic",
            "epilepsy",
            "cushings",
            "addisons",
            "pancreatitis",
            "inflammatory bowel",
        ],
        75,
    ),
    (["arthritis", "allergy", "dermatitis", "thyroid", "asthma", "hip dysplasia", "luxating"], 85),
]

_BAND_LABELS: dict[WellnessBand, str] = {
    WellnessBand.EXCELLENT: "Excellent",
    WellnessBand.GOOD: "Good",
    WellnessBand.FAIR: "Fair",
    WellnessBand.CONCERNING: "Concerning",
    WellnessBand.CRITICAL: "Critical",
}


# ── Internal Gemini response model ─────────────────────────────────────────


class _WellnessNarrative(BaseModel):
    narrative: str = Field(
        ...,
        min_length=1,
        description="A short, mobile-friendly summary — 1-2 sentences, max ~35 words.",
    )
    recommendations: list[str] = Field(default_factory=list)


# Mobile cards have limited room — keep the narrative to ~2 sentences.
_NARRATIVE_MAX_SENTENCES = 2
_NARRATIVE_MAX_CHARS = 240


def _shorten_narrative(text: str) -> str:
    """Hard guard so the narrative stays mobile-friendly regardless of the model.

    Keeps the first couple of sentences and caps total length. Never fails — just
    trims. Recommendations carry the detail, so trimming the narrative is safe.
    """
    text = " ".join(text.split()).strip()
    # Keep the first N sentence-ending segments.
    parts, out = text.replace("! ", ". ").replace("? ", ". ").split(". "), []
    for count, part in enumerate(parts, start=1):
        out.append(part)
        if count >= _NARRATIVE_MAX_SENTENCES:
            break
    short = ". ".join(p.rstrip(".") for p in out).strip()
    if short and not short.endswith((".", "!", "?")):
        short += "."
    if len(short) > _NARRATIVE_MAX_CHARS:
        short = short[:_NARRATIVE_MAX_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return short


# ── Helper functions ────────────────────────────────────────────────────────


def _norm(species: str) -> str:
    return species.lower().strip()


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _breakdown_item(
    *,
    score: float,
    max_score: float,
    availability: WellnessDimensionAvailability,
    reason_codes: list[WellnessReasonCode],
    evidence: dict[str, bool | int | float | str] | None = None,
) -> WellnessBreakdownItem:
    return WellnessBreakdownItem(
        score=score,
        max_score=max_score,
        availability=availability,
        included=availability == WellnessDimensionAvailability.AVAILABLE,
        reason_codes=reason_codes,
        evidence=evidence or {},
    )


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
    if (
        calorie_weight is not None
        and calorie_target is not None
        and calorie_ratio is not None
    ):
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


def _score_preventive(
    care: WellnessPreventiveCare | None,
    adherence: float | None,
) -> WellnessBreakdownItem:
    MAX = 10.0
    earned = 0.0
    possible = 0.0

    if care is not None:
        possible += 8
        if care.recent_vet_visit:
            earned += 4
        if care.vaccinations_up_to_date:
            earned += 4

    if adherence is not None:
        # Adherence is always worth 20% of this dimension. If it is the only
        # available preventive input, keep the unobserved care portion at zero
        # instead of normalizing adherence from 2 points up to all 10.
        possible += 2 if care is not None else MAX
        earned += adherence * 2

    if possible == 0:
        return _breakdown_item(
            score=0,
            max_score=MAX,
            availability=WellnessDimensionAvailability.MISSING,
            reason_codes=[WellnessReasonCode.PREVENTIVE_CARE_DATA_MISSING],
        )

    scaled = (earned / possible) * MAX
    reason = (
        WellnessReasonCode.PREVENTIVE_CARE_CURRENT
        if scaled / MAX >= 0.9
        else WellnessReasonCode.PREVENTIVE_CARE_NEEDS_ATTENTION
    )
    evidence: dict[str, bool | int | float | str] = {}
    if care is not None:
        evidence["recentVetVisit"] = care.recent_vet_visit
        evidence["vaccinationsUpToDate"] = care.vaccinations_up_to_date
    if adherence is not None:
        evidence["medicationAdherence"] = round(adherence, 4)
    return _breakdown_item(
        score=round(scaled, 1),
        max_score=MAX,
        availability=WellnessDimensionAvailability.AVAILABLE,
        reason_codes=[reason],
        evidence=evidence,
    )


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
            any(item.included for item in behavior)
            or not any(item.applicable for item in behavior)
        )
        and included(
            WellnessDimension.PREVENTIVE_CARE,
            WellnessDimension.BASELINE,
        ) > 0
    )

    if not reliability_gate_met:
        return coverage, WellnessScoreStatus.INSUFFICIENT_DATA
    if any(
        item.availability == WellnessDimensionAvailability.MISSING for item in items
    ):
        return coverage, WellnessScoreStatus.PARTIAL
    return coverage, WellnessScoreStatus.COMPLETE


class _TrackingSpec(NamedTuple):
    """What a dimension needs tracked, and how to phrase it in either context."""

    required_inputs: tuple[str, ...]
    reminder_types: tuple[ReminderType, ...]
    gap_text: str
    # Absent when the dimension has no meaningful "keep it up" nudge.
    maintenance_text: str | None = None


_TRACKING_SPECS: dict[WellnessDimension, _TrackingSpec] = {
    WellnessDimension.ACTIVITY: _TrackingSpec(
        required_inputs=("activity.avgStepsPerDay", "activity.avgActiveMinutesPerDay"),
        reminder_types=(ReminderType.ACTIVITY,),
        gap_text="Track daily steps or active minutes to include activity in the wellness score.",
        maintenance_text="Keep tracking daily activity to maintain reliable wellness trends.",
    ),
    WellnessDimension.SLEEP: _TrackingSpec(
        required_inputs=("activity.avgSleepHoursPerDay",),
        reminder_types=(),
        gap_text="Track daily sleep duration to include sleep in the wellness score.",
    ),
    WellnessDimension.DIET: _TrackingSpec(
        required_inputs=("feeding.avgMealsPerDay", "feeding.consistencyDays"),
        reminder_types=(ReminderType.FEEDING,),
        gap_text="Track meal frequency and feeding consistency to include diet in the wellness score.",
        maintenance_text="Keep logging meals consistently to maintain reliable wellness trends.",
    ),
    WellnessDimension.PREVENTIVE_CARE: _TrackingSpec(
        required_inputs=(
            "preventiveCare.vaccinationsUpToDate",
            "preventiveCare.recentVetVisit",
        ),
        reminder_types=(ReminderType.VACCINATION, ReminderType.VET_VISIT),
        gap_text="Record vaccination and recent veterinary-visit status to include preventive care.",
        maintenance_text="Keep vaccination and veterinary-visit records current for reliable wellness trends.",
    ),
    WellnessDimension.BASELINE: _TrackingSpec(
        required_inputs=("pet.ageMonths", "weightHistory"),
        reminder_types=(),
        gap_text="Record age and at least two weight measurements to establish a wellness baseline.",
    ),
}


def _get_tracking_recommendations(
    breakdown: WellnessBreakdown,
) -> list[WellnessTrackingRecommendation]:
    """Tell the owner which untracked dimensions are holding the score back."""
    return [
        WellnessTrackingRecommendation(
            dimension=dimension,
            text=spec.gap_text,
            required_inputs=list(spec.required_inputs),
            suggested_reminder_types=list(spec.reminder_types),
        )
        for dimension, item in _dimension_items(breakdown)
        if item.availability == WellnessDimensionAvailability.MISSING
        and (spec := _TRACKING_SPECS.get(dimension)) is not None
    ]


def _get_maintenance_tracking_recommendations(
    breakdown: WellnessBreakdown,
    reminders: list[WellnessReminder],
) -> list[WellnessTrackingRecommendation]:
    """Nudge the owner to keep up tracking that no reminder already covers."""
    reminder_types_in_use = {item.reminder for item in reminders}
    recommendations: list[WellnessTrackingRecommendation] = []

    for dimension, item in _dimension_items(breakdown):
        spec = _TRACKING_SPECS.get(dimension)
        if not item.included or spec is None or spec.maintenance_text is None:
            continue
        available_suggestions = [
            reminder_type
            for reminder_type in spec.reminder_types
            if reminder_type not in reminder_types_in_use
        ]
        if not available_suggestions:
            continue
        recommendations.append(
            WellnessTrackingRecommendation(
                dimension=dimension,
                text=spec.maintenance_text,
                required_inputs=list(spec.required_inputs),
                suggested_reminder_types=available_suggestions,
            )
        )

    return recommendations


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


def _is_below(item: WellnessBreakdownItem, threshold: float) -> bool:
    return item.included and item.max_score > 0 and item.score / item.max_score < threshold


def _get_safety_reminders(
    detected_urgency: UrgencyLevel | None,
) -> list[WellnessReminder]:
    if detected_urgency not in {
        UrgencyLevel.CONSULT_SOON,
        UrgencyLevel.URGENT,
        UrgencyLevel.EMERGENCY,
    }:
        return []

    return [
        WellnessReminder(
            reminder=ReminderType.VET_VISIT,
            text="Consult a veterinarian about whether clinical treatment is needed.",
        )
    ]


def _get_reminders(
    request: WellnessRequest,
    breakdown: WellnessBreakdown,
    detected_urgency: UrgencyLevel | None,
    adherence: float | None,
) -> list[WellnessReminder]:
    # Insertion-ordered; setdefault keeps the first text chosen for each type.
    reminders: dict[ReminderType, WellnessReminder] = {}

    def add(reminder: ReminderType, text: str) -> None:
        reminders.setdefault(reminder, WellnessReminder(reminder=reminder, text=text))

    if _is_below(breakdown.diet, 0.75):
        add(
            ReminderType.FEEDING,
            f"Review the {request.pet.species}'s feeding schedule and keep meal times consistent.",
        )
    if _is_below(breakdown.activity, 0.75):
        add(
            ReminderType.ACTIVITY,
            f"Schedule regular daily activity for your {request.pet.species}.",
        )

    if adherence is not None and adherence < 0.9:
        add(
            ReminderType.MEDICATION,
            "Check the existing medication schedule recorded for your pet.",
        )

    for safety_reminder in _get_safety_reminders(detected_urgency):
        add(safety_reminder.reminder, safety_reminder.text)

    if request.preventive_care is not None:
        if not request.preventive_care.vaccinations_up_to_date:
            add(
                ReminderType.VACCINATION,
                "Schedule an appointment to bring vaccinations up to date.",
            )
        if not request.preventive_care.recent_vet_visit:
            add(ReminderType.VET_VISIT, "Schedule a veterinary check-up.")

    return list(reminders.values())


_REMINDER_RECOMMENDATION_TERMS: dict[ReminderType, frozenset[str]] = {
    ReminderType.FEEDING: frozenset(
        {"feed", "meal", "diet", "food", "nutrition", "calorie", "portion"}
    ),
    ReminderType.ACTIVITY: frozenset(
        {"activity", "active time", "exercise", "walk", "strenuous"}
    ),
    ReminderType.MEDICATION: frozenset(
        {
            "medication",
            "medicine",
            "dose",
            "dosing",
            "administer",
            "prescription",
            "drug",
        }
    ),
    ReminderType.VACCINATION: frozenset({"vaccin", "immuniz"}),
    ReminderType.PARASITE_TREATMENT: frozenset(
        {"parasite", "flea", "tick", "heartworm", "deworm"}
    ),
    ReminderType.VET_VISIT: frozenset(
        {
            "veterinar",
            "vet ",
            "vet.",
            "appointment",
            "check-up",
            "checkup",
            "clinic",
        }
    ),
    ReminderType.GROOMING: frozenset({"groom", "brush", "coat"}),
}

# Gemini must never give medication, parasite-product, or supplement advice.
_MEDICATION_ADVICE_TERMS = (
    _REMINDER_RECOMMENDATION_TERMS[ReminderType.MEDICATION]
    | _REMINDER_RECOMMENDATION_TERMS[ReminderType.PARASITE_TREATMENT]
    | {"supplement"}
)

_UNOBSERVED_SYMPTOM_TERMS = frozenset({"symptom", "monitor closely"})


def _filter_recommendations(
    recommendations: list[str],
    reminders: list[WellnessReminder],
    breakdown: WellnessBreakdown,
) -> list[str]:
    blocked_terms = set(_MEDICATION_ADVICE_TERMS)

    # Structured reminders already cover these topics; don't repeat them in prose.
    for reminder in reminders:
        blocked_terms |= _REMINDER_RECOMMENDATION_TERMS[reminder.reminder]

    # Missing data must not lead to advice for an unobserved dimension.
    for item, reminder_types in (
        (breakdown.diet, (ReminderType.FEEDING,)),
        (breakdown.activity, (ReminderType.ACTIVITY,)),
        (
            breakdown.preventive_care,
            (ReminderType.VACCINATION, ReminderType.VET_VISIT),
        ),
    ):
        if not item.included:
            for reminder_type in reminder_types:
                blocked_terms |= _REMINDER_RECOMMENDATION_TERMS[reminder_type]
    if not breakdown.symptoms.included:
        blocked_terms |= _UNOBSERVED_SYMPTOM_TERMS

    filtered: list[str] = []
    seen: set[str] = set()
    for recommendation in recommendations:
        normalized = recommendation.strip()
        lowered = normalized.lower()
        if not normalized or any(term in lowered for term in blocked_terms):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        filtered.append(normalized)
    return filtered


def _add_weight_recommendation(
    recommendations: list[str],
    stability: float | None,
) -> list[str]:
    if stability is None or stability >= 0.8:
        return recommendations

    recommendation = (
        "Continue recording weight regularly and monitor the trend for further changes."
    )
    if recommendation in recommendations:
        return recommendations
    return [*recommendations, recommendation]


# ── Gemini narrative generation ─────────────────────────────────────────────

_NARRATIVE_SYSTEM = """
You are a veterinary wellness assistant generating a report for a pet owner.
Given the pet details and wellness score breakdown, write:
1. A SHORT narrative for a mobile app card: 1-2 sentences, max ~35 words total.
   Name the weakest SCORED dimension in a few words. Encouraging, non-alarmist. No preamble.
2. Up to 3 specific, actionable recommendations ordered by priority.
   Be concrete — e.g. "Add 10 minutes to morning walks" not just "exercise more".
   Put the detail HERE, not in the narrative.
IMPORTANT: dimensions marked "not tracked (no data)" are MISSING data, NOT low scores.
Never tell the owner to improve a not-tracked dimension. If a not-tracked dimension is
important, you may gently suggest they START TRACKING it — but prioritise dimensions
that were actually scored. Base the narrative's "weakest area" only on scored dimensions.
Do not repeat categories handled by structured reminders.
Do not provide medication, dosing, treatment, parasite-product, or supplement advice.
Never suggest starting, stopping, changing, or administering a medicine.
Return ONLY valid JSON matching the required schema.
"""


def _build_narrative_prompt(
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
) -> str:
    def format_item(item: WellnessBreakdownItem) -> str:
        if not item.included:
            return "not available"
        return f"{item.score}/{item.max_score}"

    lines = [
        f"Pet: {request.pet.species}, {request.pet.breed or 'unknown breed'}, "
        f"age {request.pet.age_months or '?'} months, weight {request.pet.weight_kg or '?'} kg",
        f"Wellness score: {score}/100 ({band.value})",
        f"Assessment status: {score_status.value}",
        f"Weighted data coverage: {data_coverage:.2%}",
        "Breakdown:",
        f"  Activity:      {format_item(breakdown.activity)}",
        f"  Sleep:         {format_item(breakdown.sleep)}",
        f"  Diet:          {format_item(breakdown.diet)}",
        f"  Symptoms:      {format_item(breakdown.symptoms)}",
        f"  Preventive:    {format_item(breakdown.preventive_care)}",
        f"  Baseline:      {format_item(breakdown.baseline)}",
    ]
    missing_dimensions = [
        dimension.value
        for dimension, item in _dimension_items(breakdown)
        if item.availability == WellnessDimensionAvailability.MISSING
    ]
    if missing_dimensions:
        lines.append(f"Missing dimensions: {', '.join(missing_dimensions)}")
        lines.append(
            "This is a partial assessment; do not describe it as complete."
        )
    if detected_condition:
        lines.append(f"Classifier detected: {detected_condition}")
    if condition_cap is not None:
        lines.append(f"Score capped at {condition_cap} due to active chronic condition.")
    if request.active_conditions:
        names = ", ".join(c.name for c in request.active_conditions)
        lines.append(f"Active conditions: {names}")
    if stability is not None:
        lines.append(f"Weight stability score: {stability:.0%}")
    if reminders:
        handled = ", ".join(reminder.reminder.value for reminder in reminders)
        lines.append(
            f"Handled by structured reminders (do not repeat in recommendations): {handled}"
        )
    return "\n".join(lines)


_FALLBACK_NARRATIVES: dict[WellnessBand, str] = {
    WellnessBand.EXCELLENT: "Your pet is in excellent shape based on this week's tracked data. Keep up the great routine!",
    WellnessBand.GOOD:      "Your pet is doing well overall. There are a few small areas worth improving.",
    WellnessBand.FAIR:      "Your pet's wellness is fair. Some dimensions need attention — check the breakdown above.",
    WellnessBand.CONCERNING: "Your pet's wellness is concerning this week. Consider reviewing diet, activity, and scheduling a vet check.",
    WellnessBand.CRITICAL:  "Your pet's tracked data indicates a critical wellness level. Please consult a veterinarian promptly.",
}

_FALLBACK_RECOMMENDATIONS: dict[WellnessBand, tuple[str, ...]] = {
    WellnessBand.EXCELLENT:  ("Maintain the current routine.", "Schedule a routine annual vet check."),
    WellnessBand.GOOD:       ("Review the dimension with the lowest sub-score.", "Ensure consistent meal timing."),
    WellnessBand.FAIR:       ("Increase daily active time.", "Log feeding more consistently.", "Book a vet appointment if symptoms persist."),
    WellnessBand.CONCERNING: ("Schedule a veterinary check-up soon.", "Improve feeding consistency.", "Increase monitored exercise."),
    WellnessBand.CRITICAL:   ("Contact a veterinarian as soon as possible.", "Monitor symptoms closely.", "Avoid strenuous activity until assessed."),
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


# ── Main service class ──────────────────────────────────────────────────────


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
                tracking_recommendations=_get_tracking_recommendations(breakdown),
            )

        # 2. Raw score (0-100), scaled for missing dimensions
        raw_score = _compute_score(breakdown)

        # 3. Apply condition cap
        final_score = int(min(raw_score, cap) if cap is not None else raw_score)
        final_score = max(0, min(100, final_score))

        band = _get_band(final_score)
        reminders = _get_reminders(request, breakdown, detected_urgency, adherence)
        if (
            score_status == WellnessScoreStatus.COMPLETE
            and band in {WellnessBand.GOOD, WellnessBand.EXCELLENT}
        ):
            tracking_recommendations = _get_maintenance_tracking_recommendations(
                breakdown,
                reminders,
            )
        else:
            tracking_recommendations = _get_tracking_recommendations(breakdown)

        # 4. Gemini narrative + recommendations
        narrative, recommendations = self._generate_narrative(
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

    def _generate_narrative(
        self,
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
        fallback = partial(_fallback_narrative, band, score_status, data_coverage)
        if self.client is None:
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
                lambda: self.client.models.generate_content(
                    model=self.model_name,
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
