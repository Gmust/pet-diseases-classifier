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
  Preventive    10  (vet visit + vaccinations)
  Baseline      10  (weight + age appropriateness)
  ─────────────────
  Total        100

Missing dimensions: raw sum is scaled so missing data does not punish.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from app.ml.condition_metadata import get_condition_metadata
from app.schemas import (
    TrendDirection,
    UrgencyLevel,
    WellnessActivity,
    WellnessBand,
    WellnessBreakdown,
    WellnessBreakdownItem,
    WellnessCondition,
    WellnessFeeding,
    WellnessMedication,
    WellnessPet,
    WellnessPreventiveCare,
    WellnessRequest,
    WellnessResponse,
)

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

logger = logging.getLogger(__name__)

WELLNESS_DISCLAIMER = (
    "This wellness indicator is based on tracked activity, feeding, and care data. "
    "It is not a clinical assessment and does not replace a veterinary examination."
)

# ── Species-specific norms ─────────────────────────────────────────────────

_ACTIVITY_TARGETS: dict[str, dict] = {
    "dog":        {"steps": 8000, "active_min": 45},
    "cat":        {"steps": 1500, "active_min": 20},
    "rabbit":     {"steps": 0,    "active_min": 30},
    "hamster":    {"steps": 0,    "active_min": 20},
    "guinea_pig": {"steps": 0,    "active_min": 25},
    "bird":       {"steps": 0,    "active_min": 15},
    "fish":       {"steps": 0,    "active_min": 0},
    "turtle":     {"steps": 0,    "active_min": 10},
}
_DEFAULT_ACTIVITY = {"steps": 5000, "active_min": 30}

# (min_hours, max_hours) of healthy sleep per day
_SLEEP_NORMS: dict[str, tuple[float, float]] = {
    "dog":        (12.0, 14.0),
    "cat":        (13.0, 16.0),
    "rabbit":     (8.0,  10.0),
    "hamster":    (12.0, 14.0),
    "guinea_pig": (10.0, 12.0),
    "bird":       (10.0, 12.0),
    "fish":       (0.0,  24.0),  # not applicable — full score always
    "turtle":     (12.0, 16.0),
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
    UrgencyLevel.EMERGENCY:    2.0,
    UrgencyLevel.URGENT:       9.0,
    UrgencyLevel.CONSULT_SOON: 15.0,
    UrgencyLevel.MONITOR:      21.0,
}

# Condition severity → maximum possible wellness score
_CONDITION_CAP_KEYWORDS: list[tuple[list[str], int]] = [
    # (keywords_to_match_in_name, cap)
    (["cancer", "tumor", "tumour", "lymphoma", "leukemia", "carcinoma",
      "sarcoma", "heart failure", "congestive"], 65),
    (["diabetes", "kidney", "renal", "liver", "hepatic", "epilepsy",
      "cushings", "addisons", "pancreatitis", "inflammatory bowel"], 75),
    (["arthritis", "allergy", "dermatitis", "thyroid", "asthma",
      "hip dysplasia", "luxating"], 85),
]

_BAND_LABELS: dict[WellnessBand, str] = {
    WellnessBand.EXCELLENT:  "Excellent",
    WellnessBand.GOOD:       "Good",
    WellnessBand.FAIR:       "Fair",
    WellnessBand.CONCERNING: "Concerning",
    WellnessBand.CRITICAL:   "Critical",
}


# ── Internal Gemini response model ─────────────────────────────────────────

class _WellnessNarrative(BaseModel):
    narrative: str = Field(..., min_length=1)
    recommendations: list[str] = Field(default_factory=list)


# ── Helper functions ────────────────────────────────────────────────────────

def _norm(species: str) -> str:
    return species.lower().strip()


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _score_activity(
    activity: WellnessActivity | None,
    species: str,
) -> WellnessBreakdownItem:
    MAX = 20.0
    if activity is None:
        return WellnessBreakdownItem(score=0, max_score=MAX)

    target = _ACTIVITY_TARGETS.get(_norm(species), _DEFAULT_ACTIVITY)
    earned = 0.0
    possible = 0.0

    # Steps sub-score (10 pts) — only if target > 0
    if target["steps"] > 0:
        possible += 10
        if activity.avg_steps_per_day is not None:
            ratio = _clamp(activity.avg_steps_per_day / target["steps"])
            earned += ratio * 10

    # Active minutes sub-score (10 pts)
    if target["active_min"] > 0:
        possible += 10
        if activity.avg_active_minutes_per_day is not None:
            ratio = _clamp(activity.avg_active_minutes_per_day / target["active_min"])
            earned += ratio * 10
    else:
        # Species like fish — full points automatically
        possible += 10
        earned += 10

    if possible == 0:
        return WellnessBreakdownItem(score=MAX, max_score=MAX)

    # Scale earned to MAX
    scaled = (earned / possible) * MAX
    return WellnessBreakdownItem(score=round(scaled, 1), max_score=MAX)


def _score_sleep(
    activity: WellnessActivity | None,
    species: str,
) -> WellnessBreakdownItem:
    MAX = 15.0
    if activity is None or activity.avg_sleep_hours_per_day is None:
        return WellnessBreakdownItem(score=0, max_score=MAX)

    lo, hi = _SLEEP_NORMS.get(_norm(species), _DEFAULT_SLEEP)
    hours = activity.avg_sleep_hours_per_day
    mid = (lo + hi) / 2
    spread = (hi - lo) / 2 or 1.0  # avoid div-by-zero for fish

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

    return WellnessBreakdownItem(score=round(score, 1), max_score=MAX)


def _score_diet(
    feeding: WellnessFeeding | None,
    pet: WellnessPet,
) -> WellnessBreakdownItem:
    MAX = 20.0
    if feeding is None:
        return WellnessBreakdownItem(score=0, max_score=MAX)

    earned = 0.0
    possible = 0.0

    # Consistency (6 pts): how many of last 7 days had feeding logs
    possible += 6
    if feeding.consistency_days > 0:
        consistency_ratio = _clamp(feeding.consistency_days / 7)
        earned += consistency_ratio * 6

    # Meal frequency (6 pts): 2-3 meals/day is ideal for most species
    possible += 6
    if feeding.avg_meals_per_day is not None:
        mpd = feeding.avg_meals_per_day
        if 1.8 <= mpd <= 3.2:
            earned += 6
        elif 1.0 <= mpd < 1.8 or 3.2 < mpd <= 4.0:
            earned += 4
        else:
            earned += 2

    # Variety (2 pts): more than one food type
    possible += 2
    if len(feeding.food_types) >= 2:
        earned += 2
    elif len(feeding.food_types) == 1:
        earned += 1

    # Calorie fit (6 pts): only if weight and calories are known
    if pet.weight_kg and feeding.avg_calories_per_day:
        possible += 6
        kcal_per_kg = _KCAL_PER_KG.get(_norm(pet.species), _DEFAULT_KCAL_PER_KG)
        target_kcal = kcal_per_kg * pet.weight_kg
        if target_kcal > 0:
            ratio = feeding.avg_calories_per_day / target_kcal
            # ratio=1.0 is perfect; penalise deviation
            deviation = abs(ratio - 1.0)
            if deviation <= 0.10:
                earned += 6
            elif deviation <= 0.25:
                earned += 4
            elif deviation <= 0.40:
                earned += 2
            else:
                earned += 0

    if possible == 0:
        return WellnessBreakdownItem(score=MAX, max_score=MAX)

    scaled = (earned / possible) * MAX
    return WellnessBreakdownItem(score=round(scaled, 1), max_score=MAX)


def _score_symptoms(
    symptoms_text: str | None,
    predictor,
) -> tuple[WellnessBreakdownItem, str | None]:
    """Returns (breakdown_item, detected_condition_name | None)."""
    MAX = 25.0
    if not symptoms_text or predictor is None:
        # Neutral: benefit of the doubt, no symptoms described
        return WellnessBreakdownItem(score=20, max_score=MAX), None

    try:
        prediction = predictor.predict(symptoms_text)
    except Exception as exc:
        logger.warning("Wellness symptom classifier failed: %s", exc)
        return WellnessBreakdownItem(score=20, max_score=MAX), None

    meta = get_condition_metadata(prediction.predicted_condition)
    base = _URGENCY_BASE_SCORE[meta.urgency]

    # High confidence of a mild condition → slight bonus; bad condition → stays low
    if meta.urgency == UrgencyLevel.MONITOR:
        score = base + prediction.confidence * 4        # up to 25
    elif meta.urgency == UrgencyLevel.EMERGENCY:
        score = base + (1 - prediction.confidence) * 3  # stays near 0-5
    else:
        score = base + (1 - prediction.confidence) * 3  # slight leniency for uncertainty

    return (
        WellnessBreakdownItem(score=round(_clamp(score, 0, MAX), 1), max_score=MAX),
        prediction.predicted_condition,
    )


def _score_preventive(
    care: WellnessPreventiveCare | None,
) -> WellnessBreakdownItem:
    MAX = 10.0
    if care is None:
        return WellnessBreakdownItem(score=0, max_score=MAX)
    score = 0.0
    if care.recent_vet_visit:
        score += 5
    if care.vaccinations_up_to_date:
        score += 5
    return WellnessBreakdownItem(score=score, max_score=MAX)


def _score_baseline(pet: WellnessPet) -> WellnessBreakdownItem:
    MAX = 10.0
    score = 0.0

    # Age factor (5 pts): puppies/kittens and adults full points; seniors slight leniency
    if pet.age_months is not None:
        if pet.age_months <= 24 or pet.age_months <= 120:  # up to ~10 years
            score += 5
        else:
            score += 4  # senior — still good, just leniency

    # Weight provided (5 pts): we give points for tracking, not for exact number
    # (we don't have breed-specific weight charts here)
    if pet.weight_kg is not None and pet.weight_kg > 0:
        score += 5

    return WellnessBreakdownItem(score=score, max_score=MAX)


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
    items = [
        breakdown.activity,
        breakdown.sleep,
        breakdown.diet,
        breakdown.symptoms,
        breakdown.preventive_care,
        breakdown.baseline,
    ]
    total_earned = sum(i.score for i in items)
    total_max = sum(i.max_score for i in items)
    if total_max == 0:
        return 0.0
    return (total_earned / total_max) * 100


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


# ── Gemini narrative generation ─────────────────────────────────────────────

_NARRATIVE_SYSTEM = """
You are a veterinary wellness assistant generating a report for a pet owner.
Given the pet details and wellness score breakdown, write:
1. A clear narrative (3-4 sentences) summarising the pet's wellness this week.
   Mention which dimension is the weakest and why it matters.
   Use encouraging, non-alarmist language.
2. 3-5 specific, actionable recommendations ordered by priority.
   Be concrete — e.g. "Add 10 minutes to morning walks" not just "exercise more".
Return ONLY valid JSON matching the required schema.
"""


def _build_narrative_prompt(
    request: WellnessRequest,
    breakdown: WellnessBreakdown,
    score: int,
    band: WellnessBand,
    condition_cap: int | None,
    detected_condition: str | None,
) -> str:
    lines = [
        f"Pet: {request.pet.species}, {request.pet.breed or 'unknown breed'}, "
        f"age {request.pet.age_months or '?'} months, weight {request.pet.weight_kg or '?'} kg",
        f"Wellness score: {score}/100 ({band.value})",
        f"Breakdown:",
        f"  Activity:      {breakdown.activity.score}/{breakdown.activity.max_score}",
        f"  Sleep:         {breakdown.sleep.score}/{breakdown.sleep.max_score}",
        f"  Diet:          {breakdown.diet.score}/{breakdown.diet.max_score}",
        f"  Symptoms:      {breakdown.symptoms.score}/{breakdown.symptoms.max_score}",
        f"  Preventive:    {breakdown.preventive_care.score}/{breakdown.preventive_care.max_score}",
        f"  Baseline:      {breakdown.baseline.score}/{breakdown.baseline.max_score}",
    ]
    if detected_condition:
        lines.append(f"Classifier detected: {detected_condition}")
    if condition_cap is not None:
        lines.append(f"Score capped at {condition_cap} due to active chronic condition.")
    if request.active_conditions:
        names = ", ".join(c.name for c in request.active_conditions)
        lines.append(f"Active conditions: {names}")
    if request.active_medications:
        meds = ", ".join(m.name for m in request.active_medications)
        lines.append(f"Current medications: {meds}")
    return "\n".join(lines)


def _fallback_narrative(band: WellnessBand, score: int) -> tuple[str, list[str]]:
    narratives = {
        WellnessBand.EXCELLENT: "Your pet is in excellent shape based on this week's tracked data. Keep up the great routine!",
        WellnessBand.GOOD:      "Your pet is doing well overall. There are a few small areas worth improving.",
        WellnessBand.FAIR:      "Your pet's wellness is fair. Some dimensions need attention — check the breakdown above.",
        WellnessBand.CONCERNING: "Your pet's wellness is concerning this week. Consider reviewing diet, activity, and scheduling a vet check.",
        WellnessBand.CRITICAL:  "Your pet's tracked data indicates a critical wellness level. Please consult a veterinarian promptly.",
    }
    recs = {
        WellnessBand.EXCELLENT:  ["Maintain the current routine.", "Schedule a routine annual vet check."],
        WellnessBand.GOOD:       ["Review the dimension with the lowest sub-score.", "Ensure consistent meal timing."],
        WellnessBand.FAIR:       ["Increase daily active time.", "Log feeding more consistently.", "Book a vet appointment if symptoms persist."],
        WellnessBand.CONCERNING: ["Schedule a veterinary check-up soon.", "Improve feeding consistency.", "Increase monitored exercise."],
        WellnessBand.CRITICAL:   ["Contact a veterinarian as soon as possible.", "Monitor symptoms closely.", "Avoid strenuous activity until assessed."],
    }
    return narratives[band], recs[band]


# ── Main service class ──────────────────────────────────────────────────────

class WellnessService:
    def __init__(self, api_key: Optional[str], model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name
        self.client = None
        if api_key and genai is not None:
            self.client = genai.Client(api_key=api_key)
        else:
            logger.warning("Gemini not available — /wellness will use fallback narratives.")

    def score(self, request: WellnessRequest, predictor=None) -> WellnessResponse:
        # 1. Score each dimension
        activity_item = _score_activity(request.activity, request.pet.species)
        sleep_item    = _score_sleep(request.activity, request.pet.species)
        diet_item     = _score_diet(request.feeding, request.pet)
        symptoms_item, detected_condition = _score_symptoms(request.current_symptoms, predictor)
        preventive_item = _score_preventive(request.preventive_care)
        baseline_item   = _score_baseline(request.pet)

        breakdown = WellnessBreakdown(
            activity=activity_item,
            sleep=sleep_item,
            diet=diet_item,
            symptoms=symptoms_item,
            preventive_care=preventive_item,
            baseline=baseline_item,
        )

        # 2. Raw score (0-100), scaled for missing dimensions
        raw_score = _compute_score(breakdown)

        # 3. Apply condition cap
        cap = _condition_cap(request.active_conditions)
        final_score = int(min(raw_score, cap) if cap is not None else raw_score)
        final_score = max(0, min(100, final_score))

        band = _get_band(final_score)
        trend = _get_trend(final_score, request.previous_score)

        # 4. Gemini narrative + recommendations
        narrative, recommendations = self._generate_narrative(
            request=request,
            breakdown=breakdown,
            score=final_score,
            band=band,
            condition_cap=cap,
            detected_condition=detected_condition,
        )

        return WellnessResponse(
            wellness_score=final_score,
            band=band,
            band_label=_BAND_LABELS[band],
            trend=trend,
            breakdown=breakdown,
            condition_cap=cap,
            classifier_condition=detected_condition,
            narrative=narrative,
            recommendations=recommendations,
            disclaimer=WELLNESS_DISCLAIMER,
        )

    def _generate_narrative(
        self,
        request: WellnessRequest,
        breakdown: WellnessBreakdown,
        score: int,
        band: WellnessBand,
        condition_cap: int | None,
        detected_condition: str | None,
    ) -> tuple[str, list[str]]:
        if self.client is None:
            return _fallback_narrative(band, score)

        prompt = _build_narrative_prompt(request, breakdown, score, band, condition_cap, detected_condition)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.35,
                    "system_instruction": _NARRATIVE_SYSTEM,
                    "response_mime_type": "application/json",
                    "response_json_schema": _WellnessNarrative.model_json_schema(),
                },
            )
            if not response.text:
                raise ValueError("Empty Gemini response.")
            parsed = _WellnessNarrative.model_validate_json(response.text)
            return parsed.narrative, parsed.recommendations
        except (ValidationError, ValueError) as exc:
            logger.warning("Wellness narrative parsing failed: %s", exc)
            return _fallback_narrative(band, score)
        except Exception as exc:  # pragma: no cover
            logger.warning("Wellness Gemini request failed: %s", exc)
            return _fallback_narrative(band, score)
