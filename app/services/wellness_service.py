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
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.ml.condition_metadata import get_condition_metadata
from app.ml.protocols import Classifier, GeneratorMetadata
from app.schemas import (
    TrendDirection,
    UrgencyLevel,
    WellnessActivity,
    WellnessBand,
    WellnessBreakdown,
    WellnessBreakdownItem,
    WellnessCondition,
    WellnessFeeding,
    WellnessPet,
    WellnessPreventiveCare,
    WellnessRequest,
    WellnessResponse,
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

# (min_hours, max_hours) of healthy sleep per day
_SLEEP_NORMS: dict[str, tuple[float, float]] = {
    "dog": (12.0, 14.0),
    "cat": (13.0, 16.0),
    "rabbit": (8.0, 10.0),
    "hamster": (12.0, 14.0),
    "guinea_pig": (10.0, 12.0),
    "bird": (10.0, 12.0),
    "fish": (0.0, 24.0),  # not applicable — full score always
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


def _score_activity(
    activity: WellnessActivity | None,
    species: str,
) -> WellnessBreakdownItem:
    """Score activity. Absent data is EXCLUDED (max_score=0) so it doesn't punish."""
    MAX = 20.0
    if activity is None:
        return WellnessBreakdownItem(score=0, max_score=0)  # excluded from the total

    target = _ACTIVITY_TARGETS.get(_norm(species), _DEFAULT_ACTIVITY)
    earned = 0.0
    possible = 0.0

    # Steps sub-score (10 pts) — only counts toward `possible` when provided.
    if target["steps"] > 0 and activity.avg_steps_per_day is not None:
        possible += 10
        earned += _clamp(activity.avg_steps_per_day / target["steps"]) * 10

    # Active minutes sub-score (10 pts)
    if target["active_min"] > 0:
        if activity.avg_active_minutes_per_day is not None:
            possible += 10
            earned += _clamp(activity.avg_active_minutes_per_day / target["active_min"]) * 10
    else:
        # Species like fish — active minutes not applicable; full points automatically.
        possible += 10
        earned += 10

    if possible == 0:
        return WellnessBreakdownItem(score=0, max_score=0)  # no usable activity data → excluded

    return WellnessBreakdownItem(score=round((earned / possible) * MAX, 1), max_score=MAX)


def _score_sleep(
    activity: WellnessActivity | None,
    species: str,
) -> WellnessBreakdownItem:
    MAX = 15.0
    if activity is None or activity.avg_sleep_hours_per_day is None:
        return WellnessBreakdownItem(score=0, max_score=0)  # no sleep data → excluded

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
        return WellnessBreakdownItem(score=0, max_score=0)  # no feeding data → excluded

    earned = 0.0
    possible = 0.0

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
        target_kcal = kcal_per_kg * pet.weight_kg
        if target_kcal > 0:
            deviation = abs(feeding.avg_calories_per_day / target_kcal - 1.0)
            if deviation <= 0.10:
                earned += 6
            elif deviation <= 0.25:
                earned += 4
            elif deviation <= 0.40:
                earned += 2

    if possible == 0:
        return WellnessBreakdownItem(score=0, max_score=0)  # no usable diet data → excluded

    return WellnessBreakdownItem(score=round((earned / possible) * MAX, 1), max_score=MAX)


def _score_symptoms(
    symptoms_text: str | None,
    predictor: Classifier | None,
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
        score = base + prediction.confidence * 4  # up to 25
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
        return WellnessBreakdownItem(score=0, max_score=0)  # no preventive-care data → excluded
    score = 0.0
    if care.recent_vet_visit:
        score += 5
    if care.vaccinations_up_to_date:
        score += 5
    return WellnessBreakdownItem(score=score, max_score=MAX)


def _score_baseline(pet: WellnessPet) -> WellnessBreakdownItem:
    """Each provided field contributes to BOTH earned and possible, so absent
    fields are scaled out rather than scored as zero."""
    earned = 0.0
    possible = 0.0

    # Age factor (5 pts): adults full points; seniors (>~10y) slight leniency.
    if pet.age_months is not None:
        possible += 5
        earned += 5 if pet.age_months <= 120 else 4

    # Weight provided (5 pts): points for tracking, not for an exact number.
    if pet.weight_kg is not None and pet.weight_kg > 0:
        possible += 5
        earned += 5

    if possible == 0:
        return WellnessBreakdownItem(score=0, max_score=0)  # no baseline data → excluded
    return WellnessBreakdownItem(score=round(earned, 1), max_score=possible)


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
1. A SHORT narrative for a mobile app card: 1-2 sentences, max ~35 words total.
   Name the weakest SCORED dimension in a few words. Encouraging, non-alarmist. No preamble.
2. 3-5 specific, actionable recommendations ordered by priority.
   Be concrete — e.g. "Add 10 minutes to morning walks" not just "exercise more".
   Put the detail HERE, not in the narrative.
IMPORTANT: dimensions marked "not tracked (no data)" are MISSING data, NOT low scores.
Never tell the owner to improve a not-tracked dimension. If a not-tracked dimension is
important, you may gently suggest they START TRACKING it — but prioritise dimensions
that were actually scored. Base the narrative's "weakest area" only on scored dimensions.
Return ONLY valid JSON matching the required schema.
"""


def _fmt_dimension(label: str, item: WellnessBreakdownItem) -> str:
    if item.max_score == 0:
        return f"  {label:<13}not tracked (no data)"
    return f"  {label:<13}{item.score}/{item.max_score}"


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
        "Breakdown (only scored dimensions count toward the score):",
        _fmt_dimension("Activity:", breakdown.activity),
        _fmt_dimension("Sleep:", breakdown.sleep),
        _fmt_dimension("Diet:", breakdown.diet),
        _fmt_dimension("Symptoms:", breakdown.symptoms),
        _fmt_dimension("Preventive:", breakdown.preventive_care),
        _fmt_dimension("Baseline:", breakdown.baseline),
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
        WellnessBand.GOOD: "Your pet is doing well overall. There are a few small areas worth improving.",
        WellnessBand.FAIR: "Your pet's wellness is fair. Some dimensions need attention — check the breakdown above.",
        WellnessBand.CONCERNING: "Your pet's wellness is concerning this week. Consider reviewing diet, activity, and scheduling a vet check.",
        WellnessBand.CRITICAL: "Your pet's tracked data indicates a critical wellness level. Please consult a veterinarian promptly.",
    }
    recs = {
        WellnessBand.EXCELLENT: [
            "Maintain the current routine.",
            "Schedule a routine annual vet check.",
        ],
        WellnessBand.GOOD: [
            "Review the dimension with the lowest sub-score.",
            "Ensure consistent meal timing.",
        ],
        WellnessBand.FAIR: [
            "Increase daily active time.",
            "Log feeding more consistently.",
            "Book a vet appointment if symptoms persist.",
        ],
        WellnessBand.CONCERNING: [
            "Schedule a veterinary check-up soon.",
            "Improve feeding consistency.",
            "Increase monitored exercise.",
        ],
        WellnessBand.CRITICAL: [
            "Contact a veterinarian as soon as possible.",
            "Monitor symptoms closely.",
            "Avoid strenuous activity until assessed.",
        ],
    }
    return narratives[band], recs[band]


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
        # 1. Score each dimension
        activity_item = _score_activity(request.activity, request.pet.species)
        sleep_item = _score_sleep(request.activity, request.pet.species)
        diet_item = _score_diet(request.feeding, request.pet)
        symptoms_item, detected_condition = _score_symptoms(request.current_symptoms, predictor)
        preventive_item = _score_preventive(request.preventive_care)
        baseline_item = _score_baseline(request.pet)

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

        prompt = _build_narrative_prompt(
            request, breakdown, score, band, condition_cap, detected_condition
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
            return _fallback_narrative(band, score)
        except (ValidationError, ValueError) as exc:
            log_fallback("wellness.narrative", "invalid_response", exc)
            return _fallback_narrative(band, score)
        except Exception as exc:  # pragma: no cover
            log_fallback("wellness.narrative", "request_error", exc)
            return _fallback_narrative(band, score)
