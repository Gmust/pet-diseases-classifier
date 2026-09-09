"""Actionable reminders, and the filter that stops a recommendation from
repeating something a reminder already says.
"""

from __future__ import annotations

import logging

from app.domain.enums import UrgencyLevel
from app.wellness.responses import (
    WellnessBreakdown,
    WellnessBreakdownItem,
    WellnessReminder,
)
from app.wellness.schemas import (
    ReminderType,
    WellnessRequest,
)

logger = logging.getLogger(__name__)


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
    ReminderType.ACTIVITY: frozenset({"activity", "active time", "exercise", "walk", "strenuous"}),
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
    ReminderType.PARASITE_TREATMENT: frozenset({"parasite", "flea", "tick", "heartworm", "deworm"}),
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
    ReminderType.WEIGHING: frozenset({"weigh", "weight", "body condition"}),
    ReminderType.DEWORMING: frozenset({"deworm", "worming", "worm"}),
    ReminderType.BATHING: frozenset({"bath", "shampoo"}),
    ReminderType.BRUSHING: frozenset({"brush", "coat", "shedding", "matting"}),
    ReminderType.EAR_CLEANING: frozenset({"ear clean", "ear canal", "clean the ears"}),
    ReminderType.NAIL_TRIMMING: frozenset({"nail", "claw"}),
    ReminderType.PAW_CARE: frozenset({"paw", "pad"}),
    ReminderType.TEETH_CLEANING: frozenset({"teeth", "tooth", "dental", "gum"}),
}

# Gemini must never give medication, parasite-product, or supplement advice.
_MEDICATION_ADVICE_TERMS = (
    _REMINDER_RECOMMENDATION_TERMS[ReminderType.MEDICATION]
    | _REMINDER_RECOMMENDATION_TERMS[ReminderType.PARASITE_TREATMENT]
    | _REMINDER_RECOMMENDATION_TERMS[ReminderType.DEWORMING]
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
