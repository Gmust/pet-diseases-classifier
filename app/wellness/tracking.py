"""Tracking recommendations: what the owner should start logging.

Deterministic, and the primary payload when data is too thin to score.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import NamedTuple

from app.wellness.responses import (
    WellnessBreakdown,
    WellnessReminder,
    WellnessTrackingRecommendation,
)
from app.wellness.schemas import (
    ReminderType,
    WellnessDimension,
    WellnessDimensionAvailability,
    WellnessRequest,
)
from app.wellness.scoring.aggregate import _dimension_items

logger = logging.getLogger(__name__)


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
        reminder_types=(ReminderType.WEIGHING,),
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


# Routine grooming care the owner is expected to keep up with month to month.
# Deworming and other parasite products are deliberately excluded: this service
# never suggests medication-style treatments.
_ROUTINE_CARE_TYPES: tuple[ReminderType, ...] = (
    ReminderType.BATHING,
    ReminderType.BRUSHING,
    ReminderType.NAIL_TRIMMING,
    ReminderType.EAR_CLEANING,
    ReminderType.PAW_CARE,
    ReminderType.TEETH_CLEANING,
)

_ROUTINE_CARE_MAX_AGE_DAYS = 30

_ROUTINE_CARE_REQUIRED_INPUTS = ("routineCare",)

_ROUTINE_CARE_GAP_TEXT = (
    "No grooming or routine-care activity has been recorded in the last "
    f"{_ROUTINE_CARE_MAX_AGE_DAYS} days. Log routine care to keep it on schedule."
)

_ROUTINE_CARE_PARTIAL_TEXT = (
    "Some routine grooming care has not been recorded in the last "
    f"{_ROUTINE_CARE_MAX_AGE_DAYS} days."
)


def _routine_care_as_of(request: WellnessRequest) -> date:
    """Anchor staleness to the evaluated period, not to wall-clock time."""
    if request.evaluation_window is not None:
        return request.evaluation_window.end_date
    return datetime.now(UTC).date()


def _get_routine_care_recommendations(
    request: WellnessRequest,
) -> list[WellnessTrackingRecommendation]:
    """Suggest grooming reminders for routine care the backend reported nothing
    recent for. Suggestion only — routine care is never scored."""
    as_of = _routine_care_as_of(request)
    last_done = {
        entry.type: entry.last_done_at
        for entry in request.routine_care
        if entry.last_done_at is not None
    }

    def is_recent(reminder_type: ReminderType) -> bool:
        done_at = last_done.get(reminder_type)
        return done_at is not None and (as_of - done_at).days <= _ROUTINE_CARE_MAX_AGE_DAYS

    # A recent unspecified grooming record stands in for every grooming label.
    if is_recent(ReminderType.GROOMING):
        return []

    stale = [reminder_type for reminder_type in _ROUTINE_CARE_TYPES if not is_recent(reminder_type)]
    if not stale:
        return []

    return [
        WellnessTrackingRecommendation(
            dimension=WellnessDimension.ROUTINE_CARE,
            text=(
                _ROUTINE_CARE_GAP_TEXT
                if len(stale) == len(_ROUTINE_CARE_TYPES)
                else _ROUTINE_CARE_PARTIAL_TEXT
            ),
            required_inputs=list(_ROUTINE_CARE_REQUIRED_INPUTS),
            suggested_reminder_types=stale,
        )
    ]
