"""Enumerated vocabulary for `/wellness`: bands, statuses, dimensions, reason
codes, and the backend-compatible reminder types.

Split from `schemas.py` so request/response models and the values they speak
stay separately importable (and so neither module outgrows the size gate).
"""

from enum import StrEnum


class WellnessBand(StrEnum):
    EXCELLENT = "EXCELLENT"  # 90-100
    GOOD = "GOOD"  # 75-89
    FAIR = "FAIR"  # 60-74
    CONCERNING = "CONCERNING"  # 40-59
    CRITICAL = "CRITICAL"  # 0-39


class TrendDirection(StrEnum):
    IMPROVING = "IMPROVING"  # score rose by > 3 pts
    STABLE = "STABLE"  # score changed by ≤ 3 pts
    DECLINING = "DECLINING"  # score fell by > 3 pts


class WellnessScoreStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class WellnessDimensionAvailability(StrEnum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class WellnessDimension(StrEnum):
    ACTIVITY = "Activity"
    SLEEP = "Sleep"
    DIET = "Diet"
    SYMPTOMS = "Symptoms"
    PREVENTIVE_CARE = "PreventiveCare"
    BASELINE = "Baseline"
    # Tracking-only: routine care is suggested, never scored.
    ROUTINE_CARE = "RoutineCare"


class WellnessReasonCode(StrEnum):
    ACTIVITY_DATA_MISSING = "ACTIVITY_DATA_MISSING"
    ACTIVITY_TARGET_MET = "ACTIVITY_TARGET_MET"
    ACTIVITY_BELOW_TARGET = "ACTIVITY_BELOW_TARGET"
    ACTIVITY_NOT_APPLICABLE = "ACTIVITY_NOT_APPLICABLE"
    SLEEP_DATA_MISSING = "SLEEP_DATA_MISSING"
    SLEEP_WITHIN_RANGE = "SLEEP_WITHIN_RANGE"
    SLEEP_OUTSIDE_RANGE = "SLEEP_OUTSIDE_RANGE"
    SLEEP_NOT_APPLICABLE = "SLEEP_NOT_APPLICABLE"
    DIET_DATA_MISSING = "DIET_DATA_MISSING"
    DIET_TRACKING_STRONG = "DIET_TRACKING_STRONG"
    DIET_TRACKING_NEEDS_ATTENTION = "DIET_TRACKING_NEEDS_ATTENTION"
    SYMPTOMS_NOT_REPORTED = "SYMPTOMS_NOT_REPORTED"
    SYMPTOM_CLASSIFIER_UNAVAILABLE = "SYMPTOM_CLASSIFIER_UNAVAILABLE"
    SYMPTOM_CLASSIFIER_FAILED = "SYMPTOM_CLASSIFIER_FAILED"
    SYMPTOM_RESULT_AVAILABLE = "SYMPTOM_RESULT_AVAILABLE"
    PREVENTIVE_CARE_DATA_MISSING = "PREVENTIVE_CARE_DATA_MISSING"
    PREVENTIVE_CARE_CURRENT = "PREVENTIVE_CARE_CURRENT"
    PREVENTIVE_CARE_NEEDS_ATTENTION = "PREVENTIVE_CARE_NEEDS_ATTENTION"
    BASELINE_DATA_MISSING = "BASELINE_DATA_MISSING"
    BASELINE_STABLE = "BASELINE_STABLE"
    BASELINE_NEEDS_ATTENTION = "BASELINE_NEEDS_ATTENTION"


class ReminderType(StrEnum):
    """Exact wire values from the C# backend's ReminderType enum.

    Declaration order mirrors the backend enum, which persists as an integer:
    values are appended there, never inserted.
    """

    FEEDING = "Feeding"
    ACTIVITY = "Activity"
    MEDICATION = "Medication"
    VACCINATION = "Vaccination"
    PARASITE_TREATMENT = "ParasiteTreatment"
    VET_VISIT = "VetVisit"
    # Unspecified grooming; the values below are the specific grooming labels.
    GROOMING = "Grooming"
    WEIGHING = "Weighing"
    DEWORMING = "Deworming"
    BATHING = "Bathing"
    BRUSHING = "Brushing"
    EAR_CLEANING = "EarCleaning"
    NAIL_TRIMMING = "NailTrimming"
    PAW_CARE = "PawCare"
    TEETH_CLEANING = "TeethCleaning"
