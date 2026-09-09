"""Request and response models for `POST /wellness`."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.wellness.enums import (
    ReminderType,
    TrendDirection,
    WellnessBand,
    WellnessDimension,
    WellnessDimensionAvailability,
    WellnessReasonCode,
    WellnessScoreStatus,
)

# Re-exported: the enums moved to `enums.py`, but callers import them from here.
__all__ = [
    "ReminderType",
    "TrendDirection",
    "WellnessActivity",
    "WellnessBand",
    "WellnessCondition",
    "WellnessDimension",
    "WellnessDimensionAvailability",
    "WellnessEvaluationWindow",
    "WellnessFeeding",
    "WellnessMedication",
    "WellnessPet",
    "WellnessPreventiveCare",
    "WellnessReasonCode",
    "WellnessRequest",
    "WellnessRoutineCareEntry",
    "WellnessScoreStatus",
    "WellnessWeightMeasurement",
]


class WellnessPet(BaseModel):
    species: str = Field(..., examples=["dog"])
    breed: str | None = Field(default=None, examples=["Labrador"])
    age_months: int | None = Field(
        default=None,
        alias="ageMonths",
        examples=[36],
        ge=0,
        le=600,
    )
    sex: str | None = Field(default=None, examples=["male"])
    weight_kg: float | None = Field(
        default=None,
        alias="weightKg",
        examples=[28.5],
        gt=0,
        le=500,
    )
    behavioral_notes: str | None = Field(default=None, alias="behavioralNotes")

    model_config = ConfigDict(populate_by_name=True)


class WellnessActivity(BaseModel):
    """Aggregated from ActivityDailies over the last N days."""

    avg_steps_per_day: float | None = Field(
        default=None,
        alias="avgStepsPerDay",
        ge=0,
        le=1_000_000,
    )
    avg_active_minutes_per_day: float | None = Field(
        default=None,
        alias="avgActiveMinutesPerDay",
        ge=0,
        le=1440,
    )
    avg_sleep_hours_per_day: float | None = Field(
        default=None,
        alias="avgSleepHoursPerDay",
        ge=0,
        le=24,
    )
    days_tracked: int = Field(default=0, alias="daysTracked", ge=0, le=365)

    model_config = ConfigDict(populate_by_name=True)


class WellnessFeeding(BaseModel):
    """Aggregated from FeedingLogs over the last N days."""

    avg_meals_per_day: float | None = Field(
        default=None,
        alias="avgMealsPerDay",
        ge=0,
        le=20,
    )
    avg_calories_per_day: float | None = Field(
        default=None,
        alias="avgCaloriesPerDay",
        ge=0,
        le=100_000,
    )
    food_types: list[str] = Field(default_factory=list, alias="foodTypes")
    consistency_days: int = Field(default=0, alias="consistencyDays", ge=0, le=7)

    model_config = ConfigDict(populate_by_name=True)


class WellnessCondition(BaseModel):
    """From PetConditions where IsActive = true."""

    name: str
    type_label: str | None = Field(default=None, alias="typeLabel")

    model_config = ConfigDict(populate_by_name=True)


class WellnessMedication(BaseModel):
    """From PetMedications where EndDate is null or in the future."""

    name: str
    frequency: str | None = None
    scheduled_doses: int | None = Field(default=None, alias="scheduledDoses", ge=0)
    completed_doses: int | None = Field(default=None, alias="completedDoses", ge=0)

    model_config = ConfigDict(populate_by_name=True)


class WellnessWeightMeasurement(BaseModel):
    """Compatible with the backend's PetWeightLogResponseDto fields."""

    weight_kg: float = Field(..., alias="weightKg", gt=0)
    measured_at: datetime = Field(..., alias="measuredAt")

    model_config = ConfigDict(populate_by_name=True)


class WellnessPreventiveCare(BaseModel):
    """Derived from PetEvents over the last 12 months."""

    recent_vet_visit: bool = Field(default=False, alias="recentVetVisit")
    vaccinations_up_to_date: bool = Field(default=False, alias="vaccinationsUpToDate")

    model_config = ConfigDict(populate_by_name=True)


class WellnessRoutineCareEntry(BaseModel):
    """One routine-care activity the backend has a record for.

    Derived from PetEvents/CareRecords of a grooming-style type. `lastDoneAt`
    is null when the backend knows the activity is tracked but has no record.
    """

    type: ReminderType
    last_done_at: date | None = Field(default=None, alias="lastDoneAt")

    model_config = ConfigDict(populate_by_name=True)


class WellnessEvaluationWindow(BaseModel):
    start_date: date = Field(..., alias="startDate")
    end_date: date = Field(..., alias="endDate")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def validate_date_order(self) -> "WellnessEvaluationWindow":
        if self.start_date > self.end_date:
            raise ValueError("evaluationWindow.startDate must be on or before endDate")
        return self


class WellnessRequest(BaseModel):
    pet: WellnessPet
    activity: WellnessActivity | None = None
    feeding: WellnessFeeding | None = None
    active_conditions: list[WellnessCondition] = Field(
        default_factory=list, alias="activeConditions"
    )
    active_medications: list[WellnessMedication] = Field(
        default_factory=list, alias="activeMedications"
    )
    weight_history: list[WellnessWeightMeasurement] = Field(
        default_factory=list, alias="weightHistory"
    )
    preventive_care: WellnessPreventiveCare | None = Field(default=None, alias="preventiveCare")
    routine_care: list[WellnessRoutineCareEntry] = Field(
        default_factory=list,
        alias="routineCare",
        description=(
            "Grooming-style care records. Anything absent or older than 30 days is "
            "returned as a routine-care tracking suggestion."
        ),
    )
    current_symptoms: str | None = Field(
        default=None,
        alias="currentSymptoms",
        description="Optional free-text symptom description — passed through the classifier.",
        max_length=4000,
    )
    previous_score: int | None = Field(
        default=None,
        alias="previousScore",
        description="Last wellness score for this pet — used to calculate trend.",
        ge=0,
        le=100,
    )
    evaluation_window: WellnessEvaluationWindow | None = Field(
        default=None,
        alias="evaluationWindow",
        description="Optional inclusive date range represented by the aggregated inputs.",
    )

    model_config = ConfigDict(populate_by_name=True)
