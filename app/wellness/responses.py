"""Response models for `POST /wellness`.

Split from the request models because the scoring code builds these while
only the API layer parses those — they change for different reasons.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.wellness.schemas import (
    ReminderType,
    TrendDirection,
    WellnessBand,
    WellnessDimension,
    WellnessDimensionAvailability,
    WellnessEvaluationWindow,
    WellnessReasonCode,
    WellnessScoreStatus,
)


class WellnessBreakdownItem(BaseModel):
    score: float
    max_score: float = Field(..., alias="maxScore")
    availability: WellnessDimensionAvailability
    included: bool = Field(
        description="False when the source data is unavailable and this dimension is excluded from the total.",
    )
    reason_codes: list[WellnessReasonCode] = Field(
        ...,
        alias="reasonCodes",
        min_length=1,
    )
    evidence: dict[str, bool | int | float | str] = Field(
        default_factory=dict,
        max_length=8,
        description="Allowlisted scalar values used by the deterministic scoring rule.",
    )

    model_config = ConfigDict(populate_by_name=True)

    @property
    def applicable(self) -> bool:
        """True unless the dimension does not apply to this species at all."""
        return self.availability != WellnessDimensionAvailability.NOT_APPLICABLE


class WellnessBreakdown(BaseModel):
    activity: WellnessBreakdownItem
    sleep: WellnessBreakdownItem
    diet: WellnessBreakdownItem
    symptoms: WellnessBreakdownItem
    preventive_care: WellnessBreakdownItem = Field(..., alias="preventiveCare")
    baseline: WellnessBreakdownItem

    model_config = ConfigDict(populate_by_name=True)


class WellnessReminder(BaseModel):
    """Actionable reminder suggestion using the backend ReminderType wire value."""

    reminder: ReminderType
    text: str = Field(..., min_length=1)


class WellnessTrackingRecommendation(BaseModel):
    dimension: WellnessDimension
    text: str = Field(..., min_length=1)
    required_inputs: list[str] = Field(
        ...,
        alias="requiredInputs",
        min_length=1,
    )
    suggested_reminder_types: list[ReminderType] = Field(
        default_factory=list,
        alias="suggestedReminderTypes",
        description=(
            "Backend-compatible reminder types a client may offer for explicit "
            "user-confirmed creation; this service does not schedule reminders."
        ),
    )

    model_config = ConfigDict(populate_by_name=True)


class WellnessResponse(BaseModel):
    wellness_score: int | None = Field(default=None, alias="wellnessScore", ge=0, le=100)
    band: WellnessBand | None = None
    band_label: str | None = Field(default=None, alias="bandLabel")
    score_status: WellnessScoreStatus = Field(..., alias="scoreStatus")
    data_coverage: float = Field(..., alias="dataCoverage", ge=0, le=1)
    calculation_version: str = Field(..., alias="calculationVersion", pattern=r"^\d+\.\d+\.\d+$")
    evaluated_at: datetime = Field(..., alias="evaluatedAt")
    evaluation_window: WellnessEvaluationWindow | None = Field(
        default=None,
        alias="evaluationWindow",
    )
    trend: TrendDirection | None = None
    breakdown: WellnessBreakdown
    condition_cap: int | None = Field(default=None, alias="conditionCap")
    classifier_condition: str | None = Field(
        default=None,
        alias="classifierCondition",
        description="Condition detected from currentSymptoms, if provided.",
    )
    narrative: str
    recommendations: list[str]
    reminders: list[WellnessReminder] = Field(default_factory=list)
    tracking_recommendations: list[WellnessTrackingRecommendation] = Field(
        default_factory=list,
        alias="trackingRecommendations",
    )
    disclaimer: str

    model_config = ConfigDict(populate_by_name=True)
