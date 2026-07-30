from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UrgencyLevel(StrEnum):
    """How urgently the owner should seek veterinary attention."""

    MONITOR = "MONITOR"  # Watch at home; vet visit only if it worsens
    CONSULT_SOON = "CONSULT_SOON"  # Book an appointment within 1-3 days
    URGENT = "URGENT"  # Same-day or next-morning vet visit recommended
    EMERGENCY = "EMERGENCY"  # Go to an emergency clinic immediately


class SpecialistType(StrEnum):
    """Type of veterinary specialist best suited for the predicted condition."""

    GENERAL_VET = "general_vet"
    DERMATOLOGIST = "dermatologist"
    NEUROLOGIST = "neurologist"
    CARDIOLOGIST = "cardiologist"
    ONCOLOGIST = "oncologist"
    OPHTHALMOLOGIST = "ophthalmologist"
    INTERNIST = "internist"
    SURGEON = "surgeon"
    EMERGENCY_VET = "emergency_vet"


class DiseaseCategory(StrEnum):
    """Broad biomedical category the predicted condition belongs to."""

    INFECTIOUS = "INFECTIOUS"
    METABOLIC = "METABOLIC"
    STRUCTURAL = "STRUCTURAL"
    NEOPLASTIC = "NEOPLASTIC"
    IMMUNE = "IMMUNE"
    NEUROLOGICAL = "NEUROLOGICAL"
    CARDIOVASCULAR = "CARDIOVASCULAR"
    DERMATOLOGICAL = "DERMATOLOGICAL"
    GASTROINTESTINAL = "GASTROINTESTINAL"
    RESPIRATORY = "RESPIRATORY"
    OPHTHALMIC = "OPHTHALMIC"
    UROGENITAL = "UROGENITAL"
    TRAUMA = "TRAUMA"
    HEMATOLOGICAL = "HEMATOLOGICAL"
    REPRODUCTIVE = "REPRODUCTIVE"
    EAR = "EAR"


class PetType(StrEnum):
    """Supported pet species (optional hint on /chat, used by /wellness)."""

    DOG = "dog"
    CAT = "cat"
    RABBIT = "rabbit"
    HAMSTER = "hamster"
    GUINEA_PIG = "guinea_pig"
    BIRD = "bird"
    FISH = "fish"
    TURTLE = "turtle"
    OTHER = "other"


SUPPORTED_PET_TYPES: frozenset[PetType] = frozenset(
    {
        PetType.DOG,
        PetType.CAT,
        PetType.RABBIT,
        PetType.HAMSTER,
        PetType.GUINEA_PIG,
        PetType.BIRD,
        PetType.FISH,
        PetType.TURTLE,
    }
)


# ── Wellness score models ──────────────────────────────────────────────────


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


class WellnessPreventiveCare(BaseModel):
    """Derived from PetEvents over the last 12 months."""

    recent_vet_visit: bool = Field(default=False, alias="recentVetVisit")
    vaccinations_up_to_date: bool = Field(default=False, alias="vaccinationsUpToDate")

    model_config = ConfigDict(populate_by_name=True)


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
    preventive_care: WellnessPreventiveCare | None = Field(default=None, alias="preventiveCare")
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

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def require_score_bearing_evidence(self) -> "WellnessRequest":
        activity_evidence = self.activity is not None and any(
            value is not None
            for value in (
                self.activity.avg_steps_per_day,
                self.activity.avg_active_minutes_per_day,
                self.activity.avg_sleep_hours_per_day,
            )
        )
        feeding_evidence = self.feeding is not None and (
            self.feeding.avg_meals_per_day is not None
            or self.feeding.avg_calories_per_day is not None
            or bool(self.feeding.food_types)
            or self.feeding.consistency_days > 0
        )
        has_evidence = any(
            (
                self.pet.age_months is not None,
                self.pet.weight_kg is not None,
                activity_evidence,
                feeding_evidence,
                bool(self.active_conditions),
                self.preventive_care is not None,
                bool(self.current_symptoms and self.current_symptoms.strip()),
            )
        )
        if not has_evidence:
            raise ValueError(
                "At least one tracked wellness dimension is required; species alone is insufficient."
            )
        return self


class WellnessBreakdownItem(BaseModel):
    score: float
    max_score: float = Field(..., alias="maxScore")

    model_config = ConfigDict(populate_by_name=True)


class WellnessBreakdown(BaseModel):
    activity: WellnessBreakdownItem
    sleep: WellnessBreakdownItem
    diet: WellnessBreakdownItem
    symptoms: WellnessBreakdownItem
    preventive_care: WellnessBreakdownItem = Field(..., alias="preventiveCare")
    baseline: WellnessBreakdownItem

    model_config = ConfigDict(populate_by_name=True)


class WellnessResponse(BaseModel):
    wellness_score: int = Field(..., alias="wellnessScore", ge=0, le=100)
    band: WellnessBand
    band_label: str = Field(..., alias="bandLabel")
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
    disclaimer: str

    model_config = ConfigDict(populate_by_name=True)


# ── Predict models ─────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        examples=["My dog has been vomiting and has low appetite"],
    )


class PredictResponse(BaseModel):
    predicted_condition: str = Field(..., alias="predictedCondition")
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    disclaimer: str
    urgency: UrgencyLevel
    specialist: SpecialistType
    disease_category: DiseaseCategory = Field(..., alias="diseaseCategory")
    home_advice: list[str] = Field(default_factory=list, alias="homeAdvice")

    model_config = ConfigDict(populate_by_name=True)


# ── Chat models ────────────────────────────────────────────────────────────
# Stateless multi-turn chat. The backend owns all session state: it stores the
# message history and the rolling `symptomSummary`, and replays the relevant
# slice on every call. The microservice persists nothing.


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMode(StrEnum):
    """How the unified /chat endpoint handled the turn (decided under the hood)."""

    GENERAL = "general"  # general pet-care Q&A — `prediction` is null, `relatedTopics` set
    HEALTH = "health"  # symptom/health concern — `prediction` populated
    EMERGENCY = "emergency"  # red-flag detected — emergency message + advice


class ChatMessage(BaseModel):
    role: ChatRole
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    # Aliased fields use Annotated[...] form so the camelCase alias is attached
    # unambiguously (pydantic 2.12+ warns about Field(alias=...) defaults on
    # union-typed fields). populate_by_name=True keeps the snake_case names valid too.
    session_id: Annotated[
        str | None,
        Field(
            alias="sessionId",
            description="Opaque session id — used only for logging/telemetry, never for storage.",
        ),
    ] = None
    messages: Annotated[
        list[ChatMessage],
        Field(
            min_length=1,
            max_length=50,
            description="Recent conversation, oldest-first. The last entry MUST be the new user message.",
        ),
    ]
    symptom_summary: Annotated[
        str | None,
        Field(
            alias="symptomSummary",
            description="Rolling symptom summary returned by the previous turn. Null/empty on the first turn.",
        ),
    ] = None
    pet_type: Annotated[
        PetType | None,
        Field(
            alias="petType",
            description="Optional species hint — improves the conversational answer.",
        ),
    ] = None

    model_config = ConfigDict(populate_by_name=True)


class FeedingSummaryStatus(StrEnum):
    """How a pet's logged intake compares to its computed daily calorie target.

    No free-text title/message here on purpose — the frontend owns wording/i18n
    and renders per-status copy in the user's locale from this enum + the
    numeric fields (targetCalories/actualCalories/deviationPct).
    """

    EXTREME_UNDER_TARGET = "EXTREME_UNDER_TARGET"  # < -50% of target
    UNDER_TARGET = "UNDER_TARGET"  # -50% to -10% of target
    ON_TARGET = "ON_TARGET"  # within ±10% of target
    OVER_TARGET = "OVER_TARGET"  # +10% to +100% of target
    EXTREME_OVER_TARGET = "EXTREME_OVER_TARGET"  # > +100% of target


class FeedingProduct(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    calories: float = Field(..., ge=0, le=20_000)


class FeedingSummaryPet(BaseModel):
    pet_id: str = Field(..., alias="petId", min_length=1, max_length=100)
    species: PetType = Field(default=PetType.DOG)
    breed: str | None = Field(default=None, examples=["Labrador"])
    weight_kg: float = Field(..., alias="weightKg", gt=0, le=500)
    age_months: int | None = Field(
        default=None,
        alias="ageMonths",
        description="Used to apply a growth-energy multiplier for juveniles (< 12mo) "
        "instead of the adult maintenance factor — a growing kitten/puppy legitimately "
        "needs far more kcal/kg than an adult at the same weight.",
        ge=0,
        le=600,
    )
    products: list[FeedingProduct] = Field(default_factory=list, max_length=100)

    model_config = ConfigDict(populate_by_name=True)


class FeedingSummaryRequest(BaseModel):
    pets: list[FeedingSummaryPet] = Field(..., min_length=1, max_length=1000)


class FeedingSummaryResult(BaseModel):
    pet_id: str = Field(..., alias="petId")
    status: FeedingSummaryStatus
    target_calories: float = Field(..., alias="targetCalories")
    actual_calories: float = Field(..., alias="actualCalories")
    deviation_pct: float = Field(..., alias="deviationPct")

    model_config = ConfigDict(populate_by_name=True)


class FeedingSummaryResponse(BaseModel):
    results: list[FeedingSummaryResult]
    disclaimer: str

    model_config = ConfigDict(populate_by_name=True)


class TopKItem(BaseModel):
    condition: str
    confidence: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(populate_by_name=True)


class ChatPrediction(BaseModel):
    predicted_condition: str = Field(..., alias="predictedCondition")
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_k: list[TopKItem] = Field(default_factory=list, alias="topK")
    urgency: UrgencyLevel
    specialist: SpecialistType
    disease_category: DiseaseCategory = Field(..., alias="diseaseCategory")
    home_advice: list[str] = Field(default_factory=list, alias="homeAdvice")

    model_config = ConfigDict(populate_by_name=True)


class ChatResponse(BaseModel):
    mode: ChatMode = Field(
        ...,
        description="What the endpoint did this turn: general Q&A, health triage, or emergency. "
        "Branch on this — `prediction` is only present for health/emergency.",
    )
    answer: str = Field(..., description="Conversational reply for the user.")
    symptom_summary: str = Field(
        ...,
        alias="symptomSummary",
        description="Updated rolling symptom summary — the backend MUST persist this and send it back next turn.",
    )
    prediction: ChatPrediction | None = Field(
        default=None,
        description="Classifier result. Present for mode=health/emergency; null for mode=general.",
    )
    related_topics: list[str] = Field(
        default_factory=list,
        alias="relatedTopics",
        description="Keyword tags for a general-care answer (mode=general). Empty otherwise.",
    )
    needs_clarification: bool = Field(
        default=False,
        alias="needsClarification",
        description="True when confidence was low and the answer asks a follow-up question instead of asserting.",
    )
    disclaimer: str

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "mode": "health",
                "answer": "These symptoms may be related to digestive issues. How long has this been happening?",
                "symptomSummary": "The dog has been vomiting and has a reduced appetite.",
                "prediction": {
                    "predictedCondition": "Digestive Issues",
                    "confidence": 0.84,
                    "topK": [
                        {"condition": "Digestive Issues", "confidence": 0.84},
                        {"condition": "Infectious and Parasitic Diseases", "confidence": 0.09},
                    ],
                    "urgency": "CONSULT_SOON",
                    "specialist": "general_vet",
                    "diseaseCategory": "GASTROINTESTINAL",
                    "homeAdvice": ["Ensure fresh water is available."],
                },
                "relatedTopics": [],
                "needsClarification": True,
                "disclaimer": "This is an AI-assisted pre-assessment and not a veterinary diagnosis.",
            }
        },
    )
