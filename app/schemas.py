from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class UrgencyLevel(str, Enum):
    """How urgently the owner should seek veterinary attention."""

    MONITOR = "MONITOR"           # Watch at home; vet visit only if it worsens
    CONSULT_SOON = "CONSULT_SOON" # Book an appointment within 1-3 days
    URGENT = "URGENT"             # Same-day or next-morning vet visit recommended
    EMERGENCY = "EMERGENCY"       # Go to an emergency clinic immediately


class SpecialistType(str, Enum):
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


class DiseaseCategory(str, Enum):
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


class PetType(str, Enum):
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


SUPPORTED_PET_TYPES: frozenset[PetType] = frozenset({
    PetType.DOG,
    PetType.CAT,
    PetType.RABBIT,
    PetType.HAMSTER,
    PetType.GUINEA_PIG,
    PetType.BIRD,
    PetType.FISH,
    PetType.TURTLE,
})


# ── Wellness score models ──────────────────────────────────────────────────

class WellnessBand(str, Enum):
    EXCELLENT = "EXCELLENT"    # 90-100
    GOOD = "GOOD"              # 75-89
    FAIR = "FAIR"              # 60-74
    CONCERNING = "CONCERNING"  # 40-59
    CRITICAL = "CRITICAL"      # 0-39


class TrendDirection(str, Enum):
    IMPROVING = "IMPROVING"    # score rose by > 3 pts
    STABLE = "STABLE"          # score changed by ≤ 3 pts
    DECLINING = "DECLINING"    # score fell by > 3 pts


class WellnessPet(BaseModel):
    species: str = Field(..., examples=["dog"])
    breed: str | None = Field(default=None, examples=["Labrador"])
    age_months: int | None = Field(default=None, alias="ageMonths", examples=[36])
    sex: str | None = Field(default=None, examples=["male"])
    weight_kg: float | None = Field(default=None, alias="weightKg", examples=[28.5])
    behavioral_notes: str | None = Field(default=None, alias="behavioralNotes")

    model_config = ConfigDict(populate_by_name=True)


class WellnessActivity(BaseModel):
    """Aggregated from ActivityDailies over the last N days."""
    avg_steps_per_day: float | None = Field(default=None, alias="avgStepsPerDay")
    avg_active_minutes_per_day: float | None = Field(default=None, alias="avgActiveMinutesPerDay")
    avg_sleep_hours_per_day: float | None = Field(default=None, alias="avgSleepHoursPerDay")
    days_tracked: int = Field(default=0, alias="daysTracked")

    model_config = ConfigDict(populate_by_name=True)


class WellnessFeeding(BaseModel):
    """Aggregated from FeedingLogs over the last N days."""
    avg_meals_per_day: float | None = Field(default=None, alias="avgMealsPerDay")
    avg_calories_per_day: float | None = Field(default=None, alias="avgCaloriesPerDay")
    food_types: list[str] = Field(default_factory=list, alias="foodTypes")
    consistency_days: int = Field(default=0, alias="consistencyDays")

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
    active_conditions: list[WellnessCondition] = Field(default_factory=list, alias="activeConditions")
    active_medications: list[WellnessMedication] = Field(default_factory=list, alias="activeMedications")
    preventive_care: WellnessPreventiveCare | None = Field(default=None, alias="preventiveCare")
    current_symptoms: str | None = Field(
        default=None,
        alias="currentSymptoms",
        description="Optional free-text symptom description — passed through the classifier.",
    )
    previous_score: int | None = Field(
        default=None,
        alias="previousScore",
        description="Last wellness score for this pet — used to calculate trend.",
        ge=0,
        le=100,
    )

    model_config = ConfigDict(populate_by_name=True)


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

class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMode(str, Enum):
    """How the unified /chat endpoint handled the turn (decided under the hood)."""

    GENERAL = "general"      # general pet-care Q&A — `prediction` is null, `relatedTopics` set
    HEALTH = "health"        # symptom/health concern — `prediction` populated
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
        Field(alias="sessionId",
              description="Opaque session id — used only for logging/telemetry, never for storage."),
    ] = None
    messages: Annotated[
        list[ChatMessage],
        Field(min_length=1, max_length=50,
              description="Recent conversation, oldest-first. The last entry MUST be the new user message."),
    ]
    symptom_summary: Annotated[
        str | None,
        Field(alias="symptomSummary",
              description="Rolling symptom summary returned by the previous turn. Null/empty on the first turn."),
    ] = None
    pet_type: Annotated[
        PetType | None,
        Field(alias="petType", description="Optional species hint — improves the conversational answer."),
    ] = None

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

    model_config = ConfigDict(populate_by_name=True)

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "predictedCondition": "Digestive Issues",
                "confidence": 0.84,
                "explanation": (
                    "The symptoms are most consistent with digestive issues. "
                    "Monitor your pet closely and contact a veterinarian if symptoms persist."
                ),
                "disclaimer": "This is an AI-assisted pre-assessment and not a veterinary diagnosis.",
                "urgency": "CONSULT_SOON",
                "specialist": "general_vet",
                "diseaseCategory": "GASTROINTESTINAL",
                "homeAdvice": [
                    "Withhold food for 12-24 hours (water only) to rest the stomach.",
                    "Offer small portions of bland food (boiled chicken and plain rice).",
                    "Ensure fresh water is always available.",
                    "Monitor stool consistency and frequency.",
                ],
            }
        },
    )
