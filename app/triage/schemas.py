"""Request and response models for `POST /predict` and `POST /chat`."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import DiseaseCategory, PetType, SpecialistType, UrgencyLevel

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
