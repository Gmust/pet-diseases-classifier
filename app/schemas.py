from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["My dog has been vomiting and has low appetite"])


class PredictResponse(BaseModel):
    predicted_condition: str = Field(..., alias="predictedCondition")
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    disclaimer: str

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
            }
        },
    )

