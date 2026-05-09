import logging
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

try:
    from google import genai
except ImportError:  # pragma: no cover - runtime guard for missing dependency
    genai = None


DEFAULT_DISCLAIMER = "This is an AI-assisted pre-assessment and not a veterinary diagnosis."

logger = logging.getLogger(__name__)


class ExplanationPayload(BaseModel):
    explanation: str = Field(..., min_length=1)
    disclaimer: str = Field(..., min_length=1)
    home_advice: list[str] = Field(
        default_factory=list,
        description="3-5 practical home-care tips the owner can follow right now.",
    )


def fallback_explanation(
    predicted_condition: str,
    default_home_advice: list[str] | None = None,
) -> ExplanationPayload:
    explanation = (
        f"The described symptoms may be related to {predicted_condition.lower()}. "
        "Observe your pet closely and seek veterinary advice if symptoms continue or worsen."
    )
    return ExplanationPayload(
        explanation=explanation,
        disclaimer=DEFAULT_DISCLAIMER,
        home_advice=default_home_advice or [],
    )


class GeminiService:
    def __init__(self, api_key: Optional[str], model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name
        self.client = None

        if api_key and genai is not None:
            self.client = genai.Client(api_key=api_key)
        elif genai is None:
            logger.warning("google-genai SDK is not available. Falling back to local explanation.")
        else:
            logger.warning("GEMINI_API_KEY is not set. Falling back to local explanation.")

    def generate_explanation(
        self,
        user_text: str,
        predicted_condition: str,
        default_home_advice: list[str] | None = None,
    ) -> ExplanationPayload:
        if self.client is None:
            return fallback_explanation(
                predicted_condition=predicted_condition,
                default_home_advice=default_home_advice,
            )

        prompt = f"""
You are a veterinary triage assistant.
You MUST follow these rules:
- The predicted condition is already decided by a classifier and cannot be changed.
- Explain that predicted condition in 2-3 sentences using cautious wording (may, might, could).
- Never claim certainty or a diagnosis.
- Include a short disclaimer.
- Write 3-5 practical home-care tips the owner can follow RIGHT NOW at home.
  Tips should be specific and actionable: diet adjustments, feeding schedule, rest, observation signs to watch for, things to avoid.
  Do NOT just say "visit a vet" — that goes in the disclaimer. Focus on what the owner can do themselves.
- Return ONLY valid JSON matching the required schema.

User symptom text: "{user_text}"
Predicted condition: "{predicted_condition}"
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.3,
                    "response_mime_type": "application/json",
                    "response_json_schema": ExplanationPayload.model_json_schema(),
                },
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response body.")

            parsed = ExplanationPayload.model_validate_json(response.text)
            if not parsed.disclaimer.strip():
                parsed = ExplanationPayload(
                    explanation=parsed.explanation,
                    disclaimer=DEFAULT_DISCLAIMER,
                    home_advice=parsed.home_advice,
                )
            # Fall back to static advice if Gemini returned an empty list
            if not parsed.home_advice and default_home_advice:
                parsed = ExplanationPayload(
                    explanation=parsed.explanation,
                    disclaimer=parsed.disclaimer,
                    home_advice=default_home_advice,
                )
            return parsed
        except (ValidationError, ValueError) as exc:
            logger.warning("Gemini response parsing failed: %s", exc)
            return fallback_explanation(
                predicted_condition=predicted_condition,
                default_home_advice=default_home_advice,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Gemini request failed: %s", exc)
            return fallback_explanation(
                predicted_condition=predicted_condition,
                default_home_advice=default_home_advice,
            )
