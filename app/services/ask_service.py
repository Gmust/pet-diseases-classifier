"""
General pet-care Q&A service.

Covers: breeds, diet, grooming, training, behaviour, housing, general care.
Supported species: dog, cat, rabbit, hamster, guinea pig, bird, fish, turtle.

Medical / symptom questions are redirected to the /predict endpoint.
Unsupported species receive a polite "not covered" response.
"""
from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from app.schemas import AskResponse, PetType, SUPPORTED_PET_TYPES

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None


ASK_DISCLAIMER = "General pet care information — not a substitute for professional veterinary advice."

_SUPPORTED_NAMES = "dogs, cats, rabbits, hamsters, guinea pigs, birds, fish, and turtles"

_NOT_SUPPORTED_ANSWER = (
    "We currently only have information for {species}. "
    "For other species, please consult a specialist exotic-animal veterinarian."
)

_MEDICAL_REDIRECT = (
    "This sounds like it could be a health or symptom-related question. "
    "Please use the /predict endpoint — describe your pet's symptoms there "
    "to get a condition assessment with urgency guidance and home-care advice."
)

logger = logging.getLogger(__name__)


# Internal model for structured Gemini output
class _AskPayload(BaseModel):
    answer: str = Field(..., min_length=1)
    related_topics: list[str] = Field(default_factory=list)
    is_medical: bool = Field(
        default=False,
        description="True if the question is about symptoms or illness rather than general care.",
    )
    is_unsupported_species: bool = Field(
        default=False,
        description="True if the question is about a species we do not cover.",
    )


_SYSTEM_PROMPT = f"""
You are a knowledgeable pet-care assistant for common household pets.

SUPPORTED SPECIES: {_SUPPORTED_NAMES}.

RULES — follow all of them strictly:
1. Only answer questions about: breeds, diet & nutrition, grooming, training, behaviour,
   housing & environment, reproduction, lifespan, and general care routines.
2. If the question is about symptoms, illness, injury, medication, or diagnosis,
   set is_medical=true and write a short answer explaining the user should use /predict.
3. If the question is about a species NOT in the supported list (e.g. reptiles other than
   turtles, exotic animals, livestock), set is_unsupported_species=true and write a short
   answer saying you do not currently cover that species.
4. Never claim to diagnose or treat any condition.
5. Keep answers clear and practical — 3-6 sentences or a short list.
6. Populate related_topics with 2-4 short keyword tags relevant to the answer.
7. Return ONLY valid JSON matching the required schema.
"""


def _fallback_response() -> AskResponse:
    return AskResponse(
        answer=(
            "I couldn't process your question right now. "
            "Please try rephrasing, or consult a veterinarian for health-related concerns."
        ),
        related_topics=[],
        disclaimer=ASK_DISCLAIMER,
    )


class AskService:
    def __init__(self, api_key: Optional[str], model_name: str = "gemini-2.5-flash") -> None:
        self.model_name = model_name
        self.client = None

        if api_key and genai is not None:
            self.client = genai.Client(api_key=api_key)
        elif genai is None:
            logger.warning("google-genai SDK not available — /ask will return fallback responses.")
        else:
            logger.warning("GEMINI_API_KEY not set — /ask will return fallback responses.")

    def answer(self, question: str, pet_type: PetType | None = None) -> AskResponse:
        # Fast-path: unsupported species specified explicitly in the request
        if pet_type is not None and pet_type not in SUPPORTED_PET_TYPES:
            return AskResponse(
                answer=_NOT_SUPPORTED_ANSWER.format(species=_SUPPORTED_NAMES),
                related_topics=[],
                disclaimer=ASK_DISCLAIMER,
            )

        if self.client is None:
            return _fallback_response()

        pet_context = f"The user's pet type is: {pet_type.value}." if pet_type else ""

        prompt = f"""
{pet_context}
User question: "{question}"
""".strip()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.4,
                    "system_instruction": _SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                    "response_json_schema": _AskPayload.model_json_schema(),
                },
            )

            if not response.text:
                raise ValueError("Gemini returned an empty response.")

            parsed: _AskPayload = _AskPayload.model_validate_json(response.text)

            # Gemini detected a medical question — redirect to /predict
            if parsed.is_medical:
                return AskResponse(
                    answer=_MEDICAL_REDIRECT,
                    related_topics=["health", "symptoms", "predict"],
                    disclaimer=ASK_DISCLAIMER,
                )

            # Gemini detected an unsupported species in the question text
            if parsed.is_unsupported_species:
                return AskResponse(
                    answer=_NOT_SUPPORTED_ANSWER.format(species=_SUPPORTED_NAMES),
                    related_topics=[],
                    disclaimer=ASK_DISCLAIMER,
                )

            return AskResponse(
                answer=parsed.answer,
                related_topics=parsed.related_topics,
                disclaimer=ASK_DISCLAIMER,
            )

        except (ValidationError, ValueError) as exc:
            logger.warning("AskService response parsing failed: %s", exc)
            return _fallback_response()
        except Exception as exc:  # pragma: no cover
            logger.warning("AskService Gemini request failed: %s", exc)
            return _fallback_response()
