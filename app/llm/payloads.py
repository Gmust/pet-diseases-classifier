"""Structured payloads Gemini is asked to return, and the deterministic
fallbacks used when it is unavailable or answers unusably.

Kept apart from the client so the fallback path stays readable and testable
without any notion of an API call.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

genai: Any
try:
    from google import genai
except ImportError:  # pragma: no cover - runtime guard for missing dependency
    genai = None


DEFAULT_DISCLAIMER = "This is an AI-assisted pre-assessment and not a veterinary diagnosis."
GENERAL_DISCLAIMER = (
    "General pet care information — not a substitute for professional veterinary advice."
)

logger = logging.getLogger(__name__)


def _bounded_chat_transcript(conversation: list[dict[str, str]], max_chars: int) -> str:
    """Keep newest turns first when fitting a chronological transcript budget."""
    lines = [
        f"{message.get('role', 'user').upper()}: {message.get('content', '').strip()}"
        for message in conversation
        if message.get("content", "").strip()
    ]
    if lines and len(lines[-1]) > max_chars:
        raise ValueError("Prompt instructions leave insufficient room for the latest chat message.")
    selected: list[str] = []
    used = 0
    for line in reversed(lines):
        separator = 1 if selected else 0
        if used + separator + len(line) > max_chars:
            continue
        selected.append(line)
        used += separator + len(line)
    return "\n".join(reversed(selected))


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


class ChatTurnPayload(BaseModel):
    mode: Literal["general", "health"] = Field(
        default="health",
        description="'general' for a general pet-care question, 'health' for a symptom/health concern.",
    )
    answer: str = Field(..., min_length=1, description="Conversational reply for the user.")
    symptom_summary: str = Field(
        default="",
        description="Updated rolling summary of symptoms so far (health mode). Carry forward unchanged for general mode.",
    )
    related_topics: list[str] = Field(
        default_factory=list,
        description="2-4 keyword tags for a general-care answer (general mode). Empty for health mode.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if the reply asks a follow-up question because the signal is still weak.",
    )


def fallback_chat_turn(
    predicted_condition: str,
    prior_summary: str | None,
    latest_message: str,
    low_confidence: bool,
) -> ChatTurnPayload:
    """Local reply used when Gemini is unavailable or returns an unusable response.

    Without the LLM we can't reliably tell general-care from health, so we default
    to the safer triage (health) path."""
    if low_confidence:
        answer = (
            "Thanks for the detail. I'm not fully certain yet — could you tell me a bit more "
            "about when this started, how your pet is eating and drinking, and any other changes "
            "you've noticed? In the meantime, keep your pet calm and watch closely for any worsening."
        )
        needs_clarification = True
    else:
        answer = (
            f"Based on what you've described, the symptoms may be related to {predicted_condition.lower()}. "
            "Keep monitoring your pet and seek veterinary advice if things continue or get worse."
        )
        needs_clarification = False

    # Keep the rolling summary moving even without the LLM: append the newest message.
    summary_bits = [s for s in [(prior_summary or "").strip(), (latest_message or "").strip()] if s]
    return ChatTurnPayload(
        mode="health",
        answer=answer,
        symptom_summary=" ".join(summary_bits)[:1000],
        needs_clarification=needs_clarification,
    )
