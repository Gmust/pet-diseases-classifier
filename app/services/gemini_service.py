import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.ml.protocols import GeneratorMetadata
from app.services.gemini_rotation import RotatingGeminiClient
from app.services.generation_policy import (
    MAX_PROMPT_CHARS,
    GenerationTimeoutError,
    bound_prompt,
    call_with_policy,
    log_fallback,
)

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


class GeminiService:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        api_keys: list[str] | None = None,
    ) -> None:
        """`api_keys` (if given) takes priority over the single `api_key` and
        enables quota rotation: a 429 on one key retries on the next before
        falling back to the local template."""
        self.model_name = model_name
        self.client: Any = None
        keys = api_keys or ([api_key] if api_key else [])

        if keys and genai is not None:
            self.client = RotatingGeminiClient(keys)
        elif genai is None:
            logger.warning("google-genai SDK is not available. Falling back to local explanation.")
        else:
            logger.warning("GEMINI_API_KEY is not set. Falling back to local explanation.")

    @property
    def metadata(self) -> GeneratorMetadata:
        return GeneratorMetadata(
            backend="gemini" if self.client is not None else "fallback",
            model_name=self.model_name,
            available=self.client is not None,
        )

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
            response = call_with_policy(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=bound_prompt(prompt),
                    config={
                        "temperature": 0.3,
                        "response_mime_type": "application/json",
                        "response_json_schema": ExplanationPayload.model_json_schema(),
                    },
                )
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
        except GenerationTimeoutError as exc:
            log_fallback("predict.explanation", "timeout", exc)
            return fallback_explanation(
                predicted_condition=predicted_condition,
                default_home_advice=default_home_advice,
            )
        except (ValidationError, ValueError) as exc:
            log_fallback("predict.explanation", "invalid_response", exc)
            return fallback_explanation(
                predicted_condition=predicted_condition,
                default_home_advice=default_home_advice,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            log_fallback("predict.explanation", "request_error", exc)
            return fallback_explanation(
                predicted_condition=predicted_condition,
                default_home_advice=default_home_advice,
            )

    def generate_chat_turn(
        self,
        conversation: list[dict[str, str]],
        predicted_condition: str,
        confidence: float,
        prior_summary: str | None,
        low_confidence: bool,
        pet_type: str | None = None,
    ) -> ChatTurnPayload:
        """One chat turn: produce a conversational answer AND an updated rolling summary.

        This is the single Gemini call per turn. Asking it to also return
        `symptom_summary` means the rolling distillation costs no extra API call.

        Args:
            conversation: Recent messages as [{"role": "user"|"assistant", "content": str}], oldest-first.
            predicted_condition: Classifier's current top label (cannot be overridden).
            confidence: Classifier confidence for that label (0-1).
            prior_summary: Rolling symptom summary from the previous turn.
            low_confidence: True when confidence is below the configured threshold.
            pet_type: Optional species hint.
        """
        latest_message = next(
            (m["content"] for m in reversed(conversation) if m.get("role") == "user"),
            "",
        )

        if self.client is None:
            return fallback_chat_turn(
                predicted_condition=predicted_condition,
                prior_summary=prior_summary,
                latest_message=latest_message,
                low_confidence=low_confidence,
            )

        pet_context = f"Pet species: {pet_type}.\n" if pet_type else ""
        confidence_guidance = (
            "The classifier confidence is LOW. Do NOT assert the condition. Instead, ask one or two "
            "focused follow-up questions to gather more detail, and set needs_clarification to true."
            if low_confidence
            else "The classifier confidence is adequate. Explain the likely condition with cautious wording "
            "and set needs_clarification to false (unless genuinely ambiguous)."
        )

        prior_summary_text = bound_prompt((prior_summary or "").strip(), max_chars=1000)
        prompt_prefix = f"""
You are a pet-care chat assistant having an ongoing conversation with a pet owner.

FIRST, classify the owner's LATEST message into one of two modes:
- "general": a general pet-care question (breeds, diet, nutrition, grooming, training,
  behaviour, housing, lifespan, general routines) with no symptom/illness concern.
- "health": anything about symptoms, illness, injury, pain, or the pet not being well.

THEN respond according to the mode:

If mode = "general":
- Answer the question directly and practically in 3-6 sentences. IGNORE the predicted
  condition below — it is irrelevant to a general question.
- Populate `related_topics` with 2-4 short keyword tags.
- Set `symptom_summary` to EXACTLY the prior symptom summary, unchanged (do not add
  general chit-chat to it). Set `needs_clarification` to false.

If mode = "health":
- The predicted condition below was decided by a classifier and CANNOT be changed or contradicted.
- Reply conversationally using the whole conversation. Use cautious wording (may, might, could);
  never claim certainty or give a definitive diagnosis.
- {confidence_guidance}
- Maintain a running `symptom_summary`: one short paragraph capturing every symptom and relevant
  detail across the WHOLE conversation (merge prior summary with new info, drop non-medical chit-chat).
- Leave `related_topics` empty.

ALWAYS:
- Keep `answer` to 2-6 sentences. Do not include a disclaimer (added separately).
- Return ONLY valid JSON matching the required schema (including the `mode` field).

{pet_context}Predicted condition (only relevant if mode=health): "{predicted_condition}" (confidence {confidence:.2f})
Prior symptom summary: "{prior_summary_text}"

Conversation so far (oldest first):
"""
        transcript = _bounded_chat_transcript(
            conversation, max_chars=MAX_PROMPT_CHARS - len(prompt_prefix)
        )
        prompt = prompt_prefix + transcript
        try:
            response = call_with_policy(
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": 0.4,
                        "response_mime_type": "application/json",
                        "response_json_schema": ChatTurnPayload.model_json_schema(),
                    },
                )
            )
            if not response.text:
                raise ValueError("Gemini returned an empty response body.")

            parsed = ChatTurnPayload.model_validate_json(response.text)
            # Never lose context: if the model returned an empty summary, retain the prior one.
            if not parsed.symptom_summary.strip():
                if parsed.mode == "general":
                    # A general question must not alter the medical summary.
                    summary = (prior_summary or "").strip()
                else:
                    summary = fallback_chat_turn(
                        predicted_condition, prior_summary, latest_message, low_confidence
                    ).symptom_summary
                parsed = ChatTurnPayload(
                    mode=parsed.mode,
                    answer=parsed.answer,
                    symptom_summary=summary,
                    related_topics=parsed.related_topics,
                    needs_clarification=parsed.needs_clarification,
                )
            return parsed
        except GenerationTimeoutError as exc:
            log_fallback("chat.turn", "timeout", exc)
            return fallback_chat_turn(
                predicted_condition, prior_summary, latest_message, low_confidence
            )
        except (ValidationError, ValueError) as exc:
            log_fallback("chat.turn", "invalid_response", exc)
            return fallback_chat_turn(
                predicted_condition, prior_summary, latest_message, low_confidence
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            log_fallback("chat.turn", "request_error", exc)
            return fallback_chat_turn(
                predicted_condition, prior_summary, latest_message, low_confidence
            )
