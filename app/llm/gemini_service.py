"""Gemini client for triage prose: condition explanations and chat turns.

Generates human-facing text only. The classifier remains the sole decider of
the predicted condition — nothing here may set or override it.
"""

import logging
from typing import Any

from pydantic import ValidationError

from app.inference.protocols import GeneratorMetadata
from app.llm.generation_policy import (
    MAX_PROMPT_CHARS,
    GenerationTimeoutError,
    bound_prompt,
    call_with_policy,
    log_fallback,
)
from app.llm.payloads import (
    DEFAULT_DISCLAIMER,
    GENERAL_DISCLAIMER,
    ChatTurnPayload,
    ExplanationPayload,
    _bounded_chat_transcript,
    fallback_chat_turn,
    fallback_explanation,
)
from app.llm.prompts import chat_prompt_prefix, explanation_prompt
from app.llm.rotation import RotatingGeminiClient

logger = logging.getLogger(__name__)

genai: Any
try:
    from google import genai
except ImportError:  # pragma: no cover - runtime guard for missing dependency
    genai = None


__all__ = [
    "DEFAULT_DISCLAIMER",
    "GENERAL_DISCLAIMER",
    "GeminiService",
]


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

        prompt = explanation_prompt(user_text, predicted_condition)
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
        prompt_prefix = chat_prompt_prefix(
            confidence_guidance=confidence_guidance,
            pet_context=pet_context,
            predicted_condition=predicted_condition,
            confidence=confidence,
            prior_summary_text=prior_summary_text,
        )
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
