"""Application use case for POST /chat — the unified stateless conversational
endpoint. No FastAPI/HTTP concerns here; `app.main` maps `UseCaseError`
subclasses to HTTP responses.

Flow: red-flag check (always) → local classifier → one Gemini call that
decides general vs health AND writes the answer. Branch the response on
`mode`. The backend owns all session state; this service stores nothing.
"""

from __future__ import annotations

import logging

from app.app_services import AppServices
from app.domain.conditions import ConditionMetadata, get_condition_metadata
from app.domain.enums import UrgencyLevel
from app.errors import InferenceUnavailableError, InvalidInputError
from app.inference.predictor import PredictionResult
from app.llm.gemini_service import DEFAULT_DISCLAIMER, GENERAL_DISCLAIMER
from app.observability import log_event
from app.triage.context import build_classifier_input, build_safety_context
from app.triage.safety import (
    EMERGENCY_HOME_ADVICE,
    apply_red_flag_urgency,
    emergency_explanation,
    should_abstain,
)
from app.triage.schemas import (
    ChatMode,
    ChatPrediction,
    ChatRequest,
    ChatResponse,
    ChatRole,
    TopKItem,
)

logger = logging.getLogger(__name__)


def _build_chat_prediction(
    top_predictions: list[PredictionResult],
    meta: ConditionMetadata,
    urgency: UrgencyLevel,
    home_advice: list[str],
) -> ChatPrediction:
    top = top_predictions[0]
    return ChatPrediction(
        predicted_condition=top.predicted_condition,
        confidence=round(top.confidence, 4),
        top_k=[
            TopKItem(condition=p.predicted_condition, confidence=round(p.confidence, 4))
            for p in top_predictions
        ],
        urgency=urgency,
        specialist=meta.specialist,
        disease_category=meta.disease_category,
        home_advice=home_advice,
    )


def run_chat(payload: ChatRequest, services: AppServices) -> ChatResponse:
    final_message = payload.messages[-1]
    if final_message.role != ChatRole.USER or not final_message.content.strip():
        raise InvalidInputError("The last message must be a non-empty user message.")
    new_message = final_message.content

    classifier_input = build_classifier_input(payload.symptom_summary, new_message)

    try:
        top_predictions = services.predictor.predict_top_k(classifier_input, k=3)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "Prediction failed",
            extra={"fields": {"event": "prediction_error", "endpoint": "chat"}},
        )
        raise InferenceUnavailableError("chat") from exc

    top = top_predictions[0]
    meta = get_condition_metadata(top.predicted_condition)
    low_confidence = top.confidence < services.low_confidence_threshold
    abstain = should_abstain(top.confidence)
    safety_context = build_safety_context(payload.symptom_summary, new_message)
    urgency, red_flag = apply_red_flag_urgency(safety_context, meta.urgency)

    # --- Emergency: red-flag always wins, regardless of general/health intent ---
    if red_flag.triggered:
        summary_bits = [
            s for s in [(payload.symptom_summary or "").strip(), new_message.strip()] if s
        ]
        log_event(
            "prediction",
            endpoint="chat",
            mode="emergency",
            condition=top.predicted_condition,
            confidence=round(top.confidence, 4),
            red_flag=red_flag.reason,
            gemini_used=False,
        )
        return ChatResponse(
            mode=ChatMode.EMERGENCY,
            answer=emergency_explanation(red_flag.reason),
            symptom_summary=" ".join(summary_bits)[:1000],
            prediction=_build_chat_prediction(
                top_predictions, meta, urgency, list(EMERGENCY_HOME_ADVICE)
            ),
            related_topics=[],
            needs_clarification=False,
            disclaimer=DEFAULT_DISCLAIMER,
        )

    # --- One Gemini call decides general vs health AND writes the answer ---
    conversation = [{"role": m.role.value, "content": m.content} for m in payload.messages]
    turn = services.gemini_service.generate_chat_turn(
        conversation=conversation,
        predicted_condition=top.predicted_condition,
        confidence=top.confidence,
        prior_summary=payload.symptom_summary,
        low_confidence=low_confidence or abstain,
        pet_type=payload.pet_type.value if payload.pet_type else None,
    )
    gemini_used = services.gemini_service.client is not None

    if turn.mode == "general":
        # General-care Q&A: no clinical prediction, keep the medical summary untouched.
        log_event("prediction", endpoint="chat", mode="general", gemini_used=gemini_used)
        return ChatResponse(
            mode=ChatMode.GENERAL,
            answer=turn.answer,
            symptom_summary=turn.symptom_summary or (payload.symptom_summary or ""),
            prediction=None,
            related_topics=turn.related_topics,
            needs_clarification=False,
            disclaimer=GENERAL_DISCLAIMER,
        )

    # Health triage.
    log_event(
        "prediction",
        endpoint="chat",
        mode="health",
        condition=top.predicted_condition,
        confidence=round(top.confidence, 4),
        low_confidence=low_confidence,
        abstain=abstain,
        red_flag=None,
        gemini_used=gemini_used,
    )
    return ChatResponse(
        mode=ChatMode.HEALTH,
        answer=turn.answer,
        symptom_summary=turn.symptom_summary,
        prediction=_build_chat_prediction(top_predictions, meta, urgency, list(meta.home_advice)),
        related_topics=[],
        needs_clarification=turn.needs_clarification or abstain,
        disclaimer=DEFAULT_DISCLAIMER,
    )
