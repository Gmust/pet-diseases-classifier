import os
import hmac
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.ml.predictor import Predictor
from app.ml.condition_metadata import build_static_explanation, get_condition_metadata
from app.observability import RequestLoggingMiddleware, configure_logging, log_event
from app.schemas import (
    ChatMode,
    ChatPrediction,
    ChatRequest,
    ChatResponse,
    PredictRequest,
    PredictResponse,
    TopKItem,
    WellnessRequest,
    WellnessResponse,
)
from app.services.chat_context import build_classifier_input, latest_user_message
from app.services.gemini_service import DEFAULT_DISCLAIMER, GENERAL_DISCLAIMER, GeminiService
from app.services.triage_safety import (
    EMERGENCY_HOME_ADVICE,
    apply_red_flag_urgency,
    emergency_explanation,
    should_abstain,
)
from app.services.wellness_service import WellnessService

load_dotenv()
configure_logging(os.getenv("LOG_LEVEL", "INFO"))


LOW_CONFIDENCE_NOTE = (
    "Model confidence is limited for this prediction. Monitor your pet closely and seek veterinary advice."
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _parse_threshold(raw_value: str | None, default: float = 0.65) -> float:
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return min(max(value, 0.0), 1.0)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_chat_prediction(top_predictions, meta, urgency, home_advice) -> ChatPrediction:
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


def api_key_auth(api_key: str | None = Security(api_key_header)) -> None:
    expected_api_key = os.getenv("API_KEY")
    if not expected_api_key:
        return
    if not api_key or not hmac.compare_digest(api_key, expected_api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")


@dataclass
class AppServices:
    predictor: Predictor
    gemini_service: GeminiService
    wellness_service: WellnessService
    low_confidence_threshold: float
    use_static_explanations: bool = False


def build_services() -> AppServices:
    """Load the model + services. Extracted from the lifespan so it can also be
    called at Lambda INIT (and by the keep-warm ping) — that way a warm container
    has the model already loaded, instead of paying the load on the first real request."""
    model_path = os.getenv("MODEL_PATH", "models/transformer_model")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    low_confidence_threshold = _parse_threshold(os.getenv("LOW_CONFIDENCE_THRESHOLD"), default=0.65)

    # Backend selection: torch (default) or quantized ONNX (cheaper on Lambda).
    backend = os.getenv("MODEL_BACKEND", "torch").strip().lower()
    if backend == "onnx":
        from app.ml.onnx_predictor import OnnxPredictor

        predictor = OnnxPredictor.from_paths(model_path=model_path)
    else:
        predictor = Predictor.from_paths(model_path=model_path)

    return AppServices(
        predictor=predictor,
        gemini_service=GeminiService(api_key=gemini_api_key, model_name=gemini_model),
        wellness_service=WellnessService(api_key=gemini_api_key, model_name=gemini_model),
        low_confidence_threshold=low_confidence_threshold,
        # When true, /predict serves a cautious templated explanation instead of
        # calling Gemini — zero per-request API cost. See README / cost notes.
        use_static_explanations=_env_flag("USE_STATIC_EXPLANATIONS", default=False),
    )


def ensure_services() -> AppServices:
    """Idempotently load services onto app.state (safe to call multiple times)."""
    if getattr(app.state, "services", None) is None:
        app.state.services = build_services()
    return app.state.services


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_services()  # no-op if already loaded at INIT (Lambda) — loads on first run (uvicorn)
    yield


app = FastAPI(
    title="Pet Care AI Microservice",
    description="Classifier-based pet condition pre-assessment with Gemini-generated explanation.",
    version="1.0.0",
    lifespan=lifespan,
    # ROOT_PATH tells FastAPI it is mounted behind a proxy at this prefix.
    # Set to "/Prod" on Lambda (API Gateway stage), leave empty for local dev.
    root_path=os.getenv("ROOT_PATH", ""),
)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(api_key_auth)])
def predict(payload: PredictRequest) -> PredictResponse:
    services: AppServices = app.state.services

    try:
        prediction = services.predictor.predict(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    meta = get_condition_metadata(prediction.predicted_condition)
    urgency, red_flag = apply_red_flag_urgency(payload.text, meta.urgency)

    if red_flag.triggered:
        # Emergency: override condition-specific content (the predicted class may be
        # wrong) and skip Gemini — lead with emergency message + first-aid advice.
        explanation = emergency_explanation(red_flag.reason)
        disclaimer = DEFAULT_DISCLAIMER
        home_advice = list(EMERGENCY_HOME_ADVICE)
        gemini_used = False
    elif services.use_static_explanations:
        # Cost path: skip Gemini and serve a cautious templated explanation.
        explanation = build_static_explanation(prediction.predicted_condition, meta)
        disclaimer = DEFAULT_DISCLAIMER
        home_advice = list(meta.home_advice)
        gemini_used = False
    else:
        explanation_payload = services.gemini_service.generate_explanation(
            user_text=payload.text,
            predicted_condition=prediction.predicted_condition,
            default_home_advice=list(meta.home_advice),
        )
        explanation = explanation_payload.explanation
        disclaimer = explanation_payload.disclaimer or DEFAULT_DISCLAIMER
        home_advice = explanation_payload.home_advice
        gemini_used = True

    # Low-confidence note — only when this is NOT an emergency override.
    if (
        not red_flag.triggered
        and prediction.confidence < services.low_confidence_threshold
        and LOW_CONFIDENCE_NOTE not in explanation
    ):
        explanation = f"{explanation} {LOW_CONFIDENCE_NOTE}"

    log_event(
        "prediction",
        endpoint="predict",
        condition=prediction.predicted_condition,
        confidence=round(prediction.confidence, 4),
        low_confidence=prediction.confidence < services.low_confidence_threshold,
        red_flag=red_flag.reason,
        gemini_used=gemini_used,
    )

    return PredictResponse(
        predicted_condition=prediction.predicted_condition,
        confidence=round(prediction.confidence, 4),
        explanation=explanation,
        disclaimer=disclaimer,
        urgency=urgency,
        specialist=meta.specialist,
        disease_category=meta.disease_category,
        home_advice=home_advice,
    )


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(api_key_auth)])
def chat(payload: ChatRequest) -> ChatResponse:
    """
    Unified stateless conversational endpoint. Self-routes per turn:

      - mode="general"   → general pet-care Q&A; `prediction` is null, `relatedTopics` set.
      - mode="health"    → symptom triage; `prediction` populated, rolling summary updated.
      - mode="emergency" → red-flag detected; emergency message + first-aid advice.

    The backend owns all session state: it stores the message history and the rolling
    `symptomSummary` and replays them every turn. This service stores nothing.

    Flow: red-flag check (always) → local classifier → one Gemini call that decides
    general vs health AND writes the answer. Branch the response on `mode`.
    """
    services: AppServices = app.state.services

    new_message = latest_user_message(payload.messages)
    if not new_message or not new_message.strip():
        raise HTTPException(status_code=400, detail="The last message must be a non-empty user message.")

    classifier_input = build_classifier_input(payload.symptom_summary, new_message)

    try:
        top_predictions = services.predictor.predict_top_k(classifier_input, k=3)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    top = top_predictions[0]
    meta = get_condition_metadata(top.predicted_condition)
    low_confidence = top.confidence < services.low_confidence_threshold
    abstain = should_abstain(top.confidence)
    urgency, red_flag = apply_red_flag_urgency(new_message, meta.urgency)

    # --- Emergency: red-flag always wins, regardless of general/health intent ---
    if red_flag.triggered:
        summary_bits = [
            s for s in [(payload.symptom_summary or "").strip(), new_message.strip()] if s
        ]
        log_event("prediction", endpoint="chat", mode="emergency",
                   condition=top.predicted_condition, confidence=round(top.confidence, 4),
                   red_flag=red_flag.reason, gemini_used=False)
        return ChatResponse(
            mode=ChatMode.EMERGENCY,
            answer=emergency_explanation(red_flag.reason),
            symptom_summary=" ".join(summary_bits)[:1000],
            prediction=_build_chat_prediction(top_predictions, meta, urgency, list(EMERGENCY_HOME_ADVICE)),
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
    log_event("prediction", endpoint="chat", mode="health",
               condition=top.predicted_condition, confidence=round(top.confidence, 4),
               low_confidence=low_confidence, abstain=abstain, red_flag=None, gemini_used=gemini_used)
    return ChatResponse(
        mode=ChatMode.HEALTH,
        answer=turn.answer,
        symptom_summary=turn.symptom_summary,
        prediction=_build_chat_prediction(top_predictions, meta, urgency, list(meta.home_advice)),
        related_topics=[],
        needs_clarification=turn.needs_clarification or abstain,
        disclaimer=DEFAULT_DISCLAIMER,
    )


@app.post("/wellness", response_model=WellnessResponse, dependencies=[Depends(api_key_auth)])
def wellness(payload: WellnessRequest) -> WellnessResponse:
    """
    Pet wellness indicator (0-100) derived from tracked activity, feeding, and care data.

    - Score is rule-based across 6 dimensions; Gemini generates the narrative and recommendations.
    - Active chronic conditions cap the maximum possible score.
    - If currentSymptoms is provided, it is passed through the classifier to influence the score.
    - Missing dimensions are scaled out — partial data is always accepted.
    """
    services: AppServices = app.state.services
    return services.wellness_service.score(
        request=payload,
        predictor=services.predictor,
    )
