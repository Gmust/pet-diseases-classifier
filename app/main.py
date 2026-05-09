import os
import hmac
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.ml.predictor import Predictor
from app.ml.condition_metadata import get_condition_metadata
from app.schemas import AskRequest, AskResponse, PredictRequest, PredictResponse, WellnessRequest, WellnessResponse
from app.services.gemini_service import DEFAULT_DISCLAIMER, GeminiService
from app.services.ask_service import AskService
from app.services.wellness_service import WellnessService

load_dotenv()


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
    ask_service: AskService
    wellness_service: WellnessService
    low_confidence_threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "models/transformer_model")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    low_confidence_threshold = _parse_threshold(os.getenv("LOW_CONFIDENCE_THRESHOLD"), default=0.65)

    predictor = Predictor.from_paths(model_path=model_path)
    gemini_service = GeminiService(api_key=gemini_api_key, model_name=gemini_model)
    ask_service = AskService(api_key=gemini_api_key, model_name=gemini_model)
    wellness_service = WellnessService(api_key=gemini_api_key, model_name=gemini_model)

    app.state.services = AppServices(
        predictor=predictor,
        gemini_service=gemini_service,
        ask_service=ask_service,
        wellness_service=wellness_service,
        low_confidence_threshold=low_confidence_threshold,
    )
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

    explanation_payload = services.gemini_service.generate_explanation(
        user_text=payload.text,
        predicted_condition=prediction.predicted_condition,
        default_home_advice=list(meta.home_advice),
    )

    explanation = explanation_payload.explanation
    if prediction.confidence < services.low_confidence_threshold and LOW_CONFIDENCE_NOTE not in explanation:
        explanation = f"{explanation} {LOW_CONFIDENCE_NOTE}"

    return PredictResponse(
        predicted_condition=prediction.predicted_condition,
        confidence=round(prediction.confidence, 4),
        explanation=explanation,
        disclaimer=explanation_payload.disclaimer or DEFAULT_DISCLAIMER,
        urgency=meta.urgency,
        specialist=meta.specialist,
        disease_category=meta.disease_category,
        home_advice=explanation_payload.home_advice,
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


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(api_key_auth)])
def ask(payload: AskRequest) -> AskResponse:
    """
    General pet-care Q&A: breeds, diet, grooming, training, behaviour, housing.

    - Supported species: dog, cat, rabbit, hamster, guinea_pig, bird, fish, turtle.
    - Medical / symptom questions are redirected to POST /predict.
    - Unsupported species receive a \"not covered\" response without consuming Gemini quota.
    """
    services: AppServices = app.state.services
    return services.ask_service.answer(
        question=payload.question,
        pet_type=payload.pet_type,
    )
