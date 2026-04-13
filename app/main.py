import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.ml.predictor import Predictor
from app.schemas import PredictRequest, PredictResponse
from app.services.gemini_service import DEFAULT_DISCLAIMER, GeminiService

load_dotenv()


LOW_CONFIDENCE_NOTE = (
    "Model confidence is limited for this prediction. Monitor your pet closely and seek veterinary advice."
)


def _parse_threshold(raw_value: str | None, default: float = 0.65) -> float:
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return min(max(value, 0.0), 1.0)


@dataclass
class AppServices:
    predictor: Predictor
    gemini_service: GeminiService
    low_confidence_threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "models/transformer_model")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    low_confidence_threshold = _parse_threshold(os.getenv("LOW_CONFIDENCE_THRESHOLD"), default=0.65)

    predictor = Predictor.from_paths(model_path=model_path)
    gemini_service = GeminiService(api_key=gemini_api_key, model_name=gemini_model)
    app.state.services = AppServices(
        predictor=predictor,
        gemini_service=gemini_service,
        low_confidence_threshold=low_confidence_threshold,
    )
    yield


app = FastAPI(
    title="Pet Care AI Microservice",
    description="Classifier-based pet condition pre-assessment with Gemini-generated explanation.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    services: AppServices = app.state.services

    try:
        prediction = services.predictor.predict(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    explanation_payload = services.gemini_service.generate_explanation(
        user_text=payload.text,
        predicted_condition=prediction.predicted_condition,
    )

    explanation = explanation_payload.explanation
    if prediction.confidence < services.low_confidence_threshold and LOW_CONFIDENCE_NOTE not in explanation:
        explanation = f"{explanation} {LOW_CONFIDENCE_NOTE}"

    return PredictResponse(
        predicted_condition=prediction.predicted_condition,
        confidence=round(prediction.confidence, 4),
        explanation=explanation,
        disclaimer=explanation_payload.disclaimer or DEFAULT_DISCLAIMER,
    )
