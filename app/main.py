import hmac
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from app.app_services import AppServices
from app.config import Settings, get_settings
from app.ml.predictor import Predictor
from app.ml.protocols import Classifier
from app.observability import RequestLoggingMiddleware, configure_logging, log_event
from app.schemas import (
    ChatRequest,
    ChatResponse,
    FeedingSummaryRequest,
    FeedingSummaryResponse,
    PredictRequest,
    PredictResponse,
    WellnessRequest,
    WellnessResponse,
)
from app.services.gemini_service import GeminiService
from app.services.wellness_service import WellnessService
from app.use_cases.chat import run_chat
from app.use_cases.errors import InferenceUnavailableError, InvalidInputError
from app.use_cases.feeding_summary import run_feeding_summary
from app.use_cases.predict import run_predict
from app.use_cases.wellness import run_wellness

load_dotenv()
configure_logging(get_settings().log_level)

logger = logging.getLogger(__name__)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def api_key_auth(api_key: str | None = Security(api_key_header)) -> None:
    expected_api_key = get_settings().api_key
    if not expected_api_key:
        return
    if not api_key or not hmac.compare_digest(api_key, expected_api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")


def build_services(settings: Settings | None = None) -> AppServices:
    """Load the model + services. Extracted from the lifespan so it can also be
    called at Lambda INIT (and by the keep-warm ping) — that way a warm container
    has the model already loaded, instead of paying the load on the first real request."""
    settings = settings or get_settings()

    # Backend selection: torch (default) or quantized ONNX (cheaper on Lambda).
    predictor: Classifier
    if settings.model_backend == "onnx":
        from app.ml.onnx_predictor import OnnxPredictor

        predictor = OnnxPredictor.from_paths(model_path=settings.model_path)
    else:
        predictor = Predictor.from_paths(model_path=settings.model_path)

    gemini_api_keys = settings.resolved_gemini_api_keys()
    services = AppServices(
        predictor=predictor,
        gemini_service=GeminiService(api_keys=gemini_api_keys, model_name=settings.gemini_model),
        wellness_service=WellnessService(
            api_keys=gemini_api_keys, model_name=settings.gemini_model
        ),
        low_confidence_threshold=settings.low_confidence_threshold,
        # When true, /predict serves a cautious templated explanation instead of
        # calling Gemini — zero per-request API cost. See README / cost notes.
        use_static_explanations=settings.use_static_explanations,
    )
    metadata = services.predictor.metadata
    log_event(
        "model_loaded",
        backend=metadata.backend,
        model_version=metadata.model_version,
        label_count=len(metadata.labels),
    )
    return services


def ensure_services() -> AppServices:
    """Idempotently load services onto app.state (safe to call multiple times)."""
    if getattr(app.state, "services", None) is None:
        app.state.services = build_services()
    return app.state.services


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ensure_services()  # no-op if already loaded at INIT (Lambda) — loads on first run (uvicorn)
    yield


app = FastAPI(
    title="Pet Care AI Microservice",
    description="Classifier-based pet condition pre-assessment with Gemini-generated explanation.",
    version="1.0.0",
    lifespan=lifespan,
    # ROOT_PATH tells FastAPI it is mounted behind a proxy at this prefix.
    # Set to "/Prod" on Lambda (API Gateway stage), leave empty for local dev.
    root_path=get_settings().root_path,
)
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    # Preserve FastAPI's default `{"detail": ...}` shape (existing clients rely on
    # it) and attach the request id from RequestLoggingMiddleware so a caller can
    # correlate a public error with the matching server-side log line.
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "requestId": request_id},
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Safety net for any exception that escapes a route's own try/except (each
    # endpoint already converts expected classifier/generator failures to a
    # stable HTTPException — this only fires for genuinely unexpected errors).
    # Never expose exc's text to the caller; log it server-side instead.
    request_id = getattr(request.state, "request_id", None)
    logger.exception(
        "Unhandled exception",
        extra={
            "fields": {
                "event": "unhandled_exception",
                "path": request.url.path,
                "request_id": request_id,
            }
        },
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "requestId": request_id},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/live")
def liveness() -> dict[str, str]:
    """Process is up and able to serve HTTP. Does not check the model."""
    return {"status": "ok"}


@app.get("/health/ready")
def readiness(response: Response) -> dict[str, object]:
    """Model bundle is loaded and validated. No filesystem paths or secrets exposed."""
    services = getattr(app.state, "services", None)
    if services is None:
        response.status_code = 503
        return {"status": "not_ready", "reason": "model_not_loaded"}
    meta = services.predictor.metadata
    return {
        "status": "ready",
        "backend": meta.backend,
        "modelVersion": meta.model_version,
        "labelCount": len(meta.labels),
    }


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(api_key_auth)])
def predict(payload: PredictRequest) -> PredictResponse:
    services: AppServices = app.state.services
    try:
        return run_predict(payload, services)
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InferenceUnavailableError as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(api_key_auth)])
def chat(payload: ChatRequest) -> ChatResponse:
    """
    Unified stateless conversational endpoint. Self-routes per turn:

      - mode="general"   → general pet-care Q&A; `prediction` is null, `relatedTopics` set.
      - mode="health"    → symptom triage; `prediction` populated, rolling summary updated.
      - mode="emergency" → red-flag detected; emergency message + first-aid advice.

    The backend owns all session state: it stores the message history and the rolling
    `symptomSummary` and replays them every turn. This service stores nothing.
    """
    services: AppServices = app.state.services
    try:
        return run_chat(payload, services)
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InferenceUnavailableError as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


@app.post("/wellness", response_model=WellnessResponse, dependencies=[Depends(api_key_auth)])
def wellness(payload: WellnessRequest) -> WellnessResponse:
    """
    Pet wellness indicator (0-100) derived from tracked activity, feeding, and care data.

    - Score is rule-based across 6 dimensions; Gemini generates the narrative and recommendations.
    - Active chronic conditions cap the maximum possible score.
    - If currentSymptoms is provided, it is passed through the classifier to influence the score.
    - Missing dimensions are scaled out and reported through scoreStatus and dataCoverage.
    - Data below the reliability gate returns nullable score fields plus deterministic trackingRecommendations.
    - reminders use the C# backend ReminderType wire values with actionable text.
    """
    services: AppServices = app.state.services
    return run_wellness(payload, services)


@app.post(
    "/feeding-summary",
    response_model=FeedingSummaryResponse,
    dependencies=[Depends(api_key_auth)],
)
def feeding_summary(payload: FeedingSummaryRequest) -> FeedingSummaryResponse:
    """
    Daily per-pet feeding summary, meant to be called once/day (batched across all
    pets in one request) by a scheduler in the backend, to drive a feeding notification.

    - Fully deterministic (RER/MER calorie-target formula) — no classifier, no Gemini,
      no per-pet API cost, so it stays cheap at any batch size.
    - Each pet's logged products are summed and compared against its computed target;
      `status` is a 5-tier enum (EXTREME_UNDER_TARGET..EXTREME_OVER_TARGET) with no
      baked-in text — the frontend renders the notification copy per-locale from
      `status` + `targetCalories`/`actualCalories`/`deviationPct`.
    """
    return run_feeding_summary(payload)
