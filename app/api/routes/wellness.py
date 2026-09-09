"""`POST /wellness` — rule-based wellness indicator."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import api_key_auth, get_services
from app.app_services import AppServices
from app.wellness.entrypoint import run_wellness
from app.wellness.responses import WellnessResponse
from app.wellness.schemas import WellnessRequest

router = APIRouter()


@router.post("/wellness", response_model=WellnessResponse, dependencies=[Depends(api_key_auth)])
def wellness(
    payload: WellnessRequest,
    services: AppServices = Depends(get_services),
) -> WellnessResponse:
    """
    Pet wellness indicator (0-100) derived from tracked activity, feeding, and care data.

    - Score is rule-based across 6 dimensions; Gemini generates the narrative and recommendations.
    - Active chronic conditions cap the maximum possible score.
    - If currentSymptoms is provided, it is passed through the classifier to influence the score.
    - Missing dimensions are scaled out and reported through scoreStatus and dataCoverage.
    - Data below the reliability gate returns nullable score fields plus deterministic trackingRecommendations.
    - reminders use the C# backend ReminderType wire values with actionable text.
    """
    return run_wellness(payload, services)
