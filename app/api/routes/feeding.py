"""`POST /feeding-summary` — deterministic daily feeding batch."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import api_key_auth
from app.feeding.schemas import FeedingSummaryRequest, FeedingSummaryResponse
from app.feeding.summary import run_feeding_summary

router = APIRouter()


@router.post(
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
