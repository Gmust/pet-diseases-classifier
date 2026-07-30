"""Application use case for POST /feeding-summary. Thin wrapper: the service
is pure/deterministic (no classifier, no Gemini), so there is no transport-facing
error translation needed here."""

from __future__ import annotations

from app.schemas import FeedingSummaryRequest, FeedingSummaryResponse
from app.services.feeding_summary_service import summarize


def run_feeding_summary(payload: FeedingSummaryRequest) -> FeedingSummaryResponse:
    return summarize(payload)
