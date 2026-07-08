"""Application use case for POST /wellness. Thin wrapper: `WellnessService`
already owns the scoring/narrative orchestration and handles its own
classifier failures internally (see `_score_symptoms`), so there is no
transport-facing error translation needed here."""

from __future__ import annotations

from app.app_services import AppServices
from app.schemas import WellnessRequest, WellnessResponse


def run_wellness(payload: WellnessRequest, services: AppServices) -> WellnessResponse:
    return services.wellness_service.score(request=payload, predictor=services.predictor)
