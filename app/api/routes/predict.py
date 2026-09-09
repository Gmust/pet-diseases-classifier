"""`POST /predict` — single-shot symptom classification."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import api_key_auth, get_services
from app.app_services import AppServices
from app.triage.predict import run_predict
from app.triage.schemas import PredictRequest, PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(api_key_auth)])
def predict(
    payload: PredictRequest,
    services: AppServices = Depends(get_services),
) -> PredictResponse:
    return run_predict(payload, services)
