"""`POST /chat` — unified stateless triage + general Q&A."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import api_key_auth, get_services
from app.app_services import AppServices
from app.triage.chat import run_chat
from app.triage.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(api_key_auth)])
def chat(
    payload: ChatRequest,
    services: AppServices = Depends(get_services),
) -> ChatResponse:
    """
    Unified stateless conversational endpoint. Self-routes per turn:

      - mode="general"   → general pet-care Q&A; `prediction` is null, `relatedTopics` set.
      - mode="health"    → symptom triage; `prediction` populated, rolling summary updated.
      - mode="emergency" → red-flag detected; emergency message + first-aid advice.

    The backend owns all session state: it stores the message history and the rolling
    `symptomSummary` and replays them every turn. This service stores nothing.
    """
    return run_chat(payload, services)
