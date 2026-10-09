"""
Runtime service container.

Split out from `app.main` so application use cases (`app.triage.*`, `app.wellness.*`, `app.feeding.*`) can
type-hint against it without importing the FastAPI app module and creating a
circular import.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.inference.protocols import Classifier
from app.llm.gemini_service import GeminiService
from app.wellness.service import WellnessService


@dataclass
class AppServices:
    predictor: Classifier
    gemini_service: GeminiService
    wellness_service: WellnessService
    low_confidence_threshold: float
    use_static_explanations: bool = False
