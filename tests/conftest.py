"""
Shared test fixtures.

The suite runs fully offline:
- The classifier is replaced with a controllable FakePredictor (no torch / no
  weights needed — torch is lazy-imported, and we patch Predictor.from_paths
  before the app's lifespan runs).
- Gemini is left unconfigured (no API key), so the services use their
  deterministic local fallbacks.
"""
from __future__ import annotations

import os

import pytest

# Disable auth + Gemini before the app is imported anywhere.
os.environ.pop("API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

from app.ml.predictor import PredictionResult  # noqa: E402  (after env setup)


class FakePredictor:
    """Stand-in for the transformer. Tweak attributes per-test to steer outputs."""

    def __init__(self) -> None:
        self.condition = "Digestive Issues"
        self.confidence = 0.82
        self.runners_up = [
            ("Infectious and Parasitic Diseases", 0.09),
            ("Skin Conditions", 0.04),
        ]
        self.last_input: str | None = None

    def predict(self, text: str) -> PredictionResult:
        if not text.strip():
            raise ValueError("Text input cannot be empty.")
        self.last_input = text
        return PredictionResult(self.condition, self.confidence)

    def predict_top_k(self, text: str, k: int = 3) -> list[PredictionResult]:
        if not text.strip():
            raise ValueError("Text input cannot be empty.")
        self.last_input = text
        items = [PredictionResult(self.condition, self.confidence)]
        items += [PredictionResult(c, p) for c, p in self.runners_up]
        return items[:k]


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def client(monkeypatch, fake_predictor):
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Patch model loading so the lifespan uses the fake (no torch, no weights).
    monkeypatch.setattr(
        main.Predictor, "from_paths", classmethod(lambda cls, **kw: fake_predictor)
    )
    # ensure_services() caches on app.state and is idempotent, so reset it per test
    # to force a rebuild with THIS test's fake_predictor.
    main.app.state.services = None

    with TestClient(main.app) as test_client:
        yield test_client
