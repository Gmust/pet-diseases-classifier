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
# GEMINI_API_KEYS (plural) must be cleared too: Settings.resolved_gemini_api_keys()
# merges both vars, so leaving the plural one set builds a real rotating client
# from a developer's local .env and the suite silently starts calling the API.
GEMINI_ENV_VARS = ("GEMINI_API_KEY", "GEMINI_API_KEYS")

for _var in ("API_KEY", *GEMINI_ENV_VARS):
    os.environ.pop(_var, None)

# The suite fakes only the torch predictor, so pin the backend: with
# MODEL_BACKEND=onnx exported (or in a local .env) build_services() would reach
# the real ONNX runtime and every API test would fail in setup.
os.environ["MODEL_BACKEND"] = "torch"

from app.inference.predictor import PredictionResult  # noqa: E402  (after env setup)
from app.inference.protocols import ClassifierMetadata  # noqa: E402  (after env setup)


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

    @property
    def metadata(self) -> ClassifierMetadata:
        return ClassifierMetadata(
            backend="fake",
            model_path="",
            labels=("Digestive Issues", self.condition),
            model_version="test-model-v1",
        )

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

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[PredictionResult]:
        return [self.predict(text) for text in texts]

    def predict_top_k_batch(
        self, texts: list[str], k: int = 3, batch_size: int = 32
    ) -> list[list[PredictionResult]]:
        return [self.predict_top_k(text, k=k) for text in texts]


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def client(monkeypatch, fake_predictor):
    from fastapi.testclient import TestClient

    from app import main
    from app.inference.predictor import Predictor

    # app.main calls load_dotenv() at import, which repopulates anything cleared
    # above from a local .env, so clear it again now that the import has happened.
    monkeypatch.delenv("API_KEY", raising=False)
    for var in GEMINI_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MODEL_BACKEND", "torch")
    # Patch model loading so the lifespan uses the fake (no torch, no weights).
    # Patched on the class itself, so it applies wherever app.bootstrap resolves it.
    monkeypatch.setattr(Predictor, "from_paths", classmethod(lambda cls, **kw: fake_predictor))
    # ensure_services() caches on app.state and is idempotent, so reset it per test
    # to force a rebuild with THIS test's fake_predictor.
    main.app.state.services = None

    with TestClient(main.app) as test_client:
        yield test_client
