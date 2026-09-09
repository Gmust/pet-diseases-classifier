"""
Structural typing contracts for replaceable runtime services.

`Classifier` is satisfied by both `Predictor` (torch) and `OnnxPredictor`
(onnxruntime) without either inheriting from a shared base class — they only
need to match the shape. Adapters publish `.metadata` so callers (readiness
endpoint, evaluation scripts) read public data instead of private attributes
such as `predictor._id2label`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.inference.predictor import PredictionResult


@dataclass(frozen=True)
class ClassifierMetadata:
    backend: str  # "torch" | "onnx"
    model_path: str
    labels: tuple[str, ...]
    model_version: str = "unversioned-local"


@runtime_checkable
class Classifier(Protocol):
    @property
    def metadata(self) -> ClassifierMetadata: ...

    def predict(self, text: str) -> PredictionResult: ...

    def predict_top_k(self, text: str, k: int = 3) -> list[PredictionResult]: ...

    def predict_top_k_batch(
        self, texts: list[str], k: int = 3, batch_size: int = 32
    ) -> list[list[PredictionResult]]: ...


@dataclass(frozen=True)
class GeneratorMetadata:
    backend: str  # "gemini" | "fallback" (no API key / SDK configured)
    model_name: str
    available: bool


@runtime_checkable
class TextGenerator(Protocol):
    """Any adapter that generates content and reports whether it is live."""

    @property
    def metadata(self) -> GeneratorMetadata: ...
