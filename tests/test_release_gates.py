"""Unit tests for deterministic safety and Torch/ONNX parity release gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ml.predictor import PredictionResult
from app.ml.release_gates import (
    ReleaseGateError,
    evaluate_backend_parity,
    evaluate_safety_cases,
)


def test_tracked_safety_fixture_passes_release_gate():
    fixture = json.loads(Path("data/safety_eval.json").read_text(encoding="utf-8"))

    report = evaluate_safety_cases(
        fixture["cases"],
        min_recall=1.0,
        max_false_positive_rate=0.05,
    )

    assert report["passed"] is True
    assert report["metrics"]["recall"] == 1.0
    assert report["metrics"]["false_positive_rate"] == 0.0
    assert len(report["metrics"]["reasons_covered"]) == 11


def test_safety_gate_reports_fixture_and_threshold_failures():
    cases = [
        {"id": "miss", "text": "ordinary text", "expected_triggered": True},
        {"id": "negative", "text": "mild itching", "expected_triggered": False},
    ]

    report = evaluate_safety_cases(cases, min_recall=1.0, max_false_positive_rate=0.0)

    assert report["passed"] is False
    assert report["metrics"]["false_negative"] == 1
    assert report["fixture_failures"][0]["id"] == "miss"


def test_backend_parity_passes_with_matching_predictions():
    torch_predictions = [
        PredictionResult("Digestive Issues", 0.80),
        PredictionResult("Skin Conditions", 0.70),
    ]
    onnx_predictions = [
        PredictionResult("Digestive Issues", 0.79),
        PredictionResult("Skin Conditions", 0.69),
    ]

    report = evaluate_backend_parity(
        y_true=["Digestive Issues", "Skin Conditions"],
        torch_predictions=torch_predictions,
        onnx_predictions=onnx_predictions,
        min_label_agreement=1.0,
        max_accuracy_delta=0.0,
        max_mean_confidence_delta=0.02,
    )

    assert report["passed"] is True
    assert report["metrics"]["label_agreement"] == 1.0
    assert report["metrics"]["mean_confidence_delta"] == pytest.approx(0.01)


def test_backend_parity_fails_on_label_and_accuracy_regression():
    report = evaluate_backend_parity(
        y_true=["Digestive Issues", "Skin Conditions"],
        torch_predictions=[
            PredictionResult("Digestive Issues", 0.8),
            PredictionResult("Skin Conditions", 0.7),
        ],
        onnx_predictions=[
            PredictionResult("Skin Conditions", 0.8),
            PredictionResult("Skin Conditions", 0.7),
        ],
        min_label_agreement=1.0,
        max_accuracy_delta=0.0,
        max_mean_confidence_delta=0.1,
    )

    assert report["passed"] is False
    assert report["metrics"]["label_agreement"] == 0.5
    assert report["metrics"]["accuracy_delta"] == 0.5
    assert len(report["gate_failures"]) == 2


def test_backend_parity_rejects_mismatched_input_lengths():
    with pytest.raises(ReleaseGateError, match="equally sized"):
        evaluate_backend_parity(
            y_true=["Digestive Issues"],
            torch_predictions=[],
            onnx_predictions=[],
            min_label_agreement=1.0,
            max_accuracy_delta=0.0,
            max_mean_confidence_delta=0.0,
        )
