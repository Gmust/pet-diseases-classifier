"""
Classifier regression guard.

A small fixed set of labeled symptom strings with a minimum accuracy threshold.
This is the test that lets you retrain with confidence: if a new model regresses
on these basic cases, CI fails.

It is SKIPPED automatically when the real model weights or the ML runtime
(torch/transformers) aren't present — so the rest of the suite stays fast and
runs anywhere. To exercise it, point MODEL_PATH at a trained model directory.

Tune EXPECTED_MIN_ACCURACY to your model's real baseline once you've run it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

MODEL_PATH = os.getenv("MODEL_PATH", "models/transformer_model")

# Each: (symptom text, set of acceptable condition labels)
GOLDEN_CASES: list[tuple[str, set[str]]] = [
    ("My dog has been vomiting and has diarrhea for two days", {"Digestive Issues"}),
    ("My cat keeps scratching her ears and shaking her head", {"Ear Conditions"}),
    ("There is a red itchy rash and hair loss on my dog's belly", {"Skin Conditions"}),
    ("My dog is limping and seems stiff getting up", {"Musculoskeletal Conditions"}),
    (
        "My cat is drinking lots of water and urinating frequently",
        {"Metabolic and Endocrine Disorders", "Genitourinary Conditions"},
    ),
    ("My dog has a cough and is breathing fast", {"Respiratory Conditions"}),
    ("My pet's eye is red, swollen and weeping", {"Eye Conditions"}),
    ("I found fleas and worms on my puppy", {"Infectious and Parasitic Diseases"}),
]

EXPECTED_MIN_ACCURACY = 0.75


def _weights_present(model_dir: str) -> bool:
    p = Path(model_dir)
    if not p.exists():
        return False
    # `p.glob(...) or p.glob(...)` is a bug: a generator object is always
    # truthy, so `or` always short-circuits to the first glob and the second
    # pattern is never checked (silently skipping this test for a
    # pytorch_model.bin-only model directory). Check each pattern explicitly.
    return any(p.glob("*.safetensors")) or any(p.glob("pytorch_model.bin"))


pytestmark = pytest.mark.skipif(
    not _weights_present(MODEL_PATH),
    reason=f"No model weights at {MODEL_PATH!r}; skipping regression test.",
)


@pytest.fixture(scope="module")
def predictor():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from app.inference.predictor import Predictor

    return Predictor.from_paths(model_path=MODEL_PATH)


def test_classifier_meets_accuracy_floor(predictor):
    correct = 0
    misses: list[str] = []
    for text, acceptable in GOLDEN_CASES:
        pred = predictor.predict(text).predicted_condition
        if pred in acceptable:
            correct += 1
        else:
            misses.append(f"{text!r} → {pred} (expected one of {sorted(acceptable)})")

    accuracy = correct / len(GOLDEN_CASES)
    assert (
        accuracy >= EXPECTED_MIN_ACCURACY
    ), f"Regression: accuracy {accuracy:.0%} < {EXPECTED_MIN_ACCURACY:.0%}.\n" + "\n".join(misses)
