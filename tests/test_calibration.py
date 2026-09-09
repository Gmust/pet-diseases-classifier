"""Unit tests for confidence calibration / abstention threshold evidence."""

from __future__ import annotations

import json

import pytest

from ml_pipeline.calibration import (
    build_abstention_policy,
    compute_selective_accuracy_curve,
    find_threshold_for_target_accuracy,
    write_calibration_report,
)


def _sample():
    # High-confidence predictions are all correct; low-confidence ones are
    # a coin flip — a realistic shape for a selective-accuracy curve.
    confidences = [0.95, 0.92, 0.90, 0.55, 0.50, 0.45, 0.30, 0.20]
    correct = [True, True, True, True, False, False, False, True]
    return confidences, correct


def test_compute_selective_accuracy_curve_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_selective_accuracy_curve([0.5], [])


def test_compute_selective_accuracy_curve_rejects_empty_input():
    with pytest.raises(ValueError, match="zero predictions"):
        compute_selective_accuracy_curve([], [])


def test_compute_selective_accuracy_curve_at_threshold_zero_keeps_everything():
    confidences, correct = _sample()
    curve = compute_selective_accuracy_curve(confidences, correct, thresholds=[0.0])
    point = curve[0]
    assert point.kept == len(confidences)
    assert point.coverage == 1.0
    assert point.accuracy == sum(correct) / len(correct)


def test_compute_selective_accuracy_curve_higher_threshold_never_lowers_accuracy_here():
    confidences, correct = _sample()
    curve = compute_selective_accuracy_curve(confidences, correct, thresholds=[0.0, 0.9])
    low, high = curve
    assert high.coverage <= low.coverage
    assert high.accuracy >= low.accuracy  # in this fixture, high-confidence rows are clean


def test_find_threshold_for_target_accuracy_returns_lowest_qualifying_threshold():
    confidences, correct = _sample()
    curve = compute_selective_accuracy_curve(confidences, correct, thresholds=[0.0, 0.3, 0.5, 0.9])
    point = find_threshold_for_target_accuracy(curve, target_accuracy=1.0)
    assert point is not None
    assert point.threshold == 0.9
    assert point.accuracy == 1.0


def test_find_threshold_for_target_accuracy_returns_none_when_unreachable():
    confidences, correct = _sample()
    curve = compute_selective_accuracy_curve(confidences, correct, thresholds=[0.0])
    assert find_threshold_for_target_accuracy(curve, target_accuracy=2.0) is None


def test_build_abstention_policy_reports_evidence():
    confidences, correct = _sample()
    curve = compute_selective_accuracy_curve(confidences, correct, thresholds=[0.0, 0.9])
    policy = build_abstention_policy(curve, target_accuracy=1.0, model_commit="abc123")
    assert policy is not None
    assert policy.threshold == 0.9
    assert policy.model_commit == "abc123"
    assert policy.sample_size == len(confidences)


def test_build_abstention_policy_none_when_unreachable():
    confidences, correct = _sample()
    curve = compute_selective_accuracy_curve(confidences, correct, thresholds=[0.0])
    assert build_abstention_policy(curve, target_accuracy=2.0, model_commit=None) is None


def test_write_calibration_report_appends_history(tmp_path):
    confidences, correct = _sample()
    curve = compute_selective_accuracy_curve(confidences, correct, thresholds=[0.0, 0.9])
    policy = build_abstention_policy(curve, target_accuracy=1.0, model_commit="abc123")
    path = tmp_path / "calibration.json"

    write_calibration_report(path, curve, [policy])
    write_calibration_report(path, curve, [policy])

    history = json.loads(path.read_text())["history"]
    assert len(history) == 2  # each run appended, never overwritten
