"""Unit tests for batched, machine-readable evaluation reports."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from app.ml.evaluation_report import (  # noqa: E402
    EvaluationReportError,
    build_evaluation_report,
    compare_with_baseline,
    read_evaluation_report,
    regression_failures,
    write_evaluation_report,
)
from app.ml.predictor import PredictionResult  # noqa: E402

LABELS = ["Digestive Issues", "Skin Conditions", "Ear Conditions"]


def _predictions():
    return [
        [
            PredictionResult("Digestive Issues", 0.8),
            PredictionResult("Skin Conditions", 0.15),
        ],
        [
            PredictionResult("Skin Conditions", 0.7),
            PredictionResult("Digestive Issues", 0.2),
        ],
        [
            PredictionResult("Skin Conditions", 0.55),
            PredictionResult("Digestive Issues", 0.4),
        ],
    ]


def _report():
    return build_evaluation_report(
        y_true=["Digestive Issues", "Skin Conditions", "Digestive Issues"],
        predictions=_predictions(),
        labels=LABELS,
        top_k=2,
        model={"backend": "fake", "model_path": "model"},
        dataset={"path": "eval.parquet", "fingerprint": "abc"},
        segments={"record_type": ["owner", "owner", "clinical"]},
        elapsed_seconds=0.5,
    )


def test_build_report_uses_explicit_labels_and_top_k():
    report = _report()

    assert report["labels"] == LABELS
    assert report["summary"]["accuracy"] == pytest.approx(2 / 3)
    assert report["summary"]["top_2_accuracy"] == 1.0
    assert report["summary"]["classification_report"]["Ear Conditions"]["support"] == 0.0
    assert len(report["summary"]["confusion_matrix"]) == len(LABELS)
    assert report["performance"]["rows_per_second"] == 6.0


def test_build_report_emits_segment_metrics():
    report = _report()

    assert report["segments"]["record_type"]["owner"]["sample_count"] == 2
    assert report["segments"]["record_type"]["clinical"]["accuracy"] == 0.0


def test_report_round_trip_and_baseline_comparison(tmp_path):
    baseline = _report()
    current = _report()
    current["summary"]["accuracy"] = baseline["summary"]["accuracy"] - 0.02
    current["summary"]["macro_f1"] = baseline["summary"]["macro_f1"] + 0.01

    path = write_evaluation_report(baseline, tmp_path / "baseline.json")
    loaded = read_evaluation_report(path)
    comparison = compare_with_baseline(current, loaded)

    assert comparison["accuracy_delta"] == pytest.approx(-0.02)
    assert comparison["macro_f1_delta"] == pytest.approx(0.01)
    assert regression_failures(
        comparison,
        max_accuracy_drop=0.01,
        max_macro_f1_drop=0.02,
    ) == ["accuracy_delta=-0.020000 exceeds allowed drop -0.010000"]


def test_report_rejects_prediction_count_mismatch():
    with pytest.raises(EvaluationReportError, match="predictions"):
        build_evaluation_report(
            y_true=["Digestive Issues"],
            predictions=[],
            labels=LABELS,
            top_k=1,
            model={},
            dataset={},
        )


def test_read_report_rejects_unknown_schema(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"schema_version": 999}')

    with pytest.raises(EvaluationReportError, match="Unsupported"):
        read_evaluation_report(path)
