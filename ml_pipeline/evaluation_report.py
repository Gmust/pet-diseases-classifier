"""Machine-readable, segmented classifier evaluation reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.inference.predictor import PredictionResult

REPORT_SCHEMA_VERSION = 1


class EvaluationReportError(ValueError):
    """Raised when evaluation inputs or a baseline report are invalid."""


def _classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )
    return {
        "sample_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
    }


def _segment_reports(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    segments: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for segment_name, values in segments.items():
        if len(values) != len(y_true):
            raise EvaluationReportError(
                f"Segment {segment_name!r} has {len(values)} values for {len(y_true)} rows."
            )
        groups: dict[str, list[int]] = {}
        for index, value in enumerate(values):
            groups.setdefault(str(value), []).append(index)
        output[segment_name] = {}
        for value, indices in sorted(groups.items()):
            group_true = [y_true[index] for index in indices]
            group_pred = [y_pred[index] for index in indices]
            output[segment_name][value] = _classification_metrics(
                group_true,
                group_pred,
                labels,
            )
    return output


def build_evaluation_report(
    *,
    y_true: Sequence[str],
    predictions: Sequence[Sequence[PredictionResult]],
    labels: Sequence[str],
    top_k: int,
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    segments: Mapping[str, Sequence[str]] | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable evaluation report with explicit labels."""
    if not y_true:
        raise EvaluationReportError("Evaluation requires at least one row.")
    if len(predictions) != len(y_true):
        raise EvaluationReportError(
            f"Received {len(predictions)} predictions for {len(y_true)} labels."
        )
    if len(set(labels)) != len(labels) or not labels:
        raise EvaluationReportError("Model labels must be non-empty and unique.")
    if top_k < 1 or top_k > len(labels):
        raise EvaluationReportError(f"top_k must be between 1 and {len(labels)}.")
    if any(len(row) < top_k for row in predictions):
        raise EvaluationReportError("Every prediction row must contain at least top_k results.")

    y_pred = [row[0].predicted_condition for row in predictions]
    summary = _classification_metrics(y_true, y_pred, labels)
    top_k_hits = sum(
        gold in {result.predicted_condition for result in row[:top_k]}
        for gold, row in zip(y_true, predictions, strict=True)
    )
    summary[f"top_{top_k}_accuracy"] = top_k_hits / len(y_true)

    performance: dict[str, float] = {}
    if elapsed_seconds is not None:
        if elapsed_seconds < 0:
            raise EvaluationReportError("elapsed_seconds cannot be negative.")
        performance["elapsed_seconds"] = elapsed_seconds
        performance["rows_per_second"] = (
            len(y_true) / elapsed_seconds if elapsed_seconds > 0 else 0.0
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": dict(model),
        "dataset": dict(dataset),
        "labels": list(labels),
        "top_k": top_k,
        "summary": summary,
        "segments": _segment_reports(y_true, y_pred, labels, segments or {}),
        "performance": performance,
    }


def compare_with_baseline(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, float]:
    """Return signed metric deltas (current minus baseline)."""
    try:
        current_summary = current["summary"]
        baseline_summary = baseline["summary"]
        return {
            "accuracy_delta": float(current_summary["accuracy"])
            - float(baseline_summary["accuracy"]),
            "macro_f1_delta": float(current_summary["macro_f1"])
            - float(baseline_summary["macro_f1"]),
            "weighted_f1_delta": float(current_summary["weighted_f1"])
            - float(baseline_summary["weighted_f1"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationReportError(
            "Baseline and current reports must contain numeric summary metrics."
        ) from exc


def regression_failures(
    comparison: Mapping[str, float],
    *,
    max_accuracy_drop: float | None,
    max_macro_f1_drop: float | None,
) -> list[str]:
    failures: list[str] = []
    limits = {
        "accuracy_delta": max_accuracy_drop,
        "macro_f1_delta": max_macro_f1_drop,
    }
    for metric, maximum_drop in limits.items():
        if maximum_drop is None:
            continue
        if maximum_drop < 0:
            raise EvaluationReportError(f"Maximum drop for {metric} cannot be negative.")
        if comparison[metric] < -maximum_drop:
            failures.append(
                f"{metric}={comparison[metric]:.6f} exceeds allowed drop {-maximum_drop:.6f}"
            )
    return failures


def read_evaluation_report(path: str | Path) -> dict[str, Any]:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationReportError(f"Could not read evaluation report {path}: {exc}") from exc
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise EvaluationReportError(
            f"Unsupported evaluation report schema: {report.get('schema_version')!r}."
        )
    return report


def write_evaluation_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
