"""Safety and Torch/ONNX parity gates for model release candidates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.inference.predictor import PredictionResult
from app.triage.safety import detect_red_flags

REPORT_SCHEMA_VERSION = 1


class ReleaseGateError(ValueError):
    """Raised when gate configuration or evaluation inputs are invalid."""


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseGateError(f"Could not read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseGateError(f"Expected a JSON object in {path}.")
    return value


def _write_report(report: Mapping[str, Any], path: str | Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote release-gate report to: {output}")


def evaluate_safety_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    min_recall: float,
    max_false_positive_rate: float,
) -> dict[str, Any]:
    if not 0 <= min_recall <= 1 or not 0 <= max_false_positive_rate <= 1:
        raise ReleaseGateError("Safety thresholds must be between 0 and 1.")
    if not cases:
        raise ReleaseGateError("Safety evaluation requires at least one case.")

    seen_ids: set[str] = set()
    true_positive = false_negative = false_positive = true_negative = 0
    failures: list[dict[str, Any]] = []
    reasons_covered: set[str] = set()

    for case in cases:
        case_id = str(case.get("id", "")).strip()
        text = str(case.get("text", "")).strip()
        expected = case.get("expected_triggered")
        expected_reason = case.get("expected_reason")
        if not case_id or case_id in seen_ids:
            raise ReleaseGateError(f"Safety case ids must be non-empty and unique: {case_id!r}.")
        if not text or not isinstance(expected, bool):
            raise ReleaseGateError(f"Safety case {case_id!r} has invalid text or expectation.")
        seen_ids.add(case_id)

        actual = detect_red_flags(text)
        reason_matches = expected_reason is None or actual.reason == expected_reason
        passed = actual.triggered == expected and reason_matches
        if expected and actual.triggered:
            true_positive += 1
            if actual.reason:
                reasons_covered.add(actual.reason)
        elif expected:
            false_negative += 1
        elif actual.triggered:
            false_positive += 1
        else:
            true_negative += 1
        if not passed:
            failures.append(
                {
                    "id": case_id,
                    "expected_triggered": expected,
                    "expected_reason": expected_reason,
                    "actual_triggered": actual.triggered,
                    "actual_reason": actual.reason,
                }
            )

    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    if positive_count == 0 or negative_count == 0:
        raise ReleaseGateError("Safety evaluation requires positive and negative cases.")
    recall = true_positive / positive_count
    false_positive_rate = false_positive / negative_count
    gate_failures = []
    if recall < min_recall:
        gate_failures.append(f"recall {recall:.6f} is below {min_recall:.6f}")
    if false_positive_rate > max_false_positive_rate:
        gate_failures.append(
            f"false_positive_rate {false_positive_rate:.6f} exceeds "
            f"{max_false_positive_rate:.6f}"
        )
    if failures:
        gate_failures.append(f"{len(failures)} fixture expectation(s) failed")

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "gate": "deterministic_safety",
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": not gate_failures,
        "thresholds": {
            "min_recall": min_recall,
            "max_false_positive_rate": max_false_positive_rate,
        },
        "metrics": {
            "case_count": len(cases),
            "true_positive": true_positive,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "recall": recall,
            "false_positive_rate": false_positive_rate,
            "reasons_covered": sorted(reasons_covered),
        },
        "fixture_failures": failures,
        "gate_failures": gate_failures,
    }


def evaluate_backend_parity(
    *,
    y_true: Sequence[str],
    torch_predictions: Sequence[PredictionResult],
    onnx_predictions: Sequence[PredictionResult],
    min_label_agreement: float,
    max_accuracy_delta: float,
    max_mean_confidence_delta: float,
) -> dict[str, Any]:
    thresholds = (min_label_agreement, max_accuracy_delta, max_mean_confidence_delta)
    if not 0 <= min_label_agreement <= 1 or any(value < 0 for value in thresholds[1:]):
        raise ReleaseGateError("Parity thresholds must be non-negative; agreement is in [0, 1].")
    if not y_true or len(torch_predictions) != len(y_true) or len(onnx_predictions) != len(y_true):
        raise ReleaseGateError("Parity evaluation requires equally sized, non-empty inputs.")

    torch_labels = [prediction.predicted_condition for prediction in torch_predictions]
    onnx_labels = [prediction.predicted_condition for prediction in onnx_predictions]
    agreement = sum(
        torch_label == onnx_label
        for torch_label, onnx_label in zip(torch_labels, onnx_labels, strict=True)
    ) / len(y_true)
    torch_accuracy = sum(
        gold == predicted for gold, predicted in zip(y_true, torch_labels, strict=True)
    ) / len(y_true)
    onnx_accuracy = sum(
        gold == predicted for gold, predicted in zip(y_true, onnx_labels, strict=True)
    ) / len(y_true)
    confidence_deltas = [
        abs(torch_prediction.confidence - onnx_prediction.confidence)
        for torch_prediction, onnx_prediction in zip(
            torch_predictions,
            onnx_predictions,
            strict=True,
        )
    ]
    accuracy_delta = abs(torch_accuracy - onnx_accuracy)
    mean_confidence_delta = sum(confidence_deltas) / len(confidence_deltas)

    gate_failures = []
    if agreement < min_label_agreement:
        gate_failures.append(f"label_agreement {agreement:.6f} is below {min_label_agreement:.6f}")
    if accuracy_delta > max_accuracy_delta:
        gate_failures.append(
            f"accuracy_delta {accuracy_delta:.6f} exceeds {max_accuracy_delta:.6f}"
        )
    if mean_confidence_delta > max_mean_confidence_delta:
        gate_failures.append(
            f"mean_confidence_delta {mean_confidence_delta:.6f} exceeds "
            f"{max_mean_confidence_delta:.6f}"
        )

    mismatches = [
        {
            "index": index,
            "gold": y_true[index],
            "torch": torch_labels[index],
            "onnx": onnx_labels[index],
        }
        for index in range(len(y_true))
        if torch_labels[index] != onnx_labels[index]
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "gate": "torch_onnx_parity",
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": not gate_failures,
        "thresholds": {
            "min_label_agreement": min_label_agreement,
            "max_accuracy_delta": max_accuracy_delta,
            "max_mean_confidence_delta": max_mean_confidence_delta,
        },
        "metrics": {
            "sample_count": len(y_true),
            "label_agreement": agreement,
            "torch_accuracy": torch_accuracy,
            "onnx_accuracy": onnx_accuracy,
            "accuracy_delta": accuracy_delta,
            "mean_confidence_delta": mean_confidence_delta,
            "max_confidence_delta": max(confidence_deltas),
        },
        "label_mismatches": mismatches,
        "gate_failures": gate_failures,
    }


def _load_gate_config(path: str | Path, gate: str) -> dict[str, float]:
    config = _load_json(path)
    value = config.get(gate)
    if not isinstance(value, dict):
        raise ReleaseGateError(f"Missing {gate!r} object in {path}.")
    try:
        return {key: float(raw) for key, raw in value.items()}
    except (TypeError, ValueError) as exc:
        raise ReleaseGateError(f"Gate {gate!r} contains non-numeric thresholds.") from exc


def _load_label_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    raw = _load_json(path)
    return {key: str(value) for key, value in raw.items() if not key.startswith("//") and value}


def _run_safety(args: argparse.Namespace) -> int:
    config = _load_gate_config(args.config, "safety")
    fixture = _load_json(args.cases)
    cases = fixture.get("cases")
    if not isinstance(cases, list):
        raise ReleaseGateError("Safety fixture must contain a `cases` array.")
    report = evaluate_safety_cases(
        cases,
        min_recall=config["min_recall"],
        max_false_positive_rate=config["max_false_positive_rate"],
    )
    _write_report(report, args.output_json)
    return 0 if report["passed"] else 1


def _run_parity(args: argparse.Namespace) -> int:
    import pandas as pd

    from app.inference.onnx_predictor import OnnxPredictor
    from app.inference.predictor import Predictor

    config = _load_gate_config(args.config, "parity")
    path = Path(args.data)
    frame = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
    frame = frame.dropna(subset=["text", "condition"]).copy()
    label_map = _load_label_map(args.label_map)
    frame["condition"] = (
        frame["condition"].astype(str).str.strip().map(lambda value: label_map.get(value, value))
    )

    torch_predictor = Predictor.from_paths(model_path=args.torch_model)
    onnx_predictor = OnnxPredictor.from_paths(model_path=args.onnx_model)
    if torch_predictor.metadata.labels != onnx_predictor.metadata.labels:
        raise ReleaseGateError("Torch and ONNX bundles expose different label orders.")
    known = set(torch_predictor.metadata.labels)
    frame = frame[frame["condition"].isin(known)]
    if frame.empty:
        raise ReleaseGateError("No parity rows have labels known by both models.")

    texts = frame["text"].astype(str).tolist()
    y_true = frame["condition"].astype(str).tolist()
    torch_predictions = torch_predictor.predict_batch(texts, batch_size=args.batch_size)
    onnx_predictions = onnx_predictor.predict_batch(texts, batch_size=args.batch_size)
    report = evaluate_backend_parity(
        y_true=y_true,
        torch_predictions=torch_predictions,
        onnx_predictions=onnx_predictions,
        min_label_agreement=config["min_label_agreement"],
        max_accuracy_delta=config["max_accuracy_delta"],
        max_mean_confidence_delta=config["max_mean_confidence_delta"],
    )
    report["artifacts"] = {
        "torch_model": args.torch_model,
        "onnx_model": args.onnx_model,
        "dataset": str(path),
    }
    _write_report(report, args.output_json)
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/release-gates.json")
    subparsers = parser.add_subparsers(dest="gate", required=True)

    safety = subparsers.add_parser("safety", help="Run deterministic red-flag fixtures.")
    safety.add_argument("--cases", default="data/safety_eval.json")
    safety.add_argument("--output-json", default=None)
    safety.set_defaults(handler=_run_safety)

    parity = subparsers.add_parser("parity", help="Compare Torch and ONNX bundles.")
    parity.add_argument("--torch-model", default="models/transformer_model")
    parity.add_argument("--onnx-model", default="models/transformer_model_onnx")
    parity.add_argument("--data", default="data/owner_eval.parquet")
    parity.add_argument("--label-map", default="data/label_map.json")
    parity.add_argument("--batch-size", type=int, default=32)
    parity.add_argument("--output-json", default=None)
    parity.set_defaults(handler=_run_parity)

    args = parser.parse_args()
    if getattr(args, "batch_size", 1) < 1:
        raise ReleaseGateError("--batch-size must be at least 1.")
    return args.handler(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseGateError as exc:
        raise SystemExit(f"Release gate configuration error: {exc}") from exc
