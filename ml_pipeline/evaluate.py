"""
Evaluate a trained model on a held-out parquet — the owner-language yardstick.

Run this on `owner_eval.parquet` (from prepare_dataset.py) after each retrain.
Because the holdout is pet-OWNER text, the numbers here reflect real-usage
accuracy. Use it to judge data changes honestly: e.g. retrain with PetEVAL added,
re-run this, and keep PetEVAL only if owner-holdout macro-F1 actually improves.

Usage
-----
python -m ml_pipeline.evaluate \
    --model-dir models/transformer_model \
    --data data/owner_eval.parquet \
    --label-map data/label_map.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load_label_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: str(v) for k, v in raw.items() if not k.startswith("//") and v}


def main() -> None:
    p = argparse.ArgumentParser(description="Score a trained model on a holdout parquet.")
    p.add_argument("--model-dir", default="models/transformer_model")
    p.add_argument("--data", default="data/owner_eval.parquet")
    p.add_argument("--label-map", default="data/label_map.json")
    p.add_argument(
        "--backend",
        choices=["torch", "onnx"],
        default="torch",
        help="Which predictor to score. Use 'onnx' (with an ONNX model dir) to "
        "parity-check the quantized model against the torch baseline.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Also report top-k accuracy (correct label within top-k).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size for Torch and ONNX evaluation.",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Write the complete machine-readable evaluation report to this path.",
    )
    p.add_argument(
        "--baseline-json",
        default=None,
        help="Compare summary metrics with a previous evaluation report.",
    )
    p.add_argument(
        "--max-accuracy-drop",
        type=float,
        default=None,
        help="Fail if accuracy drops by more than this amount versus --baseline-json.",
    )
    p.add_argument(
        "--max-macro-f1-drop",
        type=float,
        default=None,
        help="Fail if macro F1 drops by more than this amount versus --baseline-json.",
    )
    p.add_argument(
        "--enforce-coverage",
        action="store_true",
        help="Release-gate mode: fail instead of scoring if any of the 16 "
        "product-claimed classes has zero rows in this eval set.",
    )
    p.add_argument(
        "--emit-calibration",
        default=None,
        help="Path to append a versioned selective-accuracy/abstention-threshold report "
        "(see ml_pipeline.calibration). Not written unless this is set.",
    )
    p.add_argument(
        "--calibration-target-accuracy",
        type=float,
        action="append",
        default=None,
        help="Target accuracy to find an abstention threshold for. Repeatable. "
        "Default: 0.90 and 0.95.",
    )
    args = p.parse_args()

    import pandas as pd
    from sklearn.metrics import classification_report

    from ml_pipeline.eval_coverage import (
        EvaluationCoverageError,
        build_coverage_report,
        require_full_coverage,
    )
    from ml_pipeline.evaluation_report import (
        EvaluationReportError,
        build_evaluation_report,
        compare_with_baseline,
        read_evaluation_report,
        regression_failures,
        write_evaluation_report,
    )
    from ml_pipeline.run_metadata import file_fingerprint

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")

    path = Path(args.data)
    df = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
    df = df.dropna(subset=["text", "condition"]).copy()
    label_map = _load_label_map(args.label_map)
    df["condition"] = df["condition"].astype(str).str.strip().map(lambda c: label_map.get(c, c))

    coverage = build_coverage_report(df)
    if coverage.missing_classes:
        print(
            f"[coverage] {len(coverage.missing_classes)} class(es) with zero eval rows: "
            f"{coverage.missing_classes}"
        )
    if not coverage.species_checked:
        print("[coverage] no `species` column — species-segment coverage was not checked.")
    if args.enforce_coverage:
        try:
            require_full_coverage(df)
        except EvaluationCoverageError as exc:
            raise SystemExit(f"Release evaluation failed coverage gate: {exc}") from exc

    from app.inference.protocols import Classifier

    predictor: Classifier
    if args.backend == "onnx":
        from app.inference.onnx_predictor import OnnxPredictor

        predictor = OnnxPredictor.from_paths(model_path=args.model_dir)
    else:
        from app.inference.predictor import Predictor

        predictor = Predictor.from_paths(model_path=args.model_dir)
    labels = list(predictor.metadata.labels)
    known = set(labels)
    df = df[df["condition"].isin(known)]
    if df.empty:
        raise SystemExit("No eval rows have labels the model knows. Check the label map / classes.")
    if args.top_k < 1 or args.top_k > len(labels):
        raise SystemExit(f"--top-k must be between 1 and {len(labels)}.")

    texts = df["text"].astype(str).tolist()
    y_true = df["condition"].astype(str).tolist()
    started = time.perf_counter()
    predictions = predictor.predict_top_k_batch(
        texts,
        k=args.top_k,
        batch_size=args.batch_size,
    )
    elapsed_seconds = time.perf_counter() - started
    y_pred = [row[0].predicted_condition for row in predictions]
    confidences = [row[0].confidence for row in predictions]

    segment_values = {
        column: df[column].fillna("unknown").astype(str).tolist()
        for column in ("record_type", "species")
        if column in df.columns
    }
    report = build_evaluation_report(
        y_true=y_true,
        predictions=predictions,
        labels=labels,
        top_k=args.top_k,
        model={
            "backend": predictor.metadata.backend,
            "model_path": predictor.metadata.model_path,
        },
        dataset={
            "path": str(path),
            "fingerprint": file_fingerprint(str(path)),
            "coverage": {
                "missing_classes": coverage.missing_classes,
                "missing_species": coverage.missing_species,
                "species_checked": coverage.species_checked,
            },
        },
        segments=segment_values,
        elapsed_seconds=elapsed_seconds,
    )

    comparison_failures: list[str] = []
    if args.baseline_json:
        try:
            baseline = read_evaluation_report(args.baseline_json)
            comparison = compare_with_baseline(report, baseline)
            comparison_failures = regression_failures(
                comparison,
                max_accuracy_drop=args.max_accuracy_drop,
                max_macro_f1_drop=args.max_macro_f1_drop,
            )
        except EvaluationReportError as exc:
            raise SystemExit(f"Evaluation baseline comparison failed: {exc}") from exc
        report["baseline_comparison"] = {
            "path": args.baseline_json,
            **comparison,
            "failures": comparison_failures,
        }

    print(f"\nEvaluated {len(y_true)} owner-language rows from {path.name}\n")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=labels,
            zero_division=0,
        )
    )
    summary = report["summary"]
    print(f"Top-1 accuracy: {summary['accuracy']:.3f}")
    print(f"Top-{args.top_k} accuracy: {summary[f'top_{args.top_k}_accuracy']:.3f}")
    print(
        f"Batched inference: {elapsed_seconds:.3f}s "
        f"({report['performance']['rows_per_second']:.1f} rows/s)"
    )
    if args.baseline_json:
        print(f"Baseline deltas: {report['baseline_comparison']}")

    if args.output_json:
        output_path = write_evaluation_report(report, args.output_json)
        print(f"Wrote evaluation report to: {output_path}")

    if args.emit_calibration:
        from ml_pipeline.calibration import (
            build_abstention_policy,
            compute_selective_accuracy_curve,
            write_calibration_report,
        )
        from ml_pipeline.run_metadata import git_commit

        correct = [gold == pred for gold, pred in zip(y_true, y_pred, strict=True)]
        curve = compute_selective_accuracy_curve(confidences, correct)
        targets = args.calibration_target_accuracy or [0.90, 0.95]
        commit = git_commit()
        policies = []
        for target in targets:
            policy = build_abstention_policy(curve, target, model_commit=commit)
            if policy is None:
                print(f"[calibration] no threshold reaches {target:.0%} accuracy on this eval set.")
            else:
                policies.append(policy)
                print(
                    f"[calibration] target={target:.0%} -> threshold={policy.threshold:.2f} "
                    f"(achieved accuracy={policy.achieved_accuracy:.3f}, "
                    f"coverage={policy.achieved_coverage:.3f})"
                )
        report_path = write_calibration_report(args.emit_calibration, curve, policies)
        print(f"Wrote calibration report to: {report_path}")

    if comparison_failures:
        raise SystemExit("Evaluation regression gate failed: " + "; ".join(comparison_failures))


if __name__ == "__main__":
    main()
