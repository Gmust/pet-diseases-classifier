"""
Evaluate a trained model on a held-out parquet — the owner-language yardstick.

Run this on `owner_eval.parquet` (from prepare_dataset.py) after each retrain.
Because the holdout is pet-OWNER text, the numbers here reflect real-usage
accuracy. Use it to judge data changes honestly: e.g. retrain with PetEVAL added,
re-run this, and keep PetEVAL only if owner-holdout macro-F1 actually improves.

Usage
-----
python -m app.ml.evaluate \
    --model-dir models/transformer_model \
    --data data/owner_eval.parquet \
    --label-map data/label_map.json
"""
from __future__ import annotations

import argparse
import json
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
    p.add_argument("--backend", choices=["torch", "onnx"], default="torch",
                   help="Which predictor to score. Use 'onnx' (with an ONNX model dir) to "
                        "parity-check the quantized model against the torch baseline.")
    p.add_argument("--top-k", type=int, default=3,
                   help="Also report top-k accuracy (correct label within top-k).")
    args = p.parse_args()

    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report

    path = Path(args.data)
    df = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
    df = df.dropna(subset=["text", "condition"]).copy()
    label_map = _load_label_map(args.label_map)
    df["condition"] = df["condition"].astype(str).str.strip().map(lambda c: label_map.get(c, c))

    if args.backend == "onnx":
        from app.ml.onnx_predictor import OnnxPredictor

        predictor = OnnxPredictor.from_paths(model_path=args.model_dir)
    else:
        from app.ml.predictor import Predictor

        predictor = Predictor.from_paths(model_path=args.model_dir)
    known = set(predictor._id2label.values())  # noqa: SLF001 — internal read is fine here
    df = df[df["condition"].isin(known)]
    if df.empty:
        raise SystemExit("No eval rows have labels the model knows. Check the label map / classes.")

    y_true, y_pred, topk_hits = [], [], 0
    for text, gold in zip(df["text"], df["condition"]):
        results = predictor.predict_top_k(text, k=args.top_k)
        y_true.append(gold)
        y_pred.append(results[0].predicted_condition)
        if gold in {r.predicted_condition for r in results}:
            topk_hits += 1

    print(f"\nEvaluated {len(y_true)} owner-language rows from {path.name}\n")
    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"Top-1 accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Top-{args.top_k} accuracy: {topk_hits / len(y_true):.3f}")


if __name__ == "__main__":
    main()
