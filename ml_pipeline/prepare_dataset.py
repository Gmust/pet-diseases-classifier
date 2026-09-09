"""
Prepare a clean, balanced training set + an owner-language evaluation holdout.

Why
---
The model is served pet-OWNER text ("my cat is sneezing"), so accuracy must be
judged on that distribution — not on clinical shorthand. This script:

  1. Applies the label map (collapse raw labels → 16 canonical classes).
  2. Deduplicates exact + near-duplicate text (prevents leaked/inflated metrics).
  3. Carves out an OWNER-OBSERVATION holdout (`owner_eval.parquet`) with NO overlap
     with training — this is the honest yardstick for "does change X help real usage?".
  4. Downsamples over-represented classes so the model stops defaulting to them.

Outputs `train_balanced.parquet` (for train.py) and `owner_eval.parquet` (for
evaluate.py). Run `evaluate.py` on the holdout after each retrain to compare,
e.g., before/after adding PetEVAL — keep a change only if it helps the holdout.

Usage
-----
python -m ml_pipeline.prepare_dataset \
    --input data/merged_augmented.parquet \
    --label-map data/label_map.json \
    --train-out data/train_balanced.parquet \
    --eval-out data/owner_eval.parquet \
    --downsample-max 600 \
    --owner-eval-frac 0.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml_pipeline.dataset_pipeline import RejectionReport, dedup_content, drop_empty_or_short
from ml_pipeline.dataset_schema import assign_row_ids, validate_label_map
from ml_pipeline.split_manifest import SplitLeakageError, write_split_manifest


def load_label_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: str(v) for k, v in raw.items() if not k.startswith("//") and v}


def apply_label_map(df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["condition"] = df["condition"].astype(str).str.strip().map(lambda c: label_map.get(c, c))
    return df


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact + near-duplicate rows by normalized text (keep first)."""
    return dedup_content(df, RejectionReport())


def carve_owner_eval(
    df: pd.DataFrame,
    record_type: str,
    frac: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fraction of owner-observation rows for evaluation.

    Training still keeps the remaining owner rows (so the model learns the owner
    register), but the holdout is disjoint — an honest test of real-usage accuracy.
    """
    if "record_type" not in df.columns:
        return df.reset_index(drop=True), df.iloc[0:0].reset_index(drop=True)
    owner = df[df["record_type"].astype(str).str.strip() == record_type]
    if owner.empty or frac <= 0:
        return df.reset_index(drop=True), df.iloc[0:0].reset_index(drop=True)
    eval_df = owner.sample(frac=min(frac, 1.0), random_state=seed)
    train_df = df.drop(eval_df.index)
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def downsample(df: pd.DataFrame, max_per_class: int | None, seed: int = 42) -> pd.DataFrame:
    """Cap each class at `max_per_class` rows (random subsample of larger classes)."""
    if not max_per_class or max_per_class <= 0:
        return df.reset_index(drop=True)
    parts = []
    for _, group in df.groupby("condition"):
        parts.append(
            group.sample(n=max_per_class, random_state=seed)
            if len(group) > max_per_class
            else group
        )
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _dist(df: pd.DataFrame) -> str:
    return df["condition"].value_counts().to_string()


def main() -> None:
    p = argparse.ArgumentParser(description="Dedup, rebalance, and carve an owner holdout.")
    p.add_argument("--input", default="data/merged_augmented.parquet")
    p.add_argument("--label-map", default="data/label_map.json")
    p.add_argument("--train-out", default="data/train_balanced.parquet")
    p.add_argument("--eval-out", default="data/owner_eval.parquet")
    p.add_argument(
        "--downsample-max",
        type=int,
        default=600,
        help="Cap rows per class in the training set (0 = no cap).",
    )
    p.add_argument(
        "--owner-eval-frac",
        type=float,
        default=0.3,
        help="Fraction of owner-observation rows held out for evaluation.",
    )
    p.add_argument("--owner-record-type", default="Owner Observation")
    p.add_argument("--min-text-length", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split-manifest-out",
        default=None,
        help="Path for the row-id-keyed split manifest (default: <train-out>.split-manifest.json).",
    )
    args = p.parse_args()

    label_map = load_label_map(args.label_map)
    label_map_problems = validate_label_map(label_map)
    if label_map_problems:
        raise SystemExit("Label map validation failed:\n" + "\n".join(label_map_problems))

    path = Path(args.input)
    df = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
    report = RejectionReport()
    df = drop_empty_or_short(df, args.min_text_length, report)
    df = apply_label_map(df, label_map)

    print(f"Loaded {len(df)} rows, {df['condition'].nunique()} classes")

    # Dedup BEFORE carving the holdout so eval rows can't be duplicates of train rows.
    before = len(df)
    df = dedup_content(df, report)
    print(f"Dedup: {before} → {len(df)} rows ({before - len(df)} duplicates removed)")

    # Rows already tagged with provenance upstream (fetch_and_merge.py source
    # adapters) keep their row_id; anything else gets one assigned here.
    if "row_id" not in df.columns:
        df = assign_row_ids(df, source="prepared")

    train_df, eval_df = carve_owner_eval(
        df, args.owner_record_type, args.owner_eval_frac, args.seed
    )
    print(
        f"Owner-language holdout: {len(eval_df)} rows "
        f"({eval_df['condition'].nunique() if len(eval_df) else 0} classes)"
    )

    train_df = downsample(train_df, args.downsample_max, args.seed)

    print("\n=== Training set class distribution (after rebalance) ===")
    print(_dist(train_df))
    if len(eval_df):
        print("\n=== Owner holdout class distribution ===")
        print(_dist(eval_df))

    # Hard gate: fail the build if any duplicate-text family crosses the
    # train/eval boundary (see ml_pipeline.split_manifest). Nothing is written on failure.
    splits = {"train": train_df, "eval": eval_df} if len(eval_df) else {"train": train_df}
    manifest_path = args.split_manifest_out or f"{args.train_out}.split-manifest.json"
    try:
        write_split_manifest(splits, manifest_path)
    except SplitLeakageError as exc:
        raise SystemExit(f"Split publication failed: {exc}") from exc
    print(f"\nNo train/eval text overlap. Wrote split manifest to {manifest_path}")

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(args.train_out, index=False)
    if len(eval_df):
        eval_df.to_parquet(args.eval_out, index=False)
    print(
        f"\nWrote {args.train_out} ({len(train_df)} rows)"
        + (f" and {args.eval_out} ({len(eval_df)} rows)" if len(eval_df) else "")
    )
    print(f"\nRejection report ({report.total} rows dropped total):")
    for reason, count in report.as_dict().items():
        print(f"  {reason}: {count}")
    print("\nNext: retrain on the balanced set, then score the holdout:")
    print(
        f"  python -m ml_pipeline.train --data-path {args.train_out} "
        f"--label-map {args.label_map} --model-dir models/transformer_model --loss ce"
    )
    print(
        f"  python -m ml_pipeline.evaluate --model-dir models/transformer_model --data {args.eval_out}"
    )


if __name__ == "__main__":
    main()
