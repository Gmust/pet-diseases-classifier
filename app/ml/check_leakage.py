"""
Train/test leakage + duplicate detector.

Reported F1 is only trustworthy if the test set contains examples the model has
not effectively already seen. The synthetic generator and the multi-source merge
can both introduce near-duplicate rows; if duplicates straddle the train/test
split, the score is inflated.

This script:
  1. Loads the dataset (applying the same label map as training).
  2. Flags EXACT duplicate texts and NEAR duplicates (normalized text).
  3. Re-creates the exact stratified split train.py uses and reports how many
     test rows have a duplicate on the training side (true leakage).

Usage
-----
python -m app.ml.check_leakage \
    --data-path data/merged_augmented.parquet \
    --label-map data/label_map.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = _PUNCT.sub("", text)
    return _WS.sub(" ", text)


def _load_label_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: str(v) for k, v in raw.items() if not k.startswith("//") and v}


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect train/test leakage and duplicates.")
    parser.add_argument("--data-path", default="data/merged_augmented.parquet")
    parser.add_argument("--label-map", default="data/label_map.json")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    path = Path(args.data_path)
    df = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
    df = df.dropna(subset=["text", "condition"]).copy()
    df["condition"] = (
        df["condition"].astype(str).str.strip().replace(_load_label_map(args.label_map))
    )
    df["norm"] = df["text"].map(_normalize)

    total = len(df)
    exact_dupes = df.duplicated(subset=["text"]).sum()
    near_dupes = df.duplicated(subset=["norm"]).sum()
    print(f"Rows: {total}")
    print(f"Exact duplicate texts:      {exact_dupes} ({exact_dupes / total:.1%})")
    print(f"Near-duplicate (normalized): {near_dupes} ({near_dupes / total:.1%})")

    # Re-create train.py's stratified split and measure cross-split overlap.
    idx = list(range(total))
    labels = df["condition"].tolist()
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, stratify=labels, random_state=args.random_state
    )
    train_norm = set(df.iloc[train_idx]["norm"])
    test_norm = df.iloc[test_idx]["norm"]
    leaked = test_norm.isin(train_norm).sum()
    print(
        f"\nLEAKAGE: {leaked}/{len(test_idx)} test rows "
        f"({leaked / len(test_idx):.1%}) have a normalized duplicate in train."
    )
    if leaked:
        print("→ Deduplicate BEFORE splitting (drop_duplicates on normalized text), then retrain.")
    else:
        print("→ No cross-split leakage detected. Reported metrics are trustworthy.")


if __name__ == "__main__":
    main()
