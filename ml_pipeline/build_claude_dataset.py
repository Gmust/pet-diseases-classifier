"""
Assemble the text-based training set and the real-owner eval set from the
`ml_pipeline.claude_labeling` outputs.

Labels follow docs/labeling-guide.md: a row's category is what its text supports,
not a cause the text never mentions.

  train      existing rows Claude marked usable, relabeled to Claude's category
           + real Big Red Bark owner questions on which both label passes agree
           + generated rows whose blind re-label matches the intended category
             and that pass the synthetic quality gates
  eval       a per-class-capped sample of the agreed Big Red Bark questions:
             real owner text, never seen in training (exact and near duplicates
             are removed from train)

Usage:
    python -m ml_pipeline.build_claude_dataset
"""

from __future__ import annotations

import argparse

import pandas as pd

from ml_pipeline.claude_labeling import GENERATED_SOURCE
from ml_pipeline.dataset_schema import compute_row_id, normalize_text, validate_rows
from ml_pipeline.synthetic_quality import apply_quality_gates

BRB_RECORD_TYPE = "Owner Question (Big Red Bark)"
BRB_SOURCE = "Sr523/big-red-bark-chat-evaluation"
COLUMNS = ["text", "condition", "record_type", "source", "row_id", "label_source"]


def relabel(rows: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Keep rows Claude marked usable and give them Claude's category."""
    merged = rows.merge(labels, left_on="row_id", right_on="id")
    kept = merged[merged["usable"].eq(True)].copy()
    kept["condition"] = kept["claude_condition"]
    kept["label_source"] = "claude-" + kept["variant"]
    return kept


def agreed(labels_a: pd.DataFrame, labels_b: pd.DataFrame) -> pd.DataFrame:
    """Ids on which two independent passes both say usable and pick one category."""
    both = labels_a.merge(labels_b, on="id", suffixes=("_a", "_b"))
    same = (
        both["usable_a"].eq(True)
        & both["usable_b"].eq(True)
        & (both["claude_condition_a"] == both["claude_condition_b"])
    )
    return both.loc[same, ["id", "claude_condition_a"]].rename(
        columns={"claude_condition_a": "condition"}
    )


def split_eval(pool: pd.DataFrame, per_class: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """At most `per_class` rows of each category go to eval; the rest stay for train."""
    eval_df = pool.sample(frac=1, random_state=seed).groupby("condition").head(per_class)
    return eval_df, pool.drop(eval_df.index)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(normalize_text(text).split())


def drop_overlap(train: pd.DataFrame, eval_df: pd.DataFrame, jaccard: float) -> pd.DataFrame:
    """Remove train rows that duplicate or near-duplicate an eval row.

    ponytail: brute-force token Jaccard (train x eval, ~10M set ops), fine at this
    size; switch to MinHash if either side grows past ~50k rows.
    """
    eval_tokens = [_tokens(t) for t in eval_df["text"]]
    eval_norm = set(eval_df["text"].map(normalize_text))

    def leaks(text: str) -> bool:
        if normalize_text(text) in eval_norm:
            return True
        toks = _tokens(text)
        return any(toks and len(toks & e) / len(toks | e) >= jaccard for e in eval_tokens)

    return train[~train["text"].map(leaks)]


def build(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing = relabel(pd.read_parquet(args.train), pd.read_parquet(args.train_labels))

    questions = pd.read_parquet(args.brb)
    pool = questions.merge(
        agreed(pd.read_parquet(args.brb_labels_a), pd.read_parquet(args.brb_labels_b)),
        left_on="row_id",
        right_on="id",
    )
    pool = pool.assign(
        record_type=BRB_RECORD_TYPE, source=BRB_SOURCE, label_source="claude-a+b"
    ).drop_duplicates(subset="text")
    pool = pool[~pool["text"].map(normalize_text).duplicated()]
    eval_df, brb_train = split_eval(pool, args.eval_per_class, args.seed)

    generated = pd.read_parquet(args.gen).merge(
        pd.read_parquet(args.gen_labels), left_on="row_id", right_on="id"
    )
    generated = generated[
        generated["usable"].eq(True) & (generated["claude_condition"] == generated["condition"])
    ]
    generated = apply_quality_gates(generated.reset_index(drop=True))
    generated = generated[~generated["needs_review"]].assign(
        source=GENERATED_SOURCE, label_source="claude-gen+b"
    )
    generated["row_id"] = [
        compute_row_id(t, c, GENERATED_SOURCE)
        for t, c in zip(generated["text"], generated["condition"], strict=True)
    ]

    train = pd.concat([existing[COLUMNS], brb_train[COLUMNS], generated[COLUMNS]])
    train = train[~train["text"].map(normalize_text).duplicated()]
    train = drop_overlap(train, eval_df, args.jaccard)
    return train.reset_index(drop=True), eval_df[COLUMNS].reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build text-based train and real-owner eval sets.")
    p.add_argument("--train", default="data/train_balanced.parquet")
    p.add_argument("--train-labels", default="data/train_labels_a.parquet")
    p.add_argument("--brb", default="data/brb_questions.parquet")
    p.add_argument("--brb-labels-a", default="data/brb_labels_a.parquet")
    p.add_argument("--brb-labels-b", default="data/brb_labels_b.parquet")
    p.add_argument("--gen", default="data/claude_gen_raw.parquet")
    p.add_argument("--gen-labels", default="data/claude_gen_labels_b.parquet")
    p.add_argument("--eval-per-class", type=int, default=40)
    p.add_argument("--jaccard", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-out", default="data/train_claude.parquet")
    p.add_argument("--eval-out", default="data/owner_eval_real.parquet")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train, eval_df = build(args)
    for name, df in (("train", train), ("eval", eval_df)):
        problems = validate_rows(df)
        if problems:
            raise SystemExit(f"{name} set invalid: {problems}")
    train.to_parquet(args.train_out, index=False)
    eval_df.to_parquet(args.eval_out, index=False)
    print(f"train: {len(train)} rows → {args.train_out}")
    print(train.groupby(["condition", "record_type"]).size().unstack(fill_value=0).to_string())
    print(f"\neval: {len(eval_df)} rows → {args.eval_out}")
    print(eval_df["condition"].value_counts().to_string())


if __name__ == "__main__":
    main()
