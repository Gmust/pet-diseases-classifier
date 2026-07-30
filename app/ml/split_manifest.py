"""
Immutable, row-id-keyed split manifests with cross-split leakage checks.

Leakage is checked by *duplicate-text family* (normalized text + condition),
not just exact row id — a near-duplicate row that entered from a different
source than its sibling still shares a family key, so it cannot silently end
up in a different split (see "Immutable leakage-resistant splits" in
specs/reproducible-ml-lifecycle/spec.md). Exact-duplicate rows are already
removed before splitting (`app.ml.dataset_pipeline.dedup_content`); this
module is the defensive gate that catches a future pipeline reordering bug,
not the primary dedup step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.ml.dataset_schema import normalize_text


class SplitLeakageError(Exception):
    """Raised when a duplicate-text family has rows in more than one split."""


def duplicate_family_key(text: str, condition: str) -> str:
    return f"{normalize_text(text)}|{condition.strip()}"


def verify_no_cross_split_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """`splits` maps split name -> DataFrame (each must have `text`/`condition`).
    Raises `SplitLeakageError` if any duplicate-text family appears in more
    than one split."""
    family_to_splits: dict[str, set[str]] = {}
    for split_name, df in splits.items():
        for text, condition in zip(df["text"], df["condition"], strict=True):
            key = duplicate_family_key(text, condition)
            family_to_splits.setdefault(key, set()).add(split_name)

    leaking = {key: names for key, names in family_to_splits.items() if len(names) > 1}
    if leaking:
        sample = list(leaking.items())[:5]
        raise SplitLeakageError(
            f"{len(leaking)} duplicate-text famil{'y' if len(leaking) == 1 else 'ies'} "
            f"cross split boundaries, e.g. {sample}"
        )


def build_split_manifest(splits: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Row-id-keyed manifest: split name -> sorted list of row ids. Persisting
    this (not just the row content) lets split membership be diffed/audited
    across pipeline runs independent of how the data itself is re-serialized."""
    for name, df in splits.items():
        if "row_id" not in df.columns:
            raise ValueError(
                f"split {name!r} is missing `row_id` — assign it "
                "(app.ml.dataset_schema.assign_row_ids) before building a manifest"
            )
    return {name: sorted(df["row_id"].tolist()) for name, df in splits.items()}


def write_split_manifest(splits: dict[str, pd.DataFrame], path: str | Path) -> None:
    """Verify no cross-split leakage, then write the row-id manifest. Raises
    `SplitLeakageError` (nothing is written) if leakage is found."""
    verify_no_cross_split_leakage(splits)
    manifest = build_split_manifest(splits)
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
