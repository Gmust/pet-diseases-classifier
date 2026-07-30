"""
Shared normalization, content-based deduplication, and rejection accounting
used across every dataset-build script (prepare_dataset.py, merge_datasets.py,
fetch_and_merge.py).

Consolidating this in one place is what makes rejection counts in a build
report trustworthy: before this module existed, prepare_dataset.py and
merge_datasets.py each reimplemented text normalization slightly
differently, so a row could count as a "duplicate" under one script's rules
and not the other's.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from app.ml.dataset_schema import normalize_text

__all__ = [
    "RejectionReport",
    "drop_empty_or_short",
    "dedup_content",
    "drop_unmapped_labels",
    "normalize_text",
]


@dataclass
class RejectionReport:
    """Accumulates why rows were dropped during a dataset build, by reason.

    A dataset build without this is unauditable: "N rows removed" tells you
    nothing about whether that was expected dedup or a source silently
    losing most of its rows.
    """

    counts: dict[str, int] = field(default_factory=dict)

    def record(self, reason: str, count: int) -> None:
        if count:
            self.counts[reason] = self.counts.get(reason, 0) + count

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)


def drop_empty_or_short(df: pd.DataFrame, min_length: int, report: RejectionReport) -> pd.DataFrame:
    """Normalize whitespace, then drop rows with no text/condition or text
    shorter than `min_length`."""
    before = len(df)
    df = df.dropna(subset=["text", "condition"]).copy()
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() >= min_length]
    report.record("empty_or_short_text", before - len(df))
    return df.reset_index(drop=True)


def dedup_content(df: pd.DataFrame, report: RejectionReport) -> pd.DataFrame:
    """Drop exact + near-duplicate rows by normalized text (keep first)."""
    before = len(df)
    df = df.copy()
    df["_norm"] = df["text"].map(normalize_text)
    df = df.drop_duplicates(subset=["_norm"]).drop(columns="_norm")
    report.record("duplicate_text", before - len(df))
    return df.reset_index(drop=True)


def drop_unmapped_labels(
    df: pd.DataFrame, canonical_labels: Iterable[str], report: RejectionReport
) -> pd.DataFrame:
    """Drop rows whose `condition` is not one of the canonical classes —
    typically a label-map gap rather than a data-quality issue, but it must
    not silently create an unrecognized class."""
    before = len(df)
    labels = set(canonical_labels)
    df = df[df["condition"].isin(labels)].copy()
    report.record("non_canonical_label", before - len(df))
    return df.reset_index(drop=True)
