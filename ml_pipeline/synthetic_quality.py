"""
Quality gates for Gemini-generated synthetic training rows.

Synthetic text has failure modes real owner text doesn't:

  - **Diagnosis leakage** — the generated sentence contains the target label
    string itself (e.g. the text for "Blood Disorders" literally says "blood
    disorder"), teaching the classifier to pattern-match the label instead of
    the described symptoms.
  - **Near-duplicate templates** — the model reuses the same sentence
    structure across "different" rows in a batch; exact-dup filtering
    (`ml_pipeline.dataset_pipeline.dedup_content`) misses these because the
    wording differs slightly.

This module labels rows with both problems plus generation provenance and a
`needs_review` marker. It does not drop rows — a human (or a stricter
downstream policy) decides what to do with a flagged row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher

import pandas as pd

from ml_pipeline.dataset_schema import normalize_text


def detect_diagnosis_leakage(text: str, condition: str) -> bool:
    """True if the normalized condition label appears verbatim in the text."""
    return normalize_text(condition) in normalize_text(text)


def tag_provenance(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Record what generated each row and when — required to trace a
    synthetic row back to the model/run that produced it."""
    df = df.copy()
    df["generated_by"] = model_name
    df["generated_at"] = datetime.now(UTC).isoformat()
    return df


@dataclass(frozen=True)
class SimilarityFlag:
    row_index: int
    similar_to_index: int
    ratio: float


def find_near_duplicates(df: pd.DataFrame, threshold: float = 0.92) -> list[SimilarityFlag]:
    """O(n^2) per-class near-duplicate scan via SequenceMatcher.

    Synthetic batches are small (tens-to-hundreds of rows per class), so this
    stays fast; do not run this over the full merged dataset.
    """
    flags: list[SimilarityFlag] = []
    for _, group in df.groupby("condition"):
        texts = group["text"].map(normalize_text).tolist()
        indices = group.index.tolist()
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()
                if ratio >= threshold:
                    flags.append(SimilarityFlag(indices[i], indices[j], round(ratio, 3)))
    return flags


def apply_quality_gates(df: pd.DataFrame, similarity_threshold: float = 0.92) -> pd.DataFrame:
    """Add `leaks_diagnosis`, `near_duplicate`, and `needs_review` columns."""
    df = df.copy()
    df["leaks_diagnosis"] = [
        detect_diagnosis_leakage(text, condition)
        for text, condition in zip(df["text"], df["condition"], strict=True)
    ]
    dup_flags = find_near_duplicates(df, threshold=similarity_threshold)
    flagged_indices = {f.row_index for f in dup_flags} | {f.similar_to_index for f in dup_flags}
    df["near_duplicate"] = df.index.isin(flagged_indices)
    df["needs_review"] = df["leaks_diagnosis"] | df["near_duplicate"]
    return df


def quality_summary(df: pd.DataFrame) -> dict[str, int]:
    return {
        "total_rows": len(df),
        "leaks_diagnosis": int(df["leaks_diagnosis"].sum()) if "leaks_diagnosis" in df else 0,
        "near_duplicate": int(df["near_duplicate"].sum()) if "near_duplicate" in df else 0,
        "needs_review": int(df["needs_review"].sum()) if "needs_review" in df else 0,
    }
