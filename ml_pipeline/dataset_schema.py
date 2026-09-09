"""
Canonical dataset row contract shared by every source adapter and pipeline
stage (fetch_and_merge.py, merge_datasets.py, generate_synthetic.py,
prepare_dataset.py).

Why this exists: the pipeline previously carried only `text`, `condition`,
`record_type` with no source, license, or stable identity — so a dataset
build couldn't be traced back to what produced each row, and splits couldn't
be joined/compared by row across pipeline runs. This module is the single
place that defines what a valid row looks like and how its id is computed;
individual scripts should import it rather than re-deriving the contract.
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd

from app.domain.conditions import CONDITION_METADATA

# The 16 canonical condition classes the model is trained to predict —
# derived from condition_metadata.py so there is exactly one place that
# enumerates them (see design.md "one source of truth per configuration value").
CANONICAL_LABELS: frozenset[str] = frozenset(CONDITION_METADATA.keys())

# Required on every row after a source adapter runs. `row_id` and `source`
# are provenance; `text`/`condition`/`record_type` are the existing contract.
REQUIRED_COLUMNS: tuple[str, ...] = ("row_id", "text", "condition", "record_type", "source")

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


class DatasetValidationError(Exception):
    """Raised when a dataset build cannot proceed with a valid, traceable row set."""


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — used for both
    row-id derivation and duplicate detection so the two stay consistent."""
    text = _PUNCT.sub("", str(text).lower().strip())
    return _WS.sub(" ", text)


def compute_row_id(text: str, condition: str, source: str) -> str:
    """Stable id for a row: a content hash, not a positional index, so it
    survives re-ordering, re-filtering, and re-runs of the pipeline. Two rows
    with the same normalized text/condition/source collide by design — that
    is the intended de-duplication signal downstream (see ml_pipeline.pipeline)."""
    digest = hashlib.sha256(f"{normalize_text(text)}|{condition.strip()}|{source.strip()}".encode())
    return digest.hexdigest()[:16]


def assign_row_ids(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add `source` and a stable `row_id` column. Idempotent: re-running on
    already-tagged rows recomputes the same ids from the same content."""
    df = df.copy()
    df["source"] = source
    df["row_id"] = [
        compute_row_id(text, condition, source)
        for text, condition in zip(df["text"], df["condition"], strict=True)
    ]
    return df


def validate_label_map(label_map: dict[str, str]) -> list[str]:
    """Return problems with a label map's *targets* (not its source keys —
    those are external and expected to vary). Every target must be one of
    the 16 canonical classes; anything else is a typo or a stale mapping
    that would silently create an unrecognized 17th class."""
    problems = []
    for raw_label, target in label_map.items():
        if target not in CANONICAL_LABELS:
            problems.append(
                f"label_map[{raw_label!r}] = {target!r} is not a canonical condition "
                f"(expected one of {sorted(CANONICAL_LABELS)})"
            )
    return problems


def validate_rows(df: pd.DataFrame) -> list[str]:
    """Return schema problems with a row set: missing columns, null/empty
    required fields, or a condition outside the canonical 16 classes."""
    problems = []
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        problems.append(f"missing required column(s): {missing}")
        return problems  # further checks would KeyError

    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        problems.append("rows with null/empty `text`")
    if df["row_id"].isna().any():
        problems.append("rows with null `row_id`")
    if df["row_id"].duplicated().any():
        dup_count = int(df["row_id"].duplicated().sum())
        problems.append(f"{dup_count} row(s) share a `row_id` with another row")

    unknown = set(df["condition"].astype(str)) - CANONICAL_LABELS
    if unknown:
        problems.append(f"condition(s) outside the canonical 16 classes: {sorted(unknown)}")

    return problems


def require_valid_rows(df: pd.DataFrame) -> None:
    problems = validate_rows(df)
    if problems:
        raise DatasetValidationError("Dataset row validation failed:\n" + "\n".join(problems))
