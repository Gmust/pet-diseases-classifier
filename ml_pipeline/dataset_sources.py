"""
Source-adapter contract for fetch_and_merge.py.

Each external/local input (local base file, VetPetCare, PetEVAL, synthetic
data) is loaded through `load_source`, which:

  1. Tags every row with stable provenance (`source`, `row_id`) via
     `ml_pipeline.dataset_schema.assign_row_ids`.
  2. Records what happened — rows loaded, pinned revision, failure reason —
     in a `SourceResult`, which the caller can serialize into a build
     manifest (see `Canonical and traceable dataset builds` in
     specs/reproducible-ml-lifecycle/spec.md).
  3. Enforces the source's mode: a STRICT source that fails aborts the whole
     build (`DatasetSourceError`); a BEST_EFFORT source that fails is
     recorded and skipped, matching the pipeline's historical lenient
     default for optional/gated sources.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum

import pandas as pd

from ml_pipeline.dataset_schema import assign_row_ids


class SourceMode(StrEnum):
    STRICT = "strict"
    BEST_EFFORT = "best_effort"


class DatasetSourceError(Exception):
    """Raised when a STRICT source fails to load."""


@dataclass
class SourceResult:
    name: str
    mode: str
    revision: str | None
    rows: int
    loaded: bool
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def load_source(
    name: str,
    mode: SourceMode,
    loader: Callable[[], pd.DataFrame],
    *,
    revision: str | None = None,
) -> tuple[pd.DataFrame, SourceResult]:
    """Run `loader()` (a zero-arg callable returning a text/condition/record_type
    DataFrame), tag rows with provenance, and report what happened."""
    try:
        df = loader()
    except Exception as exc:
        if mode is SourceMode.STRICT:
            raise DatasetSourceError(f"required source {name!r} failed to load: {exc}") from exc
        return pd.DataFrame(columns=["text", "condition", "record_type"]), SourceResult(
            name=name, mode=mode.value, revision=revision, rows=0, loaded=False, error=str(exc)
        )

    if df.empty:
        if mode is SourceMode.STRICT:
            raise DatasetSourceError(f"required source {name!r} loaded zero rows")
        return df, SourceResult(name=name, mode=mode.value, revision=revision, rows=0, loaded=False)

    tagged = assign_row_ids(df, source=name)
    return tagged, SourceResult(
        name=name, mode=mode.value, revision=revision, rows=len(tagged), loaded=True
    )
