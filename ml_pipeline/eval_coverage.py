"""
Owner-language evaluation coverage gate.

Per "Calibrated release evaluation" (specs/reproducible-ml-lifecycle/spec.md):
a release evaluation set with zero examples for a class the product claims to
support must fail the release, not silently average that class away in a
macro/weighted metric.

Species coverage is checked the same way when a `species` column is present.
Today's owner-language holdout does not carry species labels (see design.md
open question "Which full-class owner-language dataset can support release
claims without synthetic-only evaluation?"), so a species-blind holdout is
reported as a named gap (`CoverageReport.species_checked = False`) rather
than silently skipped or treated as passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.domain.enums import SUPPORTED_PET_TYPES
from ml_pipeline.dataset_schema import CANONICAL_LABELS


class EvaluationCoverageError(Exception):
    """Raised when a release evaluation set is missing required class coverage."""


@dataclass
class CoverageReport:
    missing_classes: list[str] = field(default_factory=list)
    missing_species: list[str] = field(default_factory=list)
    species_checked: bool = False

    @property
    def ok(self) -> bool:
        return not self.missing_classes and (not self.species_checked or not self.missing_species)


def check_class_coverage(
    eval_df: pd.DataFrame, required_labels: frozenset[str] = CANONICAL_LABELS
) -> list[str]:
    present = set(eval_df["condition"].astype(str))
    return sorted(set(required_labels) - present)


def check_species_coverage(eval_df: pd.DataFrame) -> list[str]:
    """Returns [] if a `species` column is absent — see module docstring;
    callers should treat `CoverageReport.species_checked = False` as its own
    gap, not as "species coverage passed"."""
    if "species" not in eval_df.columns:
        return []
    required = {pet_type.value for pet_type in SUPPORTED_PET_TYPES}
    present = set(eval_df["species"].astype(str).str.lower())
    return sorted(required - present)


def build_coverage_report(eval_df: pd.DataFrame) -> CoverageReport:
    species_checked = "species" in eval_df.columns
    return CoverageReport(
        missing_classes=check_class_coverage(eval_df),
        missing_species=check_species_coverage(eval_df) if species_checked else [],
        species_checked=species_checked,
    )


def require_full_coverage(eval_df: pd.DataFrame) -> CoverageReport:
    """Raise `EvaluationCoverageError` if any claimed class has zero eval rows.

    Missing species coverage does not raise — species labels don't exist
    end-to-end in the pipeline yet — but it IS returned on the report so
    callers can surface the gap rather than claim full coverage.
    """
    report = build_coverage_report(eval_df)
    if report.missing_classes:
        raise EvaluationCoverageError(
            f"{len(report.missing_classes)} product-claimed class(es) have zero owner-language "
            f"evaluation rows: {report.missing_classes}"
        )
    return report
