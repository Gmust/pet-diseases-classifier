"""Unit tests for owner-language evaluation coverage gating. Skipped without
pandas (needs the training env, requirements-train.txt)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from ml_pipeline.dataset_schema import CANONICAL_LABELS  # noqa: E402
from ml_pipeline.eval_coverage import (  # noqa: E402
    EvaluationCoverageError,
    build_coverage_report,
    check_class_coverage,
    check_species_coverage,
    require_full_coverage,
)


def _df(rows, columns=("text", "condition")):
    return pd.DataFrame(rows, columns=list(columns))


def test_check_class_coverage_reports_missing_classes():
    df = _df([("t", "Digestive Issues")])
    missing = check_class_coverage(df)
    assert "Digestive Issues" not in missing
    assert len(missing) == len(CANONICAL_LABELS) - 1


def test_check_class_coverage_empty_when_all_present():
    df = _df([(f"t{i}", label) for i, label in enumerate(CANONICAL_LABELS)])
    assert check_class_coverage(df) == []


def test_check_species_coverage_returns_empty_without_species_column():
    df = _df([("t", "Digestive Issues")])
    assert check_species_coverage(df) == []


def test_check_species_coverage_flags_missing_species():
    df = _df([("t", "Digestive Issues", "dog")], columns=("text", "condition", "species"))
    missing = check_species_coverage(df)
    assert "dog" not in missing
    assert "cat" in missing


def test_build_coverage_report_marks_species_checked_false_without_column():
    df = _df([("t", "Digestive Issues")])
    report = build_coverage_report(df)
    assert report.species_checked is False
    assert report.missing_species == []


def test_build_coverage_report_ok_requires_class_coverage():
    full = _df([(f"t{i}", label) for i, label in enumerate(CANONICAL_LABELS)])
    assert build_coverage_report(full).ok is True

    partial = _df([("t", "Digestive Issues")])
    assert build_coverage_report(partial).ok is False


def test_require_full_coverage_raises_on_missing_classes():
    df = _df([("t", "Digestive Issues")])
    with pytest.raises(EvaluationCoverageError, match="product-claimed class"):
        require_full_coverage(df)


def test_require_full_coverage_passes_when_all_classes_present():
    df = _df([(f"t{i}", label) for i, label in enumerate(CANONICAL_LABELS)])
    report = require_full_coverage(df)  # no raise
    assert report.missing_classes == []
