"""Unit tests for synthetic-data quality gates. Skipped without pandas
(needs the training env, requirements-train.txt)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from ml_pipeline.synthetic_quality import (  # noqa: E402
    apply_quality_gates,
    detect_diagnosis_leakage,
    find_near_duplicates,
    quality_summary,
    tag_provenance,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["text", "condition"])


def test_detect_diagnosis_leakage_flags_label_verbatim_in_text():
    assert detect_diagnosis_leakage(
        "My dog has blood disorders and is lethargic.", "Blood Disorders"
    )


def test_detect_diagnosis_leakage_ignores_case_and_punctuation():
    assert detect_diagnosis_leakage("BLOOD, DISORDERS! present.", "Blood Disorders")


def test_detect_diagnosis_leakage_false_for_clean_symptom_text():
    assert not detect_diagnosis_leakage("My dog is lethargic and has pale gums.", "Blood Disorders")


def test_tag_provenance_adds_model_and_timestamp():
    df = _df([("text", "Blood Disorders")])
    out = tag_provenance(df, model_name="gemini-2.5-flash")
    assert (out["generated_by"] == "gemini-2.5-flash").all()
    assert out["generated_at"].iloc[0]  # non-empty ISO timestamp


def test_find_near_duplicates_flags_similar_rows_within_same_class():
    df = _df(
        [
            ("My dog has been vomiting since yesterday morning", "Digestive Issues"),
            ("My dog has been vomiting since yesterday evening", "Digestive Issues"),
            ("My cat is sneezing a lot lately", "Respiratory Conditions"),
        ]
    )
    flags = find_near_duplicates(df, threshold=0.85)
    assert len(flags) == 1
    assert {flags[0].row_index, flags[0].similar_to_index} == {0, 1}


def test_find_near_duplicates_does_not_cross_classes():
    df = _df(
        [
            ("identical wording here", "Digestive Issues"),
            ("identical wording here", "Skin Conditions"),
        ]
    )
    # Same text, different class — grouped separately, so no pairwise comparison.
    assert find_near_duplicates(df, threshold=0.85) == []


def test_apply_quality_gates_marks_needs_review_for_leaked_diagnosis():
    df = _df([("This is a case of blood disorders.", "Blood Disorders")])
    out = apply_quality_gates(df)
    assert out["leaks_diagnosis"].iloc[0] is True or bool(out["leaks_diagnosis"].iloc[0])
    assert bool(out["needs_review"].iloc[0])


def test_apply_quality_gates_leaves_clean_rows_unflagged():
    df = _df([("My dog is lethargic with pale gums.", "Blood Disorders")])
    out = apply_quality_gates(df)
    assert not bool(out["leaks_diagnosis"].iloc[0])
    assert not bool(out["near_duplicate"].iloc[0])
    assert not bool(out["needs_review"].iloc[0])


def test_quality_summary_counts_flags():
    df = _df(
        [
            ("This is a case of blood disorders.", "Blood Disorders"),
            ("My dog is lethargic with pale gums.", "Blood Disorders"),
        ]
    )
    out = apply_quality_gates(df)
    summary = quality_summary(out)
    assert summary["total_rows"] == 2
    assert summary["leaks_diagnosis"] == 1
    assert summary["needs_review"] == 1
