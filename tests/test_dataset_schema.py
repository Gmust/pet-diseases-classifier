"""Unit tests for the canonical dataset row contract. Skipped without pandas
(needs the training env, requirements-train.txt)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from ml_pipeline.dataset_schema import (  # noqa: E402
    DatasetValidationError,
    assign_row_ids,
    compute_row_id,
    require_valid_rows,
    validate_label_map,
    validate_rows,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["text", "condition", "record_type"])


def test_compute_row_id_is_stable_for_identical_content():
    a = compute_row_id("My dog is vomiting", "Digestive Issues", "local")
    b = compute_row_id("My dog is vomiting", "Digestive Issues", "local")
    assert a == b


def test_compute_row_id_ignores_case_and_punctuation():
    a = compute_row_id("My dog is vomiting!", "Digestive Issues", "local")
    b = compute_row_id("my dog is vomiting", "Digestive Issues", "local")
    assert a == b


def test_compute_row_id_differs_by_source_and_condition():
    base = compute_row_id("text", "Digestive Issues", "local")
    assert base != compute_row_id("text", "Digestive Issues", "vetpetcare")
    assert base != compute_row_id("text", "Skin Conditions", "local")


def test_assign_row_ids_adds_source_and_stable_ids():
    df = _df([("My dog is vomiting", "Digestive Issues", "Owner Observation")])
    out = assign_row_ids(df, source="local")
    assert (out["source"] == "local").all()
    assert out["row_id"].iloc[0] == compute_row_id(
        "My dog is vomiting", "Digestive Issues", "local"
    )


def test_validate_rows_passes_for_well_formed_canonical_rows():
    df = _df([("My dog is vomiting", "Digestive Issues", "Owner Observation")])
    df = assign_row_ids(df, source="local")
    assert validate_rows(df) == []


def test_validate_rows_flags_missing_columns():
    df = _df([("text", "Digestive Issues", "Owner Observation")]).drop(columns=["condition"])
    problems = validate_rows(df)
    assert any("missing required column" in p for p in problems)


def test_validate_rows_flags_non_canonical_condition():
    df = _df([("text", "Not A Real Condition", "Owner Observation")])
    df = assign_row_ids(df, source="local")
    problems = validate_rows(df)
    assert any("outside the canonical 16 classes" in p for p in problems)


def test_validate_rows_flags_duplicate_row_ids():
    df = _df(
        [
            ("same text", "Digestive Issues", "Owner Observation"),
            ("same text", "Digestive Issues", "Owner Observation"),
        ]
    )
    df = assign_row_ids(df, source="local")
    problems = validate_rows(df)
    assert any("share a `row_id`" in p for p in problems)


def test_require_valid_rows_raises_on_problems():
    df = _df([("text", "Not A Real Condition", "Owner Observation")])
    df = assign_row_ids(df, source="local")
    with pytest.raises(DatasetValidationError):
        require_valid_rows(df)


def test_validate_label_map_accepts_canonical_targets():
    assert validate_label_map({"Diseases of the skin": "Skin Conditions"}) == []


def test_validate_label_map_flags_non_canonical_target():
    problems = validate_label_map({"Diseases of the skin": "Skin Problems"})
    assert len(problems) == 1
    assert "Skin Problems" in problems[0]
