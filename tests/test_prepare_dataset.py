"""Unit tests for the dataset-prep helpers. Skipped if pandas isn't installed."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")  # needs the training env (requirements-train.txt)

from ml_pipeline.dataset_schema import normalize_text  # noqa: E402
from ml_pipeline.prepare_dataset import (  # noqa: E402
    apply_label_map,
    carve_owner_eval,
    dedup,
    downsample,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["text", "condition", "record_type"])


def test_normalize_text_collapses_punctuation_and_space():
    assert normalize_text("  My DOG, vomiting!! ") == "my dog vomiting"


def test_dedup_removes_near_duplicates():
    df = _df(
        [
            ("My dog is vomiting", "Digestive Issues", "Owner Observation"),
            ("my dog is vomiting!", "Digestive Issues", "Owner Observation"),  # near-dup
            ("Cat sneezing a lot", "Respiratory Conditions", "Owner Observation"),
        ]
    )
    out = dedup(df)
    assert len(out) == 2


def test_apply_label_map_collapses_raw_labels():
    df = _df([("rash", "Diseases of the skin", "Clinical Notes")])
    out = apply_label_map(df, {"Diseases of the skin": "Skin Conditions"})
    assert out.iloc[0]["condition"] == "Skin Conditions"


def test_downsample_caps_majority_classes():
    rows = [
        (f"text big {i}", "Infectious and Parasitic Diseases", "External Dataset")
        for i in range(50)
    ]
    rows += [(f"text small {i}", "Blood Disorders", "External Dataset") for i in range(5)]
    out = downsample(_df(rows), max_per_class=10)
    counts = out["condition"].value_counts()
    assert counts["Infectious and Parasitic Diseases"] == 10
    assert counts["Blood Disorders"] == 5  # small class untouched


def test_owner_eval_holdout_is_disjoint_from_train():
    rows = [(f"owner says {i}", "Digestive Issues", "Owner Observation") for i in range(10)]
    rows += [(f"clinical note {i}", "Skin Conditions", "Clinical Notes") for i in range(10)]
    train, eval_df = carve_owner_eval(_df(rows), "Owner Observation", frac=0.5, seed=1)

    # Holdout is only owner rows, ~50% of them, and absent from train.
    assert len(eval_df) == 5
    assert (eval_df["record_type"] == "Owner Observation").all()
    overlap = set(eval_df["text"]) & set(train["text"])
    assert overlap == set()
    # Clinical rows all remain in train.
    assert (train["record_type"] == "Clinical Notes").sum() == 10
