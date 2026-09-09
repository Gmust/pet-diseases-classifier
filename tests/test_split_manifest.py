"""Unit tests for split manifests and cross-split leakage detection. Skipped
without pandas (needs the training env, requirements-train.txt)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from ml_pipeline.dataset_schema import assign_row_ids  # noqa: E402
from ml_pipeline.split_manifest import (  # noqa: E402
    SplitLeakageError,
    build_split_manifest,
    duplicate_family_key,
    verify_no_cross_split_leakage,
    write_split_manifest,
)


def _df(rows):
    return pd.DataFrame(rows, columns=["text", "condition"])


def test_duplicate_family_key_ignores_case_and_punctuation():
    a = duplicate_family_key("My dog is vomiting!", "Digestive Issues")
    b = duplicate_family_key("my dog is vomiting", "Digestive Issues")
    assert a == b


def test_verify_no_cross_split_leakage_passes_for_disjoint_splits():
    train = _df([("My dog is vomiting", "Digestive Issues")])
    eval_ = _df([("My cat is sneezing", "Respiratory Conditions")])
    verify_no_cross_split_leakage({"train": train, "eval": eval_})  # no raise


def test_verify_no_cross_split_leakage_raises_for_shared_family():
    train = _df([("My dog is vomiting", "Digestive Issues")])
    eval_ = _df([("my dog is vomiting!", "Digestive Issues")])  # same family
    with pytest.raises(SplitLeakageError, match="cross split boundaries"):
        verify_no_cross_split_leakage({"train": train, "eval": eval_})


def test_verify_no_cross_split_leakage_catches_family_from_different_source():
    # Same duplicate-text family, different source/record_type — must still
    # be caught, since row_id alone (which encodes source) wouldn't collide.
    train = _df([("My dog is vomiting", "Digestive Issues")])
    eval_ = _df([("My dog is vomiting", "Digestive Issues")])
    with pytest.raises(SplitLeakageError):
        verify_no_cross_split_leakage({"train": train, "eval": eval_})


def test_build_split_manifest_requires_row_id():
    train = _df([("text", "Digestive Issues")])
    with pytest.raises(ValueError, match="missing `row_id`"):
        build_split_manifest({"train": train})


def test_build_split_manifest_lists_row_ids_per_split():
    train = assign_row_ids(_df([("text a", "Digestive Issues")]), source="train")
    eval_ = assign_row_ids(_df([("text b", "Skin Conditions")]), source="eval")
    manifest = build_split_manifest({"train": train, "eval": eval_})
    assert manifest["train"] == [train["row_id"].iloc[0]]
    assert manifest["eval"] == [eval_["row_id"].iloc[0]]


def test_write_split_manifest_writes_json(tmp_path):
    train = assign_row_ids(_df([("text a", "Digestive Issues")]), source="train")
    eval_ = assign_row_ids(_df([("text b", "Skin Conditions")]), source="eval")
    path = tmp_path / "splits.json"
    write_split_manifest({"train": train, "eval": eval_}, path)

    import json

    manifest = json.loads(path.read_text())
    assert manifest["train"] == [train["row_id"].iloc[0]]


def test_write_split_manifest_does_not_write_on_leakage(tmp_path):
    train = assign_row_ids(_df([("My dog is vomiting", "Digestive Issues")]), source="train")
    eval_ = assign_row_ids(_df([("My dog is vomiting", "Digestive Issues")]), source="eval")
    path = tmp_path / "splits.json"
    with pytest.raises(SplitLeakageError):
        write_split_manifest({"train": train, "eval": eval_}, path)
    assert not path.exists()
