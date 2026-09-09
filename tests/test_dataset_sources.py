"""Unit tests for source-adapter loading (strict vs best-effort). Skipped
without pandas (needs the training env, requirements-train.txt)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from ml_pipeline.dataset_sources import (  # noqa: E402
    DatasetSourceError,
    SourceMode,
    load_source,
)


def _ok_loader():
    return pd.DataFrame(
        [{"text": "My dog is vomiting", "condition": "Digestive Issues", "record_type": "Test"}]
    )


def _empty_loader():
    return pd.DataFrame(columns=["text", "condition", "record_type"])


def _failing_loader():
    raise RuntimeError("network unavailable")


def test_load_source_tags_rows_with_provenance():
    df, result = load_source("local", SourceMode.BEST_EFFORT, _ok_loader)
    assert (df["source"] == "local").all()
    assert df["row_id"].notna().all()
    assert result.loaded is True
    assert result.rows == 1


def test_load_source_best_effort_records_empty_result_without_raising():
    df, result = load_source("vetpetcare", SourceMode.BEST_EFFORT, _empty_loader)
    assert df.empty
    assert result.loaded is False
    assert result.error is None


def test_load_source_best_effort_swallows_failure():
    df, result = load_source("peteval", SourceMode.BEST_EFFORT, _failing_loader)
    assert df.empty
    assert result.loaded is False
    assert "network unavailable" in result.error


def test_load_source_strict_raises_on_failure():
    with pytest.raises(DatasetSourceError, match="required source 'peteval' failed"):
        load_source("peteval", SourceMode.STRICT, _failing_loader)


def test_load_source_strict_raises_on_empty_result_without_exception():
    # A source can "fail" by returning nothing (e.g. file not found, printed
    # and swallowed internally) without raising — strict mode must still catch it.
    with pytest.raises(DatasetSourceError, match="loaded zero rows"):
        load_source("local", SourceMode.STRICT, _empty_loader)


def test_load_source_strict_succeeds_normally():
    df, result = load_source("local", SourceMode.STRICT, _ok_loader, revision="v1")
    assert result.loaded is True
    assert result.revision == "v1"


def test_source_result_as_dict_is_json_serializable():
    import json

    _, result = load_source("local", SourceMode.BEST_EFFORT, _ok_loader)
    json.dumps(result.as_dict())  # no raise
