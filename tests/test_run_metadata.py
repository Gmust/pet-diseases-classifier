"""Unit tests for training-run reproducibility metadata."""

from __future__ import annotations

import json

from ml_pipeline.run_metadata import (
    build_run_metadata,
    file_fingerprint,
    git_commit,
    package_versions,
    write_run_metadata,
)


def test_git_commit_returns_a_hash_in_this_repo():
    commit = git_commit()
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)


def test_package_versions_includes_python():
    versions = package_versions()
    assert "python" in versions and versions["python"]


def test_file_fingerprint_none_for_missing_path():
    assert file_fingerprint(None) is None
    assert file_fingerprint("/nonexistent/file.json") is None


def test_file_fingerprint_stable_for_same_content(tmp_path):
    path = tmp_path / "a.json"
    path.write_text('{"a": 1}')
    assert file_fingerprint(str(path)) == file_fingerprint(str(path))


def test_file_fingerprint_differs_for_different_content(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('{"a": 1}')
    b.write_text('{"a": 2}')
    assert file_fingerprint(str(a)) != file_fingerprint(str(b))


def test_build_run_metadata_includes_all_fields():
    metadata = build_run_metadata(
        config={"epochs": 6},
        label2id={"Digestive Issues": 0},
        split_sizes={"train": 100, "val": 10, "test": 15},
        data_path=None,
        label_map_path=None,
    )
    assert set(metadata) == {
        "commit",
        "versions",
        "config",
        "labels",
        "split_sizes",
        "data_fingerprint",
        "label_map_fingerprint",
    }
    assert metadata["config"] == {"epochs": 6}
    assert metadata["split_sizes"] == {"train": 100, "val": 10, "test": 15}


def test_write_run_metadata_writes_valid_json(tmp_path):
    metadata = build_run_metadata(
        config={}, label2id={}, split_sizes={}, data_path=None, label_map_path=None
    )
    path = write_run_metadata(metadata, str(tmp_path))
    assert path.name == "run_metadata.json"
    assert json.loads(path.read_text()) == metadata
