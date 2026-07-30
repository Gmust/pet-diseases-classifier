import json
from pathlib import Path

import pytest

from app.ml.model_registry import (
    MANIFEST_NAME,
    publish_bundle,
    release_version,
    retrieve_bundle,
    verify_release_manifest,
    write_channel_metadata,
)
from app.ml.model_validation import ModelValidationError


def _bundle(path: Path, content: str = "weights") -> Path:
    path.mkdir()
    (path / "model.onnx").write_text(content, encoding="utf-8")
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def test_publish_and_retrieve_immutable_bundle(tmp_path: Path) -> None:
    source = _bundle(tmp_path / "source")
    registry = tmp_path / "registry"

    published = publish_bundle(source, registry, "2026.07.07-1")
    manifest = verify_release_manifest(published)
    retrieved = retrieve_bundle(registry, "2026.07.07-1", tmp_path / "retrieved")

    assert manifest["retention"]["minimum_versions"] == 5
    assert release_version(published) == "2026.07.07-1"
    assert manifest["rollback"]["previous_version"] is None
    assert (retrieved / "model.onnx").read_text(encoding="utf-8") == "weights"
    with pytest.raises(FileExistsError, match="already exists"):
        publish_bundle(source, registry, "2026.07.07-1")


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    published = publish_bundle(_bundle(tmp_path / "source"), tmp_path / "registry", "v1")
    (published / "model.onnx").write_text("changed", encoding="utf-8")

    with pytest.raises(ModelValidationError, match="mismatch"):
        verify_release_manifest(published)


def test_manifest_rejects_unlisted_payload(tmp_path: Path) -> None:
    published = publish_bundle(_bundle(tmp_path / "source"), tmp_path / "registry", "v1")
    (published / "model_quantized.onnx").write_text("unverified", encoding="utf-8")

    with pytest.raises(ModelValidationError, match="absent from manifest"):
        verify_release_manifest(published)


def test_channel_records_rollback_version(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    publish_bundle(_bundle(tmp_path / "one", "one"), registry, "v1")
    publish_bundle(_bundle(tmp_path / "two", "two"), registry, "v2", previous_version="v1")

    write_channel_metadata(registry, "production", "v1")
    channel_path = write_channel_metadata(registry, "production", "v2")
    channel = json.loads(channel_path.read_text(encoding="utf-8"))

    assert channel["current_version"] == "v2"
    assert channel["rollback_version"] == "v1"
    assert json.loads((registry / "releases" / "v2" / MANIFEST_NAME).read_text())["rollback"] == {
        "previous_version": "v1"
    }
