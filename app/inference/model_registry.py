"""Immutable filesystem-backed model bundle registry.

The adapter deliberately uses a small storage contract so a release can be
copied to object storage without changing its manifest format. Published
version directories are write-once; mutable channel state only names the
current and rollback versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.inference.model_validation import ModelValidationError

MANIFEST_NAME = "release-manifest.json"
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class RetentionPolicy:
    """Retention intent recorded with each release, enforced by store policy."""

    minimum_versions: int = 5
    minimum_days: int = 90
    protect_rollback: bool = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_version(version: str) -> None:
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("Model version must be a safe 1-128 character release identifier.")


def build_release_manifest(
    bundle_dir: Path,
    version: str,
    *,
    previous_version: str | None = None,
    retention: RetentionPolicy = RetentionPolicy(),
) -> dict[str, Any]:
    """Describe every regular payload file using stable relative paths."""
    _validate_version(version)
    if previous_version is not None:
        _validate_version(previous_version)
        if previous_version == version:
            raise ValueError("previous_version must differ from version")
    if retention.minimum_versions < 2 or retention.minimum_days < 1:
        raise ValueError("Retention must preserve at least two versions for at least one day.")
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Model bundle does not exist: {bundle_dir}")

    files: dict[str, dict[str, int | str]] = {}
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_symlink():
            raise ModelValidationError(f"Model bundles may not contain symlinks: {path}")
        if path.is_file() and path != bundle_dir / MANIFEST_NAME:
            relative = path.relative_to(bundle_dir).as_posix()
            files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    if not files:
        raise ModelValidationError("Model bundle contains no payload files.")

    return {
        "schema_version": 1,
        "model_version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "files": files,
        "rollback": {"previous_version": previous_version},
        "retention": asdict(retention),
    }


def verify_release_manifest(bundle_dir: Path) -> dict[str, Any]:
    """Verify manifest compatibility, version, sizes, and all payload hashes."""
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ModelValidationError(f"Model release is missing {MANIFEST_NAME}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ModelValidationError(f"Invalid {MANIFEST_NAME}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ModelValidationError("Unsupported model release manifest schema.")
    version = manifest.get("model_version")
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ModelValidationError("Invalid model_version in release manifest.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ModelValidationError("Release manifest has no files.")

    failures: list[str] = []
    actual_files: set[str] = set()
    for path in bundle_dir.rglob("*"):
        if path.is_symlink():
            failures.append(f"{path.relative_to(bundle_dir).as_posix()}: symlink not allowed")
        elif path.is_file() and path != manifest_path:
            actual_files.add(path.relative_to(bundle_dir).as_posix())
    declared_files = {relative for relative in files if isinstance(relative, str)}
    for relative in sorted(actual_files - declared_files):
        failures.append(f"{relative}: file absent from manifest")
    for relative in sorted(declared_files - actual_files):
        failures.append(f"{relative}: declared file missing")
    for relative, metadata in files.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            failures.append(f"unsafe path: {relative}")
            continue
        target = bundle_dir / relative
        if not target.is_file() or target.is_symlink():
            failures.append(f"{relative}: file missing or not regular")
            continue
        if not isinstance(metadata, dict):
            failures.append(f"{relative}: invalid metadata")
            continue
        if target.stat().st_size != metadata.get("size_bytes"):
            failures.append(f"{relative}: size mismatch")
        elif _sha256(target) != metadata.get("sha256"):
            failures.append(f"{relative}: checksum mismatch")
    if failures:
        raise ModelValidationError("Release manifest verification failed: " + "; ".join(failures))
    return manifest


def release_version(bundle_dir: Path) -> str:
    """Return the declared release version, or an explicit local-dev sentinel.

    Reads model_version from the manifest without re-hashing payload files —
    full verification already ran at build/publish time, and re-hashing the
    bundle (tens of MB) on every process start blows Lambda's fixed 10s INIT
    budget. Use verify_release_manifest directly when integrity must be checked.
    """
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return "unversioned-local"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ModelValidationError(f"Invalid {MANIFEST_NAME}: {exc}") from exc
    version = manifest.get("model_version")
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ModelValidationError("Invalid model_version in release manifest.")
    return version


def publish_bundle(
    bundle_dir: Path,
    registry_dir: Path,
    version: str,
    *,
    previous_version: str | None = None,
    retention: RetentionPolicy = RetentionPolicy(),
) -> Path:
    """Publish a write-once version directory; existing releases never mutate."""
    manifest = build_release_manifest(
        bundle_dir, version, previous_version=previous_version, retention=retention
    )
    destination = registry_dir / "releases" / version
    if destination.exists():
        raise FileExistsError(f"Immutable model release already exists: {version}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle_dir, destination, symlinks=False)
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_release_manifest(destination)
    return destination


def retrieve_bundle(registry_dir: Path, version: str, destination: Path) -> Path:
    """Retrieve one exact immutable version and verify it before returning."""
    _validate_version(version)
    source = registry_dir / "releases" / version
    verify_release_manifest(source)
    if destination.exists():
        raise FileExistsError(f"Retrieval destination already exists: {destination}")
    shutil.copytree(source, destination, symlinks=False)
    verify_release_manifest(destination)
    return destination


def write_channel_metadata(registry_dir: Path, channel: str, version: str) -> Path:
    """Atomically update a mutable channel pointer with explicit rollback metadata."""
    if not VERSION_PATTERN.fullmatch(channel):
        raise ValueError("Channel must be a safe release identifier.")
    release = registry_dir / "releases" / version
    verify_release_manifest(release)
    channels_dir = registry_dir / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    channel_path = channels_dir / f"{channel}.json"
    previous: str | None = None
    if channel_path.exists():
        previous_data = json.loads(channel_path.read_text(encoding="utf-8"))
        previous = previous_data.get("current_version")
    payload = {
        "schema_version": 1,
        "channel": channel,
        "current_version": version,
        "rollback_version": previous,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    temporary = channel_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(channel_path)
    return channel_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an immutable model release bundle.")
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()
    manifest = verify_release_manifest(args.bundle_dir)
    print(json.dumps({"status": "valid", "model_version": manifest["model_version"]}))


if __name__ == "__main__":  # pragma: no cover - exercised by container builds
    main()
