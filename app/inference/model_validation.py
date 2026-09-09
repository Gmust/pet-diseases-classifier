"""
Model bundle validation — checked before a model is trusted for inference.

Catches missing/renamed files, an empty or malformed label map, and (when a
manifest is present) artifact checksum drift. The manifest is optional so
today's undocumented model directories keep loading; once model releases
publish one (task 6.1) checksum verification stops being a no-op for them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ModelValidationError(Exception):
    """Raised when a model bundle fails validation and must not be loaded."""


def validate_required_files(model_dir: Path, required: list[str]) -> None:
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise ModelValidationError(
            f"Model directory {model_dir} is missing required file(s): {', '.join(missing)}"
        )


def validate_id2label(id2label: dict[int, str]) -> None:
    if not id2label:
        raise ModelValidationError("Model config has an empty id2label mapping.")
    expected_ids = set(range(len(id2label)))
    if set(id2label.keys()) != expected_ids:
        raise ModelValidationError(
            f"id2label keys must be a contiguous 0..N-1 range; got {sorted(id2label.keys())}"
        )
    labels = list(id2label.values())
    if len(set(labels)) != len(labels):
        raise ModelValidationError(f"id2label contains duplicate labels: {labels}")
    if any(not str(label).strip() for label in labels):
        raise ModelValidationError("id2label contains a blank label.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(model_dir: Path, manifest_name: str = "manifest.json") -> None:
    """Verify file checksums against `manifest.json` if one is present.

    No-op when no manifest exists — checksum verification is best-effort until
    model publishing (task 6.1) makes a manifest mandatory for every release.
    """
    manifest_path = model_dir / manifest_name
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums: dict[str, str] = manifest.get("sha256", {})
    mismatches: list[str] = []
    for filename, expected in checksums.items():
        target = model_dir / filename
        if not target.exists():
            mismatches.append(f"{filename}: file missing")
            continue
        actual = _sha256(target)
        if actual != expected:
            mismatches.append(f"{filename}: expected {expected}, got {actual}")
    if mismatches:
        raise ModelValidationError(
            f"Checksum verification failed for {model_dir}: " + "; ".join(mismatches)
        )


def validate_top_k(k: int, num_labels: int) -> int:
    """Validate `k` for predict_top_k and clamp it to the label count."""
    if k < 1:
        raise ValueError(f"top-k must be >= 1, got {k}")
    return min(k, num_labels)
