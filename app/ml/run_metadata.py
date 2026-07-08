"""
Reproducibility metadata emitted alongside every trained model bundle.

Answers "what exactly produced this model" without needing the training
run's terminal output: code revision, environment package versions, the
resolved training config, and fingerprints of the data/label-map artifacts
consumed. See "Reproducible training bundle" in
specs/reproducible-ml-lifecycle/spec.md.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for module_name in ("torch", "transformers", "pandas", "numpy", "sklearn"):
        try:
            versions[module_name] = __import__(module_name).__version__
        except Exception:
            versions[module_name] = "unavailable"
    return versions


def file_fingerprint(path: str | None) -> str | None:
    """Short sha256 of a file's contents, or None if the path is unset/missing."""
    if not path or not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def build_run_metadata(
    *,
    config: dict[str, Any],
    label2id: dict[str, int],
    split_sizes: dict[str, int],
    data_path: str | None,
    label_map_path: str | None,
) -> dict[str, Any]:
    return {
        "commit": git_commit(),
        "versions": package_versions(),
        "config": config,
        "labels": label2id,
        "split_sizes": split_sizes,
        "data_fingerprint": file_fingerprint(data_path),
        "label_map_fingerprint": file_fingerprint(label_map_path),
    }


def write_run_metadata(metadata: dict[str, Any], model_dir: str) -> Path:
    path = Path(model_dir) / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
