"""
Machine-readable metrics, bundle validation, and a generated model card —
emitted by train.py after every training run.

Per "Reproducible training bundle" (specs/reproducible-ml-lifecycle/spec.md):
a completed training run must produce a self-describing bundle (weights +
tokenizer + run_metadata.json + metrics.json + MODEL_CARD.md) that passes
bundle validation before it is considered done.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.inference.model_validation import (
    ModelValidationError,
    validate_id2label,
    validate_required_files,
)

BUNDLE_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "run_metadata.json",
    "metrics.json",
    "MODEL_CARD.md",
)


def build_metrics(
    test_labels: list[int],
    test_preds: list[int],
    label_names: list[str],
) -> dict[str, Any]:
    """Per-class precision/recall/f1/support, accuracy, and a confusion
    matrix — the machine-readable counterpart to the printed classification
    report, for downstream tooling (release gates, dashboards) to consume
    without re-parsing stdout."""
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(
        test_labels, test_preds, target_names=label_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(len(label_names))))
    return {
        "labels": label_names,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def write_metrics(metrics: dict[str, Any], model_dir: str) -> Path:
    path = Path(model_dir) / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def render_model_card(
    metrics: dict[str, Any],
    run_metadata: dict[str, Any],
) -> str:
    report = metrics["classification_report"]
    label_names = metrics["labels"]
    accuracy = report.get("accuracy", 0.0)
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)

    lines = [
        "# Model Card",
        "",
        "This is a machine-generated pre-assessment classifier for pet-owner symptom "
        "text. **It is not a diagnostic or clinically validated tool.** Predictions must "
        "be paired with the deterministic safety layer (`app/services/triage_safety.py`) "
        "and are not a substitute for veterinary care.",
        "",
        "## Provenance",
        "",
        f"- Base model: `{run_metadata['config'].get('base_model', 'unknown')}`",
        f"- Commit: `{run_metadata.get('commit') or 'unknown'}`",
        f"- Data fingerprint: `{run_metadata.get('data_fingerprint') or 'unknown'}`",
        f"- Split sizes: {run_metadata.get('split_sizes', {})}",
        f"- Random seed: {run_metadata['config'].get('random_state', 'unknown')}",
        "",
        "## Test-set summary",
        "",
        f"- Accuracy: {accuracy:.3f}",
        f"- Macro F1: {macro_f1:.3f}",
        "",
        "## Per-class metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for name in label_names:
        row = report.get(name, {})
        lines.append(
            f"| {name} | {row.get('precision', 0.0):.2f} | {row.get('recall', 0.0):.2f} | "
            f"{row.get('f1-score', 0.0):.2f} | {int(row.get('support', 0))} |"
        )
    lines += [
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(run_metadata.get("versions", {}), indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def write_model_card(text: str, model_dir: str) -> Path:
    path = Path(model_dir) / "MODEL_CARD.md"
    path.write_text(text, encoding="utf-8")
    return path


def validate_training_bundle(model_dir: str) -> None:
    """Fail loudly if the training run did not produce a complete,
    self-describing bundle: required files present, and (if the bundle
    already has an id2label in config.json) a well-formed label set."""
    model_path = Path(model_dir)
    validate_required_files(model_path, list(BUNDLE_REQUIRED_FILES))

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    id2label = config.get("id2label")
    if id2label is not None:
        validate_id2label({int(k): v for k, v in id2label.items()})

    metrics = json.loads((model_path / "metrics.json").read_text(encoding="utf-8"))
    if not metrics.get("labels"):
        raise ModelValidationError("metrics.json has no `labels` — bundle is incomplete.")
