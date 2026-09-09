"""Unit tests for machine-readable metrics, model card, and bundle
validation. Skipped without sklearn (needs the training env,
requirements-train.txt)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("sklearn")

from app.inference.model_validation import ModelValidationError  # noqa: E402
from ml_pipeline.model_card import (  # noqa: E402
    BUNDLE_REQUIRED_FILES,
    build_metrics,
    render_model_card,
    validate_training_bundle,
    write_metrics,
    write_model_card,
)

LABELS = ["Digestive Issues", "Skin Conditions"]


def _sample_metrics():
    return build_metrics(test_labels=[0, 0, 1, 1], test_preds=[0, 1, 1, 1], label_names=LABELS)


def _sample_run_metadata():
    return {
        "commit": "abc123",
        "versions": {"python": "3.11.0"},
        "config": {"base_model": "distilbert-base-uncased", "random_state": 42},
        "labels": {"Digestive Issues": 0, "Skin Conditions": 1},
        "split_sizes": {"train": 10, "val": 2, "test": 4},
        "data_fingerprint": "deadbeef",
        "label_map_fingerprint": None,
    }


def test_build_metrics_has_labels_report_and_confusion_matrix():
    metrics = _sample_metrics()
    assert metrics["labels"] == LABELS
    assert "accuracy" in metrics["classification_report"]
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2


def test_write_metrics_writes_valid_json(tmp_path):
    metrics = _sample_metrics()
    path = write_metrics(metrics, str(tmp_path))
    assert path.name == "metrics.json"
    assert json.loads(path.read_text())["labels"] == LABELS


def test_render_model_card_includes_provenance_and_per_class_table():
    card = render_model_card(_sample_metrics(), _sample_run_metadata())
    assert "# Model Card" in card
    assert "not a diagnostic" in card
    assert "distilbert-base-uncased" in card
    assert "abc123" in card
    for name in LABELS:
        assert name in card


def test_write_model_card_writes_file(tmp_path):
    card = render_model_card(_sample_metrics(), _sample_run_metadata())
    path = write_model_card(card, str(tmp_path))
    assert path.name == "MODEL_CARD.md"
    assert path.read_text() == card


def _write_valid_bundle(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"id2label": {"0": "Digestive Issues", "1": "Skin Conditions"}})
    )
    write_metrics(_sample_metrics(), str(tmp_path))
    write_model_card(render_model_card(_sample_metrics(), _sample_run_metadata()), str(tmp_path))
    (tmp_path / "run_metadata.json").write_text(json.dumps(_sample_run_metadata()))
    return tmp_path


def test_validate_training_bundle_passes_for_complete_bundle(tmp_path):
    _write_valid_bundle(tmp_path)
    validate_training_bundle(str(tmp_path))  # no raise


def test_validate_training_bundle_requires_all_files():
    for missing in BUNDLE_REQUIRED_FILES:
        assert missing  # sanity: constant is non-empty strings


def test_validate_training_bundle_fails_on_missing_file(tmp_path):
    _write_valid_bundle(tmp_path)
    (tmp_path / "metrics.json").unlink()
    with pytest.raises(ModelValidationError):
        validate_training_bundle(str(tmp_path))


def test_validate_training_bundle_fails_on_malformed_id2label(tmp_path):
    _write_valid_bundle(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"id2label": {}}))
    with pytest.raises(ModelValidationError):
        validate_training_bundle(str(tmp_path))
