"""Unit tests for validated training configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ml.training_config import TrainingConfig


def test_defaults_are_valid():
    TrainingConfig()  # no raise


def test_rejects_non_positive_epochs():
    with pytest.raises(ValidationError):
        TrainingConfig(epochs=0)


def test_rejects_non_positive_learning_rate():
    with pytest.raises(ValidationError):
        TrainingConfig(lr=0)


def test_rejects_unknown_loss():
    with pytest.raises(ValidationError):
        TrainingConfig(loss="mse")


def test_rejects_split_budget_over_one():
    with pytest.raises(ValidationError, match="must sum to < 1.0"):
        TrainingConfig(test_size=0.6, val_size=0.5)


def test_accepts_split_budget_under_one():
    TrainingConfig(test_size=0.15, val_size=0.1)  # no raise


def test_rejects_max_length_out_of_bounds():
    with pytest.raises(ValidationError):
        TrainingConfig(max_length=0)
    with pytest.raises(ValidationError):
        TrainingConfig(max_length=1000)
