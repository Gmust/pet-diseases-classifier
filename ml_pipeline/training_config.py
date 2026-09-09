"""
Validated training configuration for train.py.

Single source of truth for what a "valid" training config is. Before this,
train.py accepted any argparse combination (e.g. `test_size + val_size >= 1`,
`epochs <= 0`) and would fail deep inside sklearn/torch with a confusing
error instead of failing fast at the config boundary.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TrainingConfig(BaseModel):
    data_path: str = "data/merged_pet_dataset.parquet"
    hf_dataset: str | None = None
    hf_split: str = "train"
    label_map_path: str | None = None
    min_samples: int = Field(default=20, ge=1)

    base_model: str = "distilbert-base-uncased"
    model_dir: str = "models/transformer_model"

    epochs: int = Field(default=6, ge=1)
    batch_size: int = Field(default=16, ge=1)
    lr: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    max_length: int = Field(default=256, ge=1, le=512)
    warmup_ratio: float = Field(default=0.1, ge=0, le=1)
    loss: Literal["ce", "focal"] = "ce"
    focal_gamma: float = Field(default=2.0, ge=0)
    test_size: float = Field(default=0.15, gt=0, lt=1)
    val_size: float = Field(default=0.1, gt=0, lt=1)
    patience: int = Field(default=2, ge=1)
    random_state: int = 42

    @model_validator(mode="after")
    def _validate_split_budget(self) -> TrainingConfig:
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(
                f"test_size ({self.test_size}) + val_size ({self.val_size}) must sum to < 1.0"
            )
        return self
