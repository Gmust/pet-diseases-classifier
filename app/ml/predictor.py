from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class PredictionResult:
    predicted_condition: str
    confidence: float


class Predictor:
    """
    Wraps a fine-tuned HuggingFace transformer for pet-condition classification.

    The model directory is produced by app/ml/train.py and contains:
      - config.json         (includes id2label mapping)
      - model.safetensors   (or pytorch_model.bin)
      - tokenizer files
    """

    _MAX_LENGTH = 256

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        id2label: dict[int, str],
        device: torch.device,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._id2label = id2label
        self._device = device

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_paths(cls, model_path: str, **_ignored) -> "Predictor":
        """
        Load from a directory saved by train.py (HuggingFace format).

        The `**_ignored` signature intentionally swallows legacy keyword
        arguments such as `classifier_path` and `vectorizer_path` so that
        callers migrating from the old TF-IDF predictor need only change
        the `model_path` value — no other code changes required.
        """
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}\n"
                "Run training first:\n"
                "  python -m app.ml.train "
                "--data-path data/merged_pet_dataset.parquet "
                "--label-map data/label_map.json "
                f"--model-dir {model_dir}"
            )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.to(device)
        model.eval()

        # id2label is stored in model.config by train.py
        id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}

        return cls(model=model, tokenizer=tokenizer, id2label=id2label, device=device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> PredictionResult:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text input cannot be empty.")

        inputs = self._tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=self._MAX_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        top_idx = int(probabilities.argmax())
        predicted_condition = self._id2label[top_idx]
        confidence = float(probabilities[top_idx])

        return PredictionResult(predicted_condition=predicted_condition, confidence=confidence)

    def predict_top_k(self, text: str, k: int = 3) -> list[PredictionResult]:
        """Return the top-k predictions sorted by confidence (highest first)."""
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text input cannot be empty.")

        inputs = self._tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=self._MAX_LENGTH,
            padding=True,
        )
        inputs = {k_: v.to(self._device) for k_, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        top_indices = probabilities.topk(min(k, len(self._id2label))).indices.tolist()
        return [
            PredictionResult(
                predicted_condition=self._id2label[idx],
                confidence=float(probabilities[idx]),
            )
            for idx in top_indices
        ]
