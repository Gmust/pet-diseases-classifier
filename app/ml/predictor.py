from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# torch / transformers are imported lazily inside the methods that use them.
# This keeps `import app.ml.predictor` cheap: the test suite and lightweight
# tooling can import the module (and PredictionResult) without pulling in the
# heavy ML stack, and it shaves a little off the Lambda cold start before the
# model is actually loaded.
if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from app.ml.protocols import ClassifierMetadata

from app.ml.model_registry import release_version
from app.ml.model_validation import (
    validate_id2label,
    validate_required_files,
    validate_top_k,
    verify_checksums,
)


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
        model_path: str = "",
        model_version: str = "unversioned-local",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._id2label = id2label
        self._device = device
        self._model_path = model_path
        self._model_version = model_version

    @property
    def metadata(self) -> ClassifierMetadata:
        from app.ml.protocols import ClassifierMetadata

        labels = tuple(self._id2label[i] for i in sorted(self._id2label))
        return ClassifierMetadata(
            backend="torch",
            model_path=self._model_path,
            labels=labels,
            model_version=self._model_version,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_paths(cls, model_path: str) -> Predictor:
        """Load a validated Hugging Face directory produced by train.py."""
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
        validate_required_files(model_dir, ["config.json"])
        verify_checksums(model_dir)

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        validate_id2label(id2label)

        return cls(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            device=device,
            model_path=str(model_dir),
            model_version=release_version(model_dir),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> PredictionResult:
        import torch

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
        import torch

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text input cannot be empty.")
        k = validate_top_k(k, len(self._id2label))

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

        top_indices = probabilities.topk(k).indices.tolist()
        return [
            PredictionResult(
                predicted_condition=self._id2label[idx],
                confidence=float(probabilities[idx]),
            )
            for idx in top_indices
        ]

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[PredictionResult]:
        """Return batched top-1 predictions."""
        return [row[0] for row in self.predict_top_k_batch(texts, k=1, batch_size=batch_size)]

    def predict_top_k_batch(
        self,
        texts: list[str],
        k: int = 3,
        batch_size: int = 32,
    ) -> list[list[PredictionResult]]:
        """Return top-k predictions using one model forward pass per batch."""
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        k = validate_top_k(k, len(self._id2label))
        results: list[list[PredictionResult]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            cleaned = [t.strip() for t in chunk]
            for offset, c in enumerate(cleaned):
                if not c:
                    raise ValueError(f"Text input cannot be empty (index {start + offset}).")

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
                probabilities = torch.softmax(logits, dim=-1)

            top_probabilities, top_indices = probabilities.topk(k, dim=-1)
            for row_indices, row_probabilities in zip(
                top_indices.tolist(), top_probabilities.tolist(), strict=True
            ):
                results.append(
                    [
                        PredictionResult(
                            predicted_condition=self._id2label[index],
                            confidence=float(confidence),
                        )
                        for index, confidence in zip(
                            row_indices,
                            row_probabilities,
                            strict=True,
                        )
                    ]
                )
        return results
