"""
ONNX Runtime predictor — drop-in replacement for the torch `Predictor`.

Same public interface (`from_paths`, `predict`, `predict_top_k`, `PredictionResult`)
so the request handlers don't change. Selected at runtime via MODEL_BACKEND=onnx.

Deliberately uses ONLY onnxruntime + the `tokenizers` library + numpy — NO torch,
NO optimum, NO transformers. That's what keeps the Lambda image small and the cold
start fast. Produce the model dir with `python -m ml_pipeline.export_onnx` first.

All heavy imports are lazy so importing this module stays cheap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.inference.model_registry import release_version
from app.inference.model_validation import (
    validate_id2label,
    validate_required_files,
    validate_top_k,
    verify_checksums,
)
from app.inference.predictor import PredictionResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.inference.protocols import ClassifierMetadata


def _softmax(logits):
    import numpy as np

    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


class OnnxPredictor:
    _MAX_LENGTH = 256

    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        id2label: dict[int, str],
        input_names: set[str],
        model_path: str = "",
        model_version: str = "unversioned-local",
    ) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._id2label = id2label
        self._input_names = input_names
        self._model_path = model_path
        self._model_version = model_version

    @property
    def metadata(self) -> ClassifierMetadata:
        from app.inference.protocols import ClassifierMetadata

        labels = tuple(self._id2label[i] for i in sorted(self._id2label))
        return ClassifierMetadata(
            backend="onnx",
            model_path=self._model_path,
            labels=labels,
            model_version=self._model_version,
        )

    @classmethod
    def from_paths(cls, model_path: str) -> OnnxPredictor:
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"ONNX model directory not found: {model_dir}\n"
                "Export it first: python -m ml_pipeline.export_onnx "
                f"--model-dir models/transformer_model --output-dir {model_dir}"
            )
        validate_required_files(model_dir, ["config.json", "tokenizer.json"])
        verify_checksums(model_dir)

        import onnxruntime as ort
        from tokenizers import Tokenizer

        # Prefer the quantized model; fall back to fp32, then any .onnx.
        onnx_path = model_dir / "model_quantized.onnx"
        if not onnx_path.exists():
            onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            candidates = list(model_dir.glob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(f"No .onnx file in {model_dir}")
            onnx_path = candidates[0]

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_names = {i.name for i in session.get_inputs()}

        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=cls._MAX_LENGTH)

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in config["id2label"].items()}
        validate_id2label(id2label)

        return cls(
            session=session,
            tokenizer=tokenizer,
            id2label=id2label,
            input_names=input_names,
            model_path=str(model_dir),
            model_version=release_version(model_dir),
        )

    def _logits(self, text: str) -> Any:
        import numpy as np

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text input cannot be empty.")

        encoding = self._tokenizer.encode(cleaned)
        feed = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }
        # Only pass inputs the graph actually declares (DistilBERT has no token_type_ids).
        feed = {k: v for k, v in feed.items() if k in self._input_names}
        outputs = self._session.run(None, feed)
        return np.asarray(outputs[0])[0]

    def predict(self, text: str) -> PredictionResult:
        probs = _softmax(self._logits(text))
        top_idx = int(probs.argmax())
        return PredictionResult(
            predicted_condition=self._id2label[top_idx],
            confidence=float(probs[top_idx]),
        )

    def predict_top_k(self, text: str, k: int = 3) -> list[PredictionResult]:
        import numpy as np

        probs = _softmax(self._logits(text))
        k = validate_top_k(k, len(self._id2label))
        top_indices = np.argsort(probs)[::-1][:k]
        return [
            PredictionResult(
                predicted_condition=self._id2label[int(idx)],
                confidence=float(probs[int(idx)]),
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
        """Return top-k predictions using one ONNX Runtime call per batch."""
        import numpy as np

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        k = validate_top_k(k, len(self._id2label))
        results: list[list[PredictionResult]] = []
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        try:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                cleaned = [t.strip() for t in chunk]
                for offset, c in enumerate(cleaned):
                    if not c:
                        raise ValueError(f"Text input cannot be empty (index {start + offset}).")

                encodings = self._tokenizer.encode_batch(cleaned)
                feed = {
                    "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                    "attention_mask": np.array(
                        [e.attention_mask for e in encodings], dtype=np.int64
                    ),
                }
                feed = {k: v for k, v in feed.items() if k in self._input_names}
                outputs = self._session.run(None, feed)
                logits = np.asarray(outputs[0])

                for row in logits:
                    probs = _softmax(row)
                    top_indices = np.argsort(probs)[::-1][:k]
                    results.append(
                        [
                            PredictionResult(
                                predicted_condition=self._id2label[int(index)],
                                confidence=float(probs[int(index)]),
                            )
                            for index in top_indices
                        ]
                    )
        finally:
            self._tokenizer.no_padding()
        return results
