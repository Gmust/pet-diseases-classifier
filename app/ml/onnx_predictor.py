"""
ONNX Runtime predictor — drop-in replacement for the torch `Predictor`.

Same public interface (`from_paths`, `predict`, `predict_top_k`, `PredictionResult`)
so the request handlers don't change. Selected at runtime via MODEL_BACKEND=onnx.

Deliberately uses ONLY onnxruntime + the `tokenizers` library + numpy — NO torch,
NO optimum, NO transformers. That's what keeps the Lambda image small and the cold
start fast. Produce the model dir with `python -m app.ml.export_onnx` first.

All heavy imports are lazy so importing this module stays cheap.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.ml.predictor import PredictionResult


def _softmax(logits):
    import numpy as np

    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


class OnnxPredictor:
    _MAX_LENGTH = 256

    def __init__(self, session, tokenizer, id2label: dict[int, str], input_names: set[str]) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._id2label = id2label
        self._input_names = input_names

    @classmethod
    def from_paths(cls, model_path: str, **_ignored) -> "OnnxPredictor":
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"ONNX model directory not found: {model_dir}\n"
                "Export it first: python -m app.ml.export_onnx "
                f"--model-dir models/transformer_model --output-dir {model_dir}"
            )

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

        return cls(session=session, tokenizer=tokenizer, id2label=id2label, input_names=input_names)

    def _logits(self, text: str):
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
        k = min(k, len(self._id2label))
        top_indices = np.argsort(probs)[::-1][:k]
        return [
            PredictionResult(
                predicted_condition=self._id2label[int(idx)],
                confidence=float(probs[int(idx)]),
            )
            for idx in top_indices
        ]
