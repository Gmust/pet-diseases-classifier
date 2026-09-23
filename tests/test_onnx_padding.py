"""The ONNX backend must not inherit the fixed padding baked into tokenizer.json."""

import json
import sys
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
tokenizers = pytest.importorskip("tokenizers")

from app.inference.onnx_predictor import OnnxPredictor  # noqa: E402


def test_from_paths_runs_unpadded_inputs(tmp_path, monkeypatch):
    vocab = {"[PAD]": 0, "[UNK]": 1, "dog": 2}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(json.dumps({"id2label": {"0": "A", "1": "B"}}))
    (tmp_path / "model.onnx").write_bytes(b"")

    shapes = []

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

        def run(self, _outputs, feed):
            shapes.append(feed["input_ids"].shape)
            return [np.zeros((1, 2))]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=FakeSession))

    OnnxPredictor.from_paths(str(tmp_path)).predict("dog dog")

    assert shapes == [(1, 2)]
