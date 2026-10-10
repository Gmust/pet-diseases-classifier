import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from ml_pipeline.export_onnx import quantize_int8


def test_quantize_int8_uses_per_channel_reduced_range(monkeypatch) -> None:
    # Per-channel int8 without reduce_range collapsed the classifier on x86 AVX2
    # (ADR 0004 addendum); dropping either flag must fail here.
    calls = []
    fake = ModuleType("onnxruntime.quantization")
    fake.QuantType = SimpleNamespace(QInt8="QInt8")
    fake.quantize_dynamic = lambda src, dst, **kw: calls.append((src, dst, kw))
    monkeypatch.setitem(sys.modules, "onnxruntime", ModuleType("onnxruntime"))
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", fake)

    quantize_int8(Path("model.onnx"), Path("model_quantized.onnx"))

    assert calls == [
        (
            "model.onnx",
            "model_quantized.onnx",
            {"weight_type": "QInt8", "per_channel": True, "reduce_range": True},
        )
    ]
