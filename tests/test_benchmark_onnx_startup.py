import pytest

from scripts.benchmark_onnx_startup import summarize


def test_summarize_startup_samples() -> None:
    report = summarize([{"startup_ms": 100}, {"startup_ms": 300}, {"startup_ms": 200}])
    assert report["median_startup_ms"] == 200
    assert report["max_startup_ms"] == 300


def test_summarize_requires_samples() -> None:
    with pytest.raises(ValueError, match="At least one"):
        summarize([])
