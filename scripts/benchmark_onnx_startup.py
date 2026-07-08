#!/usr/bin/env python3
"""Measure ONNX model construction in fresh Python processes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHILD_CODE = """
import json
import resource
import platform
import sys
import time
from app.ml.onnx_predictor import OnnxPredictor
start = time.perf_counter()
predictor = OnnxPredictor.from_paths(sys.argv[1])
elapsed = (time.perf_counter() - start) * 1000
print(json.dumps({
    "startup_ms": round(elapsed, 2),
    "max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "platform": platform.system(),
    "model_version": predictor.metadata.model_version,
}))
"""


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("At least one startup sample is required.")
    durations = [float(sample["startup_ms"]) for sample in samples]
    return {
        "sample_count": len(samples),
        "median_startup_ms": round(statistics.median(durations), 2),
        "max_startup_ms": round(max(durations), 2),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/transformer_model_onnx")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be at least 1")

    samples: list[dict[str, Any]] = []
    for _ in range(args.iterations):
        result = subprocess.run(
            [sys.executable, "-c", CHILD_CODE, args.model_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        samples.append(json.loads(result.stdout.strip()))
    report = {
        "schema_version": 1,
        "measured_at": datetime.now(UTC).isoformat(),
        "method": "fresh local Python process; excludes Lambda image-pull/platform initialization",
        "model_dir": args.model_dir,
        **summarize(samples),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
