#!/usr/bin/env python3
"""Fail when a built container image exceeds its tracked compressed-size budget."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def check_size(size_bytes: int, budget_bytes: int, image: str) -> None:
    if size_bytes > budget_bytes:
        raise SystemExit(
            f"Image {image} is {size_bytes / 1_000_000:.1f} MB; "
            f"budget is {budget_bytes / 1_000_000:.1f} MB."
        )


def inspect_image_size(image: str) -> int:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Size}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("profile", choices=("server_torch", "lambda_torch", "lambda_onnx"))
    parser.add_argument("--config", type=Path, default=Path("configs/image-budgets.json"))
    args = parser.parse_args()
    budgets = json.loads(args.config.read_text(encoding="utf-8"))
    check_size(inspect_image_size(args.image), int(budgets[args.profile]["max_bytes"]), args.image)


if __name__ == "__main__":
    main()
