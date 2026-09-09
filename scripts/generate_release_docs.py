#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_pipeline.release_docs import render_data_card, render_release_model_card


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/release-documentation.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    metadata = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data-card.md").write_text(render_data_card(metadata), encoding="utf-8")
    (args.output_dir / "model-card.md").write_text(
        render_release_model_card(metadata), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
