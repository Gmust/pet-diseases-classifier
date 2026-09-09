import json
from pathlib import Path

from ml_pipeline.release_docs import render_data_card, render_release_model_card


def test_tracked_release_cards_match_generator() -> None:
    metadata = json.loads(Path("configs/release-documentation.json").read_text(encoding="utf-8"))
    assert Path("docs/data-card.md").read_text(encoding="utf-8") == render_data_card(metadata)
    assert Path("docs/model-card.md").read_text(encoding="utf-8") == render_release_model_card(
        metadata
    )
