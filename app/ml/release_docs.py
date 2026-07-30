"""Generate transparent model/data release cards from reviewed JSON metadata."""

from __future__ import annotations

from typing import Any


def render_data_card(metadata: dict[str, Any]) -> str:
    sources = "\n".join(
        f"- {source['name']}: {source['license']} ({source['revision']})"
        for source in metadata["sources"]
    )
    return f"""# Data Card

Generated from `configs/release-documentation.json`; do not edit by hand.

## Intended use

{metadata['data_intended_use']}

## Sources and licenses

{sources}

## Coverage

- Classes: {metadata['class_coverage']}
- Species/register: {metadata['species_register_coverage']}
- Synthetic contribution: {metadata['synthetic_proportion']}

## Transformations and exclusions

{metadata['transformations_exclusions']}

## Known gaps

{metadata['known_data_gaps']}
"""


def render_release_model_card(metadata: dict[str, Any]) -> str:
    return f"""# Release Model Card

Generated from `configs/release-documentation.json`; do not edit by hand.

## Intended and prohibited use

{metadata['model_intended_use']}

Prohibited: {metadata['prohibited_use']}

## Training and provenance

- Active backend: {metadata['active_backend']}
- Training configuration: `{metadata['training_config']}`
- Dataset/split evidence: `{metadata['dataset_evidence']}`
- Rollback version: {metadata['rollback_version']}

## Evaluation and safety evidence

{metadata['evaluation_evidence']}

Calibration/abstention: {metadata['calibration_evidence']}

Safety: {metadata['safety_evidence']}

## Limitations and governance

{metadata['model_limitations']}

Clinical status: {metadata['clinical_status']}
"""
