"""The wellness narrative prompt: response schema, system instruction,
prompt construction, and the length trim applied to what comes back.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.wellness.responses import (
    WellnessBreakdown,
    WellnessBreakdownItem,
    WellnessReminder,
)
from app.wellness.schemas import (
    WellnessBand,
    WellnessDimensionAvailability,
    WellnessRequest,
    WellnessScoreStatus,
)
from app.wellness.scoring.aggregate import _dimension_items

logger = logging.getLogger(__name__)


class _WellnessNarrative(BaseModel):
    narrative: str = Field(
        ...,
        min_length=1,
        description="A short, mobile-friendly summary — 1-2 sentences, max ~35 words.",
    )
    recommendations: list[str] = Field(default_factory=list)


# Mobile cards have limited room — keep the narrative to ~2 sentences.
_NARRATIVE_MAX_SENTENCES = 2
_NARRATIVE_MAX_CHARS = 240


def _shorten_narrative(text: str) -> str:
    """Hard guard so the narrative stays mobile-friendly regardless of the model.

    Keeps the first couple of sentences and caps total length. Never fails — just
    trims. Recommendations carry the detail, so trimming the narrative is safe.
    """
    text = " ".join(text.split()).strip()
    # Keep the first N sentence-ending segments.
    parts, out = text.replace("! ", ". ").replace("? ", ". ").split(". "), []
    for count, part in enumerate(parts, start=1):
        out.append(part)
        if count >= _NARRATIVE_MAX_SENTENCES:
            break
    short = ". ".join(p.rstrip(".") for p in out).strip()
    if short and not short.endswith((".", "!", "?")):
        short += "."
    if len(short) > _NARRATIVE_MAX_CHARS:
        short = short[:_NARRATIVE_MAX_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return short


_NARRATIVE_SYSTEM = """
You are a veterinary wellness assistant generating a report for a pet owner.
Given the pet details and wellness score breakdown, write:
1. A SHORT narrative for a mobile app card: 1-2 sentences, max ~35 words total.
   Name the weakest SCORED dimension in a few words. Encouraging, non-alarmist. No preamble.
2. Up to 3 specific, actionable recommendations ordered by priority.
   Be concrete — e.g. "Add 10 minutes to morning walks" not just "exercise more".
   Put the detail HERE, not in the narrative.
IMPORTANT: dimensions marked "not tracked (no data)" are MISSING data, NOT low scores.
Never tell the owner to improve a not-tracked dimension. If a not-tracked dimension is
important, you may gently suggest they START TRACKING it — but prioritise dimensions
that were actually scored. Base the narrative's "weakest area" only on scored dimensions.
Do not repeat categories handled by structured reminders.
Do not provide medication, dosing, treatment, parasite-product, or supplement advice.
Never suggest starting, stopping, changing, or administering a medicine.
Return ONLY valid JSON matching the required schema.
"""


def _build_narrative_prompt(
    request: WellnessRequest,
    breakdown: WellnessBreakdown,
    score: int,
    band: WellnessBand,
    score_status: WellnessScoreStatus,
    data_coverage: float,
    condition_cap: int | None,
    detected_condition: str | None,
    reminders: list[WellnessReminder],
    stability: float | None,
) -> str:
    def format_item(item: WellnessBreakdownItem) -> str:
        if not item.included:
            return "not available"
        return f"{item.score}/{item.max_score}"

    lines = [
        f"Pet: {request.pet.species}, {request.pet.breed or 'unknown breed'}, "
        f"age {request.pet.age_months or '?'} months, weight {request.pet.weight_kg or '?'} kg",
        f"Wellness score: {score}/100 ({band.value})",
        f"Assessment status: {score_status.value}",
        f"Weighted data coverage: {data_coverage:.2%}",
        "Breakdown:",
        f"  Activity:      {format_item(breakdown.activity)}",
        f"  Sleep:         {format_item(breakdown.sleep)}",
        f"  Diet:          {format_item(breakdown.diet)}",
        f"  Symptoms:      {format_item(breakdown.symptoms)}",
        f"  Preventive:    {format_item(breakdown.preventive_care)}",
        f"  Baseline:      {format_item(breakdown.baseline)}",
    ]
    missing_dimensions = [
        dimension.value
        for dimension, item in _dimension_items(breakdown)
        if item.availability == WellnessDimensionAvailability.MISSING
    ]
    if missing_dimensions:
        lines.append(f"Missing dimensions: {', '.join(missing_dimensions)}")
        lines.append("This is a partial assessment; do not describe it as complete.")
    if detected_condition:
        lines.append(f"Classifier detected: {detected_condition}")
    if condition_cap is not None:
        lines.append(f"Score capped at {condition_cap} due to active chronic condition.")
    if request.active_conditions:
        names = ", ".join(c.name for c in request.active_conditions)
        lines.append(f"Active conditions: {names}")
    if stability is not None:
        lines.append(f"Weight stability score: {stability:.0%}")
    if reminders:
        handled = ", ".join(reminder.reminder.value for reminder in reminders)
        lines.append(
            f"Handled by structured reminders (do not repeat in recommendations): {handled}"
        )
    return "\n".join(lines)
