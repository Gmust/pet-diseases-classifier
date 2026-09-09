"""Helper shared by every dimension scorer."""

from __future__ import annotations

import logging

from app.wellness.responses import (
    WellnessBreakdownItem,
)
from app.wellness.schemas import (
    WellnessDimensionAvailability,
    WellnessReasonCode,
)

logger = logging.getLogger(__name__)


def _breakdown_item(
    *,
    score: float,
    max_score: float,
    availability: WellnessDimensionAvailability,
    reason_codes: list[WellnessReasonCode],
    evidence: dict[str, bool | int | float | str] | None = None,
) -> WellnessBreakdownItem:
    return WellnessBreakdownItem(
        score=score,
        max_score=max_score,
        availability=availability,
        included=availability == WellnessDimensionAvailability.AVAILABLE,
        reason_codes=reason_codes,
        evidence=evidence or {},
    )
