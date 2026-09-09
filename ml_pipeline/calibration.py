"""
Confidence calibration and versioned abstention thresholds.

The runtime abstention floor (`app.triage.safety.ABSTAIN_THRESHOLD`)
and the soft low-confidence note threshold (`Settings.low_confidence_threshold`)
are currently fixed numbers with no empirical basis — this module is what
produces that basis: a selective-accuracy curve (at each candidate
confidence threshold, what fraction of predictions would be kept, and how
accurate are the kept ones), and a versioned, evidence-backed threshold
recommendation for a target accuracy.

This does not change the runtime thresholds itself — see design.md open
question "What calibrated safety and quality thresholds define the first
trusted model baseline?" — it produces the evidence a product/safety owner
needs to set them deliberately.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class SelectivePoint:
    threshold: float
    coverage: float  # fraction of predictions with confidence >= threshold
    accuracy: float  # accuracy among the predictions retained at this threshold
    kept: int
    total: int


def compute_selective_accuracy_curve(
    confidences: list[float],
    correct: list[bool],
    thresholds: list[float] | None = None,
) -> list[SelectivePoint]:
    """Sweep candidate thresholds and report coverage/accuracy at each.

    `confidences[i]` is the model's confidence for prediction `i`;
    `correct[i]` is whether that prediction matched the gold label.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if not confidences:
        raise ValueError("cannot compute a selective-accuracy curve with zero predictions")

    thresholds = thresholds or [round(t, 2) for t in _frange(0.0, 1.0, 0.05)]
    total = len(confidences)
    points: list[SelectivePoint] = []
    for threshold in thresholds:
        kept_correct = sum(
            1 for c, ok in zip(confidences, correct, strict=True) if c >= threshold and ok
        )
        kept_total = sum(1 for c in confidences if c >= threshold)
        accuracy = kept_correct / kept_total if kept_total else 0.0
        points.append(
            SelectivePoint(
                threshold=threshold,
                coverage=kept_total / total,
                accuracy=accuracy,
                kept=kept_total,
                total=total,
            )
        )
    return points


def _frange(start: float, stop: float, step: float) -> Iterator[float]:
    n = round((stop - start) / step)
    for i in range(n + 1):
        yield start + i * step


def find_threshold_for_target_accuracy(
    curve: list[SelectivePoint], target_accuracy: float
) -> SelectivePoint | None:
    """Smallest threshold (highest coverage) whose retained-prediction
    accuracy meets `target_accuracy`. None if no threshold achieves it."""
    candidates = [p for p in curve if p.accuracy >= target_accuracy and p.kept > 0]
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.threshold)


@dataclass(frozen=True)
class AbstentionPolicy:
    version: str  # ISO timestamp — each calibration run is a new version
    model_commit: str | None
    target_accuracy: float
    threshold: float
    achieved_accuracy: float
    achieved_coverage: float
    sample_size: int


def build_abstention_policy(
    curve: list[SelectivePoint],
    target_accuracy: float,
    model_commit: str | None,
) -> AbstentionPolicy | None:
    point = find_threshold_for_target_accuracy(curve, target_accuracy)
    if point is None:
        return None
    return AbstentionPolicy(
        version=datetime.now(UTC).isoformat(),
        model_commit=model_commit,
        target_accuracy=target_accuracy,
        threshold=point.threshold,
        achieved_accuracy=point.accuracy,
        achieved_coverage=point.coverage,
        sample_size=point.total,
    )


def write_calibration_report(
    path: str | Path,
    curve: list[SelectivePoint],
    policies: list[AbstentionPolicy],
) -> Path:
    """Append this calibration run to a versioned history file — never
    overwrites prior evidence, so a threshold change can be traced back to
    the run that justified it."""
    path = Path(path)
    history: list[dict] = []
    if path.exists():
        history = json.loads(path.read_text(encoding="utf-8")).get("history", [])

    history.append(
        {
            "curve": [asdict(p) for p in curve],
            "policies": [asdict(p) for p in policies],
        }
    )
    path.write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")
    return path
