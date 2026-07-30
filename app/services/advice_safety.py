"""
Safety checks on home-advice text before it reaches an owner.

Generated text can slip into territory a static template never would: a
specific drug dose, a "prescribe" directive, or a diagnostic-sounding claim
the classifier never actually made. This is a conservative denylist, not a
substitute for clinical review — see `app.ml.condition_metadata.AdviceReview`
for the human-approval record on the static advice each condition falls back to.
"""

from __future__ import annotations

import re

from app.observability import log_event

MAX_ADVICE_ITEMS = 6
MAX_ADVICE_CHARS = 240

# False positives (dropping a fine tip) are acceptable here; false negatives
# (an owner acting on a specific unapproved dose) are not — see triage_safety.py
# module docstring for the same asymmetry applied to red-flag detection.
_UNSAFE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d+(\.\d+)?\s*(mg|ml|mcg|milligrams?|milliliters?)\b", re.I),
    # Directive framing where the advice itself does the prescribing — distinct
    # from the common, desirable caution "...unless prescribed by a vet".
    re.compile(r"\b(you should|we) prescri\w*\b", re.I),
    re.compile(r"\b(diagnos\w*|confirm\w*)\s+(that|this|it)\b", re.I),
    re.compile(r"\bdefinitely (has|is|are)\b", re.I),
)


def is_advice_item_safe(item: str) -> bool:
    if not item or not item.strip():
        return False
    if len(item) > MAX_ADVICE_CHARS:
        return False
    return not any(pattern.search(item) for pattern in _UNSAFE_PATTERNS)


def sanitize_advice(items: list[str], *, fallback: list[str], source: str) -> list[str]:
    """Drop unsafe/oversized items and cap the list length.

    Falls back to `fallback` (reviewed static advice) if nothing survives, so
    the response never ships an empty home-advice list.
    """
    safe = [item.strip() for item in items if is_advice_item_safe(item)]
    dropped = len(items) - len(safe)
    if dropped:
        log_event("advice_filtered", source=source, dropped=dropped, kept=len(safe))
    safe = safe[:MAX_ADVICE_ITEMS]
    return safe or list(fallback)
