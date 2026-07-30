"""
Safety overrides that sit in front of the classifier.

Why this exists
---------------
`urgency` normally comes from the *predicted condition class* (see
condition_metadata.py). That means a genuinely life-threatening description can
be returned as MONITOR / CONSULT_SOON if it happens to classify into a
low-urgency class. For a triage product that is the most dangerous failure mode.

This module adds two deterministic, local (no-API) guards:

1. Red-flag escalation — if the owner's text contains emergency phrases
   (collapse, not breathing, seizure, bloat, unable to urinate, toxin
   ingestion, severe bleeding...), force urgency to EMERGENCY regardless of the
   classifier output.

2. Low-confidence abstention — when the classifier is not confident enough, the
   caller should hedge rather than present a confident-looking label.

Both are intentionally simple and conservative: false positives (over-warning)
are acceptable here; false negatives are not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import UrgencyLevel

# Each tuple: (human-readable reason, regex of trigger phrases).
# Phrases are matched case-insensitively on word boundaries where it matters.
_RED_FLAG_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "not breathing / difficulty breathing",
        re.compile(
            r"\b(not|n't|stopped|can'?t|cannot|trouble|difficulty|struggling to|labou?red)\s+breath",
            re.I,
        ),
    ),
    (
        "collapse / unresponsive",
        re.compile(
            r"\b(collaps\w*|unconscious|unresponsive|won'?t wake|passed out|faint\w*)\b", re.I
        ),
    ),
    (
        "active seizure / convulsions",
        re.compile(r"\b(seizur\w*|convuls\w*|fitting|having a fit)\b", re.I),
    ),
    (
        "suspected bloat / GDV",
        re.compile(
            r"\b(bloat\w*|gdv|distended (abdomen|belly|stomach)|"
            r"swollen (hard )?(abdomen|belly|stomach)|hard,? swollen|"
            r"(abdomen|belly|stomach) is hard and swollen)\b",
            re.I,
        ),
    ),
    (
        "unable to urinate (blocked)",
        re.compile(
            r"\b(can'?t|cannot|unable to|straining to|trying but no|not able to)\s+(pee|urinat\w*|wee)\b",
            re.I,
        ),
    ),
    (
        "toxin / poison ingestion",
        re.compile(
            r"\b(ate|ingest\w*|swallow\w*|got into|chewed)\b.{0,30}\b(poison|rat ?bait|antifreeze|chocolate|xylitol|grapes?|raisins?|lily|lilies|rodenticide|medication|pills?|toxic)\b",
            re.I,
        ),
    ),
    (
        "severe / uncontrolled bleeding",
        re.compile(
            r"\b(heav\w*|severe|profuse|won'?t stop|uncontroll\w*|gushing|spurting)\b.{0,25}\bbleed"
            r"|\bbleed\w*\b.{0,25}\b(won'?t stop|uncontroll\w*|profuse|severe|heav\w*|gushing|spurting)\b",
            re.I,
        ),
    ),
    (
        "pale / blue gums",
        re.compile(
            r"\b(blue|grey|gray|white|pale)\b.{0,15}\bgums?\b"
            r"|\bgums?\b.{0,15}\b(blue|grey|gray|white|pale)\b",
            re.I,
        ),
    ),
    ("heatstroke", re.compile(r"\b(heat ?stroke|overheat\w*)\b", re.I)),
    (
        "trauma / hit by car",
        re.compile(r"\b(hit by (a )?car|hbc|fell from|major trauma|run over)\b", re.I),
    ),
    (
        "pregnancy emergency (dystocia)",
        re.compile(
            r"\b(in labou?r|giving birth|whelp\w*|dystocia)\b.{0,30}\b(hours?|straining|stuck|can'?t)\b",
            re.I,
        ),
    ),
]

# Narrowly remove common statements that explicitly say a red-flag finding is
# absent. This is intentionally not a generic negation engine: phrases such as
# "not breathing" are themselves emergencies and must remain available to the
# positive rules above.
_NEGATED_FINDINGS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:no|without)\s+(?:active\s+)?(?:seizures?|convulsions?|collapse)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:did not|didn't|has not|hasn't|had not|hadn't|is not|isn't|was not|wasn't)\s+"
        r"(?:have|had|having|experience|experienced)?\s*(?:a\s+)?"
        r"(?:seizure|convulsion|collapse)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:gums?\s+(?:are|look|seem)\s+not|gums?\s+(?:aren't|weren't))\s+"
        r"(?:pale|blue|grey|gray|white)\b",
        re.I,
    ),
    re.compile(r"\b(?:is not|isn't|not)\s+bloat(?:ed|ing)?\b", re.I),
)


@dataclass(frozen=True)
class RedFlag:
    triggered: bool
    reason: str | None = None


def detect_red_flags(text: str) -> RedFlag:
    """Return a RedFlag if the text contains an emergency phrase."""
    if not text:
        return RedFlag(False)
    searchable = text
    for pattern in _NEGATED_FINDINGS:
        searchable = pattern.sub(" ", searchable)
    for reason, pattern in _RED_FLAG_RULES:
        if pattern.search(searchable):
            return RedFlag(True, reason)
    return RedFlag(False)


def apply_red_flag_urgency(text: str, base_urgency: UrgencyLevel) -> tuple[UrgencyLevel, RedFlag]:
    """Escalate to EMERGENCY when a red flag fires; otherwise keep base_urgency."""
    flag = detect_red_flags(text)
    if flag.triggered:
        return UrgencyLevel.EMERGENCY, flag
    return base_urgency, flag


# Below this confidence the prediction is too weak to present as a label.
# Distinct from LOW_CONFIDENCE_THRESHOLD (which only appends a soft note).
ABSTAIN_THRESHOLD = 0.40

ABSTAIN_NOTE = (
    "I couldn't confidently match these symptoms to a single condition. "
    "Please add more detail (when it started, appetite, energy, other changes), "
    "or consult a veterinarian if you're concerned."
)

RED_FLAG_NOTE = (
    "⚠️ Some of what you described can indicate a medical emergency. "
    "Please contact an emergency veterinary clinic right now."
)

# Generic, condition-agnostic first-aid/transport guidance shown when a red flag
# fires. Deliberately NOT condition-specific: on an emergency we must not present
# the (possibly wrong) predicted class's home advice.
EMERGENCY_HOME_ADVICE: list[str] = [
    "Stay calm and keep your pet as still and quiet as possible.",
    "Check that the airway is clear and watch the chest for breathing.",
    "Do not give food, water, or any medication.",
    "Keep your pet warm and minimise handling or movement.",
    "Transport to the nearest emergency veterinary clinic right now — call ahead so they can prepare.",
]


def emergency_explanation(reason: str | None = None) -> str:
    """User-facing text for a red-flag response — leads with the emergency message
    and never rationalises the predicted condition."""
    if reason:
        return f"{RED_FLAG_NOTE} (Possible sign detected: {reason}.)"
    return RED_FLAG_NOTE


def should_abstain(confidence: float, threshold: float = ABSTAIN_THRESHOLD) -> bool:
    """True when confidence is below the abstention floor."""
    return confidence < threshold
