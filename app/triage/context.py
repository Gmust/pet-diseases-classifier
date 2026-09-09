"""
Builds the single text string fed to the classifier on a chat turn.

The classifier is a single-text → single-label model with a 256-token cap, so a
multi-turn conversation must be condensed into one compact, symptom-only input.

Strategy (see README / design notes):
- Combine the rolling `symptom_summary` (carried by the backend across turns)
  with the newest user message.
- The newest message is the highest-signal content, so it is placed first and is
  never trimmed. If the combined string would blow the budget, the older summary
  is truncated from its tail instead.
- The classifier's tokenizer applies the final 256-token truncation; the char
  budget here is a coarse guard so the newest message always survives it.
"""

from __future__ import annotations

from app.triage.schemas import ChatMessage

# ~256 tokens ≈ 1000-1200 chars for English text. Stay under it with headroom.
_MAX_INPUT_CHARS = 1100
_MAX_SAFETY_SUMMARY_CHARS = 1000
_MAX_SAFETY_MESSAGE_CHARS = 4000
_SUMMARY_PREFIX = "Prior symptoms: "
_LATEST_PREFIX = "Latest update: "


def build_classifier_input(symptom_summary: str | None, latest_message: str) -> str:
    """Merge the rolling summary with the newest user message into one classifier input.

    Args:
        symptom_summary: Distilled symptoms from previous turns (may be None/empty).
        latest_message: The newest user message on this turn.

    Returns:
        A single string, newest-content-first, bounded to the classifier budget.
    """
    latest = (latest_message or "").strip()
    summary = (symptom_summary or "").strip()

    if not summary:
        return latest[:_MAX_INPUT_CHARS]
    if not latest:
        return f"{_SUMMARY_PREFIX}{summary}"[:_MAX_INPUT_CHARS]

    latest_part = f"{_LATEST_PREFIX}{latest}"
    # Reserve room for the newest message; trim the older summary if needed.
    remaining = _MAX_INPUT_CHARS - len(latest_part) - len(_SUMMARY_PREFIX) - 1
    if remaining <= 0:
        # Newest message alone already fills the budget — keep only that.
        return latest_part[:_MAX_INPUT_CHARS]

    summary_part = f"{_SUMMARY_PREFIX}{summary[:remaining]}"
    return f"{summary_part}\n{latest_part}"


def latest_user_message(messages: list[ChatMessage]) -> str | None:
    """Return the content of the most recent user-role message, or None."""
    for msg in reversed(messages):
        role = getattr(msg, "role", None)
        role_value = getattr(role, "value", role)
        if role_value == "user":
            return msg.content
    return None


def build_safety_context(symptom_summary: str | None, latest_message: str) -> str:
    """Build bounded context for deterministic emergency-phrase evaluation.

    The API already caps individual chat messages at 4,000 characters. The rolling
    summary is caller-provided and currently unbounded, so only its most recent
    1,000 characters are retained. The latest message is always included in full.
    """
    summary = (symptom_summary or "").strip()
    latest = (latest_message or "").strip()[:_MAX_SAFETY_MESSAGE_CHARS]
    recent_summary = summary[-_MAX_SAFETY_SUMMARY_CHARS:]
    return "\n".join(part for part in (recent_summary, latest) if part)
