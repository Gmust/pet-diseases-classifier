"""Unit tests for the pure helpers (no app, no network)."""
from __future__ import annotations

import pytest

from app.ml.condition_metadata import build_static_explanation, get_condition_metadata
from app.schemas import ChatMessage, ChatRole, UrgencyLevel
from app.services.chat_context import build_classifier_input, latest_user_message
from app.services.triage_safety import (
    ABSTAIN_THRESHOLD,
    EMERGENCY_HOME_ADVICE,
    apply_red_flag_urgency,
    detect_red_flags,
    emergency_explanation,
    should_abstain,
)


# --- triage_safety ----------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "my dog collapsed and is not breathing",
        "she is having a seizure right now",
        "his belly is hard and swollen, looks bloated",
        "he can't pee, keeps straining",
        "my cat ate rat bait",
        "the bleeding won't stop",
        "his gums look blue",
        "he was hit by a car",
    ],
)
def test_red_flags_detected(text):
    assert detect_red_flags(text).triggered is True


@pytest.mark.parametrize("text", ["my dog is a bit itchy", "she has been sneezing a little", ""])
def test_non_emergencies_not_flagged(text):
    assert detect_red_flags(text).triggered is False


def test_red_flag_escalates_urgency():
    urgency, flag = apply_red_flag_urgency("he collapsed", UrgencyLevel.MONITOR)
    assert urgency == UrgencyLevel.EMERGENCY and flag.triggered


def test_no_flag_keeps_base_urgency():
    urgency, flag = apply_red_flag_urgency("mild itching", UrgencyLevel.MONITOR)
    assert urgency == UrgencyLevel.MONITOR and not flag.triggered


def test_should_abstain_threshold():
    assert should_abstain(ABSTAIN_THRESHOLD - 0.01) is True
    assert should_abstain(ABSTAIN_THRESHOLD + 0.01) is False


def test_emergency_explanation_and_advice():
    assert emergency_explanation().startswith("⚠️")
    assert "collapse" in emergency_explanation("collapse / unresponsive")
    # Emergency advice is generic first-aid/transport, not condition-specific.
    joined = " ".join(EMERGENCY_HOME_ADVICE).lower()
    assert "emergency veterinary clinic" in joined
    assert "airway" in joined


# --- chat_context -----------------------------------------------------------

def test_build_input_first_turn_uses_message_only():
    assert build_classifier_input(None, "first message") == "first message"


def test_build_input_merges_summary_and_latest():
    out = build_classifier_input("Vomiting since yesterday.", "Now lethargic")
    assert "Vomiting since yesterday." in out and "Now lethargic" in out


def test_build_input_preserves_newest_under_budget():
    out = build_classifier_input("x" * 5000, "NEWEST MESSAGE")
    assert len(out) <= 1100 and "NEWEST MESSAGE" in out


def test_latest_user_message_skips_trailing_assistant():
    msgs = [
        ChatMessage(role=ChatRole.USER, content="first"),
        ChatMessage(role=ChatRole.ASSISTANT, content="reply"),
        ChatMessage(role=ChatRole.USER, content="second"),
    ]
    assert latest_user_message(msgs) == "second"


def test_latest_user_message_none_when_no_user():
    msgs = [ChatMessage(role=ChatRole.ASSISTANT, content="reply")]
    assert latest_user_message(msgs) is None


# --- static explanation -----------------------------------------------------

def test_static_explanation_is_cautious_and_mentions_condition():
    meta = get_condition_metadata("Digestive Issues")
    text = build_static_explanation("Digestive Issues", meta)
    assert "digestive issues" in text.lower()
    assert "may" in text.lower()           # hedged wording, no assertion of diagnosis


# --- wellness narrative shortening ------------------------------------------

def test_wellness_missing_dimensions_are_excluded_not_zeroed():
    from app.schemas import WellnessPet
    from app.services.wellness_service import (
        _score_baseline,
        _score_diet,
        _score_preventive,
    )

    # Absent dimensions must drop out of the total (max_score == 0), not score 0/MAX.
    assert _score_diet(None, WellnessPet(species="dog")).max_score == 0
    assert _score_preventive(None).max_score == 0
    assert _score_baseline(WellnessPet(species="dog")).max_score == 0  # no age, no weight
    # Baseline with only weight scales its own max (5, not 10).
    item = _score_baseline(WellnessPet(species="dog", weightKg=20.0))
    assert item.max_score == 5 and item.score == 5


def test_narrative_prompt_marks_missing_dimensions_not_tracked():
    from app.schemas import (
        WellnessBand,
        WellnessBreakdown,
        WellnessBreakdownItem,
        WellnessPet,
        WellnessRequest,
    )
    from app.services.wellness_service import _build_narrative_prompt

    def item(score, mx):
        return WellnessBreakdownItem(score=score, max_score=mx)

    breakdown = WellnessBreakdown(
        activity=item(20, 20),
        sleep=item(8, 15),
        diet=item(0, 0),            # not tracked
        symptoms=item(20, 25),
        preventive_care=item(0, 0),  # not tracked
        baseline=item(5, 5),
    )
    req = WellnessRequest(pet=WellnessPet(species="dog"))
    prompt = _build_narrative_prompt(req, breakdown, 81, WellnessBand.GOOD, None, None)
    # Missing dims are labelled, not shown as "0/0" (which reads as a failure).
    assert "Diet:" in prompt and "not tracked" in prompt
    assert "0/0" not in prompt and "0.0/0.0" not in prompt
    # A scored dimension still shows its score.
    assert "20/20" in prompt or "20.0/20.0" in prompt


def test_wellness_narrative_is_trimmed_for_mobile():
    from app.services.wellness_service import (
        _NARRATIVE_MAX_CHARS,
        _shorten_narrative,
    )

    long_text = (
        "Your Labrador is doing wonderfully this week, maintaining a great overall "
        "wellness score! Their activity, diet, and preventive care are all excellent. "
        "The area that could use a little extra attention is sleep, which is important "
        "for muscle repair and overall energy levels. Let's focus on more restful sleep."
    )
    short = _shorten_narrative(long_text)
    assert len(short) <= _NARRATIVE_MAX_CHARS + 1     # +1 for the ellipsis
    assert short.count(".") <= 2
    # A naturally short narrative is left essentially as-is.
    one_liner = "Your cat is in great shape this week."
    assert _shorten_narrative(one_liner) == one_liner
