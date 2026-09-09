"""Unit tests for the pure helpers (no app, no network)."""

from __future__ import annotations

import hashlib
import json

import pytest

from app.domain.conditions import (
    CONDITION_METADATA,
    build_static_explanation,
    get_condition_metadata,
)
from app.domain.enums import UrgencyLevel
from app.inference.model_validation import (
    ModelValidationError,
    validate_id2label,
    validate_required_files,
    validate_top_k,
    verify_checksums,
)
from app.triage.context import (
    build_classifier_input,
    build_safety_context,
    latest_user_message,
)
from app.triage.safety import (
    ABSTAIN_THRESHOLD,
    EMERGENCY_HOME_ADVICE,
    apply_red_flag_urgency,
    detect_red_flags,
    emergency_explanation,
    should_abstain,
)
from app.triage.schemas import ChatMessage, ChatRole

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


@pytest.mark.parametrize(
    "text",
    [
        "she has not had a seizure",
        "he did not collapse",
        "his gums are not pale",
        "her stomach is not bloated",
    ],
)
def test_negated_red_flags_are_not_detected(text):
    assert detect_red_flags(text).triggered is False


def test_not_breathing_remains_an_emergency_despite_negation_handling():
    assert detect_red_flags("my dog is not breathing").triggered is True


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


def test_safety_context_bounds_summary_and_preserves_latest_message():
    out = build_safety_context("OLD" + "x" * 5000, "LATEST seizure")
    assert len(out) <= 1000 + 1 + len("LATEST seizure")
    assert "OLD" not in out
    assert out.endswith("LATEST seizure")


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
    assert "may" in text.lower()  # hedged wording, no assertion of diagnosis


# --- wellness narrative shortening ------------------------------------------


def test_wellness_missing_dimensions_are_explicit_and_excluded():
    from app.wellness.schemas import WellnessDimensionAvailability, WellnessPet
    from app.wellness.scoring.baseline import _score_baseline
    from app.wellness.scoring.diet import _score_diet
    from app.wellness.scoring.preventive import _score_preventive

    items = (
        _score_diet(None, WellnessPet(species="dog")),
        _score_preventive(None, None),
        _score_baseline(WellnessPet(species="dog"), [], None),
    )
    for item in items:
        assert item.availability == WellnessDimensionAvailability.MISSING
        assert item.included is False
        assert item.score == 0


def test_narrative_prompt_marks_missing_dimensions_not_tracked():
    from app.wellness.prompt import _build_narrative_prompt
    from app.wellness.responses import WellnessBreakdown, WellnessBreakdownItem
    from app.wellness.schemas import (
        WellnessBand,
        WellnessDimensionAvailability,
        WellnessPet,
        WellnessReasonCode,
        WellnessRequest,
        WellnessScoreStatus,
    )

    def item(score, mx, availability, reason):
        return WellnessBreakdownItem(
            score=score,
            max_score=mx,
            availability=availability,
            included=availability == WellnessDimensionAvailability.AVAILABLE,
            reason_codes=[reason],
        )

    breakdown = WellnessBreakdown(
        activity=item(
            20, 20, WellnessDimensionAvailability.AVAILABLE, WellnessReasonCode.ACTIVITY_TARGET_MET
        ),
        sleep=item(
            8, 15, WellnessDimensionAvailability.AVAILABLE, WellnessReasonCode.SLEEP_WITHIN_RANGE
        ),
        diet=item(
            0, 20, WellnessDimensionAvailability.MISSING, WellnessReasonCode.DIET_DATA_MISSING
        ),
        symptoms=item(
            20,
            25,
            WellnessDimensionAvailability.AVAILABLE,
            WellnessReasonCode.SYMPTOM_RESULT_AVAILABLE,
        ),
        preventive_care=item(
            0,
            10,
            WellnessDimensionAvailability.MISSING,
            WellnessReasonCode.PREVENTIVE_CARE_DATA_MISSING,
        ),
        baseline=item(
            10, 10, WellnessDimensionAvailability.AVAILABLE, WellnessReasonCode.BASELINE_STABLE
        ),
    )
    req = WellnessRequest(pet=WellnessPet(species="dog", weightKg=20))
    prompt = _build_narrative_prompt(
        req,
        breakdown,
        81,
        WellnessBand.GOOD,
        WellnessScoreStatus.PARTIAL,
        0.75,
        None,
        None,
        [],
        None,
    )
    # Missing dimensions are labelled as unavailable, not shown as a failed score.
    assert "Diet:" in prompt and "not available" in prompt
    assert "Missing dimensions: Diet, PreventiveCare" in prompt
    diet_line = next(line for line in prompt.splitlines() if "Diet:" in line)
    assert "0/20" not in diet_line and "0.0/20.0" not in diet_line
    # A scored dimension still shows its score.
    assert "20/20" in prompt or "20.0/20.0" in prompt


def test_wellness_narrative_is_trimmed_for_mobile():
    from app.wellness.prompt import _NARRATIVE_MAX_CHARS, _shorten_narrative

    long_text = (
        "Your Labrador is doing wonderfully this week, maintaining a great overall "
        "wellness score! Their activity, diet, and preventive care are all excellent. "
        "The area that could use a little extra attention is sleep, which is important "
        "for muscle repair and overall energy levels. Let's focus on more restful sleep."
    )
    short = _shorten_narrative(long_text)
    assert len(short) <= _NARRATIVE_MAX_CHARS + 1  # +1 for the ellipsis
    assert short.count(".") <= 2
    # A naturally short narrative is left essentially as-is.
    one_liner = "Your cat is in great shape this week."
    assert _shorten_narrative(one_liner) == one_liner


# --- model_validation ---------------------------------------------------------


def test_validate_required_files_passes_when_all_present(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    validate_required_files(tmp_path, ["config.json"])  # no raise


def test_validate_required_files_raises_when_missing(tmp_path):
    with pytest.raises(ModelValidationError, match="missing"):
        validate_required_files(tmp_path, ["config.json"])


def test_validate_id2label_accepts_contiguous_unique_labels():
    validate_id2label({0: "A", 1: "B", 2: "C"})  # no raise


@pytest.mark.parametrize(
    "id2label",
    [
        {},
        {0: "A", 2: "C"},  # non-contiguous
        {0: "A", 1: "A"},  # duplicate
        {0: "A", 1: "  "},  # blank label
    ],
)
def test_validate_id2label_rejects_malformed_maps(id2label):
    with pytest.raises(ModelValidationError):
        validate_id2label(id2label)


def test_validate_top_k_rejects_non_positive():
    with pytest.raises(ValueError):
        validate_top_k(0, num_labels=5)


def test_validate_top_k_clamps_to_label_count():
    assert validate_top_k(10, num_labels=3) == 3
    assert validate_top_k(2, num_labels=3) == 2


def test_verify_checksums_noop_without_manifest(tmp_path):
    verify_checksums(tmp_path)  # no manifest present -> no raise


def test_verify_checksums_passes_when_matching(tmp_path):
    target = tmp_path / "model.bin"
    target.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": {"model.bin": digest}}))
    verify_checksums(tmp_path)  # no raise


def test_weights_present_detects_pytorch_bin_only(tmp_path):
    # Regression guard for the `glob(...) or glob(...)` bug: a generator is
    # always truthy, so the pytorch_model.bin pattern was never actually
    # checked when the .safetensors pattern matched nothing.
    from tests.test_classifier_regression import _weights_present

    (tmp_path / "pytorch_model.bin").write_bytes(b"fake weights")
    assert _weights_present(str(tmp_path)) is True


def test_weights_present_detects_safetensors(tmp_path):
    from tests.test_classifier_regression import _weights_present

    (tmp_path / "model.safetensors").write_bytes(b"fake weights")
    assert _weights_present(str(tmp_path)) is True


def test_weights_present_false_when_no_weights(tmp_path):
    from tests.test_classifier_regression import _weights_present

    assert _weights_present(str(tmp_path)) is False


def test_verify_checksums_fails_on_mismatch(tmp_path):
    target = tmp_path / "model.bin"
    target.write_bytes(b"weights")
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": {"model.bin": "deadbeef"}}))
    with pytest.raises(ModelValidationError, match="Checksum verification failed"):
        verify_checksums(tmp_path)


# --- generation_policy ---------------------------------------------------------


def test_call_with_policy_returns_result_on_success():
    from app.llm.generation_policy import call_with_policy

    assert call_with_policy(lambda: 42) == 42


def test_call_with_policy_raises_timeout_error_when_slow():
    import time

    from app.llm.generation_policy import GenerationTimeoutError, call_with_policy

    started = time.perf_counter()
    with pytest.raises(GenerationTimeoutError):
        call_with_policy(lambda: time.sleep(0.5), timeout=0.02, max_attempts=1)
    assert time.perf_counter() - started < 0.2


def test_call_with_policy_retries_transient_errors_then_succeeds():
    from app.llm.generation_policy import call_with_policy

    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("transient")
        return "ok"

    assert call_with_policy(flaky, max_attempts=2) == "ok"
    assert calls["count"] == 2


def test_call_with_policy_raises_after_exhausting_retries():
    from app.llm.generation_policy import call_with_policy

    def always_fails():
        raise RuntimeError("still broken")

    with pytest.raises(RuntimeError, match="still broken"):
        call_with_policy(always_fails, max_attempts=2)


def test_bound_prompt_truncates_long_text():
    from app.llm.generation_policy import bound_prompt

    assert bound_prompt("x" * 100, max_chars=10) == "x" * 10
    assert bound_prompt("short", max_chars=10) == "short"


def test_bounded_chat_transcript_drops_oldest_and_keeps_latest():
    from app.llm.gemini_service import _bounded_chat_transcript

    conversation = [
        {"role": "user", "content": "old" * 100},
        {"role": "assistant", "content": "middle" * 100},
        {"role": "user", "content": "LATEST CURRENT SYMPTOM"},
    ]
    transcript = _bounded_chat_transcript(conversation, max_chars=80)

    assert "LATEST CURRENT SYMPTOM" in transcript
    assert "oldold" not in transcript


def test_chat_generation_prompt_retains_latest_message():
    from types import SimpleNamespace

    from app.llm.gemini_service import GeminiService
    from app.llm.generation_policy import MAX_PROMPT_CHARS

    captured: dict[str, str] = {}

    class Models:
        def generate_content(self, **kwargs):
            captured["contents"] = kwargs["contents"]
            return SimpleNamespace(
                text='{"mode":"health","answer":"Current turn handled.",'
                '"symptom_summary":"latest symptom","related_topics":[],'
                '"needs_clarification":false}'
            )

    service = GeminiService(api_key=None)
    service.client = SimpleNamespace(models=Models())
    conversation = [
        {"role": "user", "content": f"old-{index}-" + "x" * 1000} for index in range(20)
    ]
    conversation.append({"role": "user", "content": "LATEST CURRENT SYMPTOM"})

    service.generate_chat_turn(
        conversation=conversation,
        predicted_condition="Digestive Issues",
        confidence=0.9,
        prior_summary="prior",
        low_confidence=False,
    )

    assert len(captured["contents"]) <= MAX_PROMPT_CHARS
    assert "LATEST CURRENT SYMPTOM" in captured["contents"]


# --- advice_safety --------------------------------------------------------------


@pytest.mark.parametrize(
    "item",
    [
        "Give 5mg of amoxicillin twice daily.",
        "Give 2.5 ml of the syrup after meals.",
        "You should prescribe an antibiotic yourself.",
        "This confirms that your pet has diabetes.",
        "Your pet definitely has cancer.",
        "",
        "   ",
        "x" * 300,
    ],
)
def test_is_advice_item_safe_rejects_unsafe_or_malformed_items(item):
    from app.triage.advice import is_advice_item_safe

    assert is_advice_item_safe(item) is False


@pytest.mark.parametrize(
    "item",
    [
        "Ensure fresh water is always available.",
        "Keep your pet calm and resting in a quiet space.",
        "Monitor for changes in appetite or behaviour.",
        "Do not use human eye drops unless specifically prescribed by a vet.",
    ],
)
def test_is_advice_item_safe_accepts_general_care_tips(item):
    from app.triage.advice import is_advice_item_safe

    assert is_advice_item_safe(item) is True


def test_sanitize_advice_drops_unsafe_items_and_keeps_safe_ones():
    from app.triage.advice import sanitize_advice

    items = ["Give 5mg of medicine.", "Ensure fresh water is available."]
    result = sanitize_advice(items, fallback=["fallback tip"], source="test")
    assert result == ["Ensure fresh water is available."]


def test_sanitize_advice_falls_back_when_nothing_survives():
    from app.triage.advice import sanitize_advice

    items = ["Give 5mg of medicine.", ""]
    result = sanitize_advice(items, fallback=["fallback tip"], source="test")
    assert result == ["fallback tip"]


def test_sanitize_advice_caps_item_count():
    from app.triage.advice import MAX_ADVICE_ITEMS, sanitize_advice

    items = [f"Tip number {i} about general care." for i in range(MAX_ADVICE_ITEMS + 5)]
    result = sanitize_advice(items, fallback=["fallback"], source="test")
    assert len(result) == MAX_ADVICE_ITEMS


def test_all_static_condition_advice_passes_the_safety_filter():
    """Regression guard: catches an unsafe phrase accidentally added to the
    reviewed static advice table in condition_metadata.py."""
    from app.triage.advice import is_advice_item_safe

    for condition, meta in CONDITION_METADATA.items():
        for item in meta.home_advice:
            assert is_advice_item_safe(item), f"{condition!r} has an unsafe advice item: {item!r}"


def test_condition_metadata_exposes_honest_review_default():
    # No condition claims veterinary approval until a named reviewer sets one —
    # see AdviceReview docstring / design.md open question.
    for meta in CONDITION_METADATA.values():
        assert meta.review.reviewed is False
        assert meta.review.reviewer is None


# --- gemini key rotation ----------------------------------------------------


def test_rotating_client_advances_key_on_quota_exhaustion():
    from types import SimpleNamespace

    from app.llm.rotation import RotatingGeminiClient

    class FakeClientError(Exception):
        def __init__(self, code: int):
            self.code = code

    class FakeClient:
        def __init__(self, api_key: str, fail: bool):
            self.api_key = api_key
            self.fail = fail
            self.models = self

        def generate_content(self, **kwargs):
            if self.fail:
                raise FakeClientError(429)
            return f"ok-from-{self.api_key}"

    import app.llm.rotation as rotation_module

    fake_clients = {
        "key1": FakeClient("key1", fail=True),
        "key2": FakeClient("key2", fail=True),
        "key3": FakeClient("key3", fail=False),
    }
    rotation_module.ClientError = FakeClientError
    rotation_module.genai = SimpleNamespace(Client=lambda api_key: fake_clients[api_key])

    client = RotatingGeminiClient(["key1", "key2", "key3"])
    result = client.models.generate_content()

    assert result == "ok-from-key3"
    assert client._index == 2  # rotated past the two exhausted keys


def test_rotating_client_raises_when_all_keys_exhausted():
    from types import SimpleNamespace

    from app.llm.rotation import RotatingGeminiClient

    class FakeClientError(Exception):
        def __init__(self, code: int):
            self.code = code

    class FakeClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.models = self

        def generate_content(self, **kwargs):
            raise FakeClientError(429)

    import app.llm.rotation as rotation_module

    rotation_module.ClientError = FakeClientError
    rotation_module.genai = SimpleNamespace(Client=lambda api_key: FakeClient(api_key))

    client = RotatingGeminiClient(["key1", "key2"])
    with pytest.raises(FakeClientError):
        client.models.generate_content()


def test_rotating_client_reraises_non_quota_errors():
    from types import SimpleNamespace

    from app.llm.rotation import RotatingGeminiClient

    class FakeClientError(Exception):
        def __init__(self, code: int):
            self.code = code

    class FakeClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.models = self

        def generate_content(self, **kwargs):
            raise FakeClientError(500)

    import app.llm.rotation as rotation_module

    rotation_module.ClientError = FakeClientError
    rotation_module.genai = SimpleNamespace(Client=lambda api_key: FakeClient(api_key))

    client = RotatingGeminiClient(["key1", "key2"])
    with pytest.raises(FakeClientError) as exc_info:
        client.models.generate_content()
    assert exc_info.value.code == 500
    assert client._index == 0  # no rotation on a non-quota error


def test_settings_resolved_gemini_api_keys_combines_primary_and_extra(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("GEMINI_API_KEY", "primary")
    monkeypatch.setenv("GEMINI_API_KEYS", "second, third ,, primary")
    settings = Settings()

    assert settings.resolved_gemini_api_keys() == ["primary", "second", "third"]
