"""Endpoint contract + safety tests (offline, fake classifier, no Gemini)."""
from __future__ import annotations


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_basic(client):
    resp = client.post("/predict", json={"text": "My dog has been vomiting and won't eat"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predictedCondition"] == "Digestive Issues"
    assert body["confidence"] == 0.82
    assert body["urgency"] == "CONSULT_SOON"          # from condition metadata
    assert body["diseaseCategory"] == "GASTROINTESTINAL"
    assert body["homeAdvice"]                          # non-empty
    assert body["disclaimer"]


def test_predict_red_flag_forces_emergency(client, fake_predictor):
    # Classifier still says a low-urgency class, but the text is an emergency.
    fake_predictor.condition = "Digestive Issues"
    resp = client.post("/predict", json={"text": "My dog collapsed and is not breathing!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["urgency"] == "EMERGENCY"              # escalated by safety layer
    assert "⚠️" in body["explanation"]
    # Content override: emergency advice, NOT the predicted condition's advice.
    from app.services.triage_safety import EMERGENCY_HOME_ADVICE
    assert body["homeAdvice"] == EMERGENCY_HOME_ADVICE
    assert "withhold food" not in " ".join(body["homeAdvice"]).lower()  # no digestive advice


def test_predict_input_too_long_is_rejected(client):
    resp = client.post("/predict", json={"text": "x" * 5000})
    assert resp.status_code == 422                     # exceeds max_length=4000


def test_predict_empty_text_is_rejected(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422                     # min_length=1


def test_chat_basic_structured_response(client):
    resp = client.post(
        "/chat",
        json={
            "sessionId": "s1",
            "petType": "dog",
            "symptomSummary": "Vomiting since yesterday.",
            "messages": [
                {"role": "user", "content": "My dog has been vomiting since yesterday"},
                {"role": "assistant", "content": "How is he eating?"},
                {"role": "user", "content": "Now he won't eat and seems weak"},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"mode", "answer", "symptomSummary", "prediction", "needsClarification", "disclaimer"}
    # No Gemini key in tests → fallback defaults to the health (triage) path.
    assert body["mode"] == "health"
    pred = body["prediction"]
    assert pred["predictedCondition"] == "Digestive Issues"
    assert isinstance(pred["topK"], list) and pred["topK"]
    assert pred["topK"][0]["condition"] == "Digestive Issues"
    assert isinstance(body["needsClarification"], bool)


def test_chat_requires_a_user_message(client):
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "assistant", "content": "Hello, how can I help?"}]},
    )
    assert resp.status_code == 400


def test_chat_too_many_messages_rejected(client):
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(51)]
    resp = client.post("/chat", json={"messages": msgs})
    assert resp.status_code == 422                     # exceeds max_length=50


def test_chat_red_flag_escalation(client):
    from app.services.triage_safety import EMERGENCY_HOME_ADVICE

    resp = client.post(
        "/chat",
        json={
            "symptomSummary": "Mild limping yesterday.",
            "messages": [{"role": "user", "content": "He had a seizure and collapsed"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "emergency"
    assert body["prediction"]["urgency"] == "EMERGENCY"
    assert body["answer"].startswith("⚠️")                     # leads with the emergency message
    assert body["prediction"]["homeAdvice"] == EMERGENCY_HOME_ADVICE
    assert body["needsClarification"] is False                # an emergency, not a clarification
    # Rolling summary still advances (prior summary + new message), no Gemini needed.
    assert "limping" in body["symptomSummary"].lower()
    assert "seizure" in body["symptomSummary"].lower()


def test_chat_low_confidence_sets_clarification(client, fake_predictor):
    fake_predictor.confidence = 0.2                    # below ABSTAIN_THRESHOLD (0.40)
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "He seems a bit off today"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["needsClarification"] is True


def test_ask_endpoint_is_removed(client):
    # /ask was merged into the unified /chat endpoint.
    resp = client.post("/ask", json={"question": "How often should I brush a Persian cat?"})
    assert resp.status_code == 404


def test_chat_answer_never_leaks_internal_endpoints(client):
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "my cat is sneezing"}]},
    )
    assert resp.status_code == 200
    assert "/predict" not in resp.json()["answer"]


def test_wellness_partial_data_not_punished(client):
    # Healthy dog, but feeding + preventiveCare omitted. Missing dims must be
    # excluded (maxScore 0), not scored as zeros that tank the result.
    resp = client.post("/wellness", json={
        "pet": {"species": "dog", "weightKg": 28.5},
        "activity": {"avgStepsPerDay": 8000, "avgSleepHoursPerDay": 12, "daysTracked": 7},
        "previousScore": 82,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["breakdown"]["diet"]["maxScore"] == 0
    assert body["breakdown"]["preventiveCare"]["maxScore"] == 0
    assert body["wellnessScore"] >= 75            # GOOD-ish, not "CONCERNING" from missing data
    assert body["band"] in ("GOOD", "EXCELLENT")


def test_auth_enforced_when_api_key_set(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")            # read per-request by api_key_auth
    payload = {"text": "my dog is itchy"}

    assert client.post("/predict", json=payload).status_code == 403
    ok = client.post("/predict", json=payload, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
