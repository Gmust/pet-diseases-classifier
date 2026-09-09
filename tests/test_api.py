"""Endpoint contract + safety tests (offline, fake classifier, no Gemini)."""

from __future__ import annotations


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_liveness_always_ok(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_reports_loaded_model(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["backend"] == "fake"
    assert body["modelVersion"] == "test-model-v1"
    assert body["labelCount"] >= 1
    # No filesystem paths or secrets in the readiness payload.
    assert "modelPath" not in body and "path" not in body


def test_readiness_reports_not_ready_when_model_unloaded(client):
    from app import main

    main.app.state.services = None
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "not_ready", "reason": "model_not_loaded"}


def test_request_id_header_present_and_matches_error_body(client, monkeypatch, fake_predictor):
    def fail(_text):
        raise RuntimeError("boom")

    monkeypatch.setattr(fake_predictor, "predict", fail)
    resp = client.post("/predict", json={"text": "My dog is unwell"})

    assert resp.status_code == 500
    assert resp.headers["X-Request-Id"]
    assert resp.json()["requestId"] == resp.headers["X-Request-Id"]


def test_chat_openapi_example_matches_chat_contract(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["ChatResponse"]
    example = schema["example"]

    assert example["mode"] == "health"
    assert example["symptomSummary"]
    assert example["prediction"]["predictedCondition"] == "Digestive Issues"
    assert "explanation" not in example


def test_predict_basic(client):
    resp = client.post("/predict", json={"text": "My dog has been vomiting and won't eat"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predictedCondition"] == "Digestive Issues"
    assert body["confidence"] == 0.82
    assert body["urgency"] == "CONSULT_SOON"  # from condition metadata
    assert body["diseaseCategory"] == "GASTROINTESTINAL"
    assert body["homeAdvice"]  # non-empty
    assert body["disclaimer"]


def test_predict_red_flag_forces_emergency(client, fake_predictor):
    # Classifier still says a low-urgency class, but the text is an emergency.
    fake_predictor.condition = "Digestive Issues"
    resp = client.post("/predict", json={"text": "My dog collapsed and is not breathing!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["urgency"] == "EMERGENCY"  # escalated by safety layer
    assert "⚠️" in body["explanation"]
    # Content override: emergency advice, NOT the predicted condition's advice.
    from app.triage.safety import EMERGENCY_HOME_ADVICE

    assert body["homeAdvice"] == EMERGENCY_HOME_ADVICE
    assert "withhold food" not in " ".join(body["homeAdvice"]).lower()  # no digestive advice


def test_predict_input_too_long_is_rejected(client):
    resp = client.post("/predict", json={"text": "x" * 5000})
    assert resp.status_code == 422  # exceeds max_length=4000


def test_predict_empty_text_is_rejected(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422  # min_length=1


def test_predict_internal_failure_does_not_leak_exception(client, monkeypatch, fake_predictor):
    def fail(_text):
        raise RuntimeError("sensitive model path: /private/model.safetensors")

    monkeypatch.setattr(fake_predictor, "predict", fail)
    resp = client.post("/predict", json={"text": "My dog is unwell"})

    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "Prediction failed."
    assert body["requestId"]
    assert "private" not in resp.text


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
    assert set(body) >= {
        "mode",
        "answer",
        "symptomSummary",
        "prediction",
        "needsClarification",
        "disclaimer",
    }
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


def test_chat_rejects_trailing_assistant_without_running_classifier(client, fake_predictor):
    resp = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "My cat is sneezing"},
                {"role": "assistant", "content": "Does she have a runny nose?"},
            ]
        },
    )

    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"] == "The last message must be a non-empty user message."
    assert body["requestId"]
    assert fake_predictor.last_input is None


def test_chat_internal_failure_does_not_leak_exception(client, monkeypatch, fake_predictor):
    def fail(_text, k=3):
        raise RuntimeError("provider token and internal model details")

    monkeypatch.setattr(fake_predictor, "predict_top_k", fail)
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "My cat is unwell"}]},
    )

    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "Prediction failed."
    assert body["requestId"]
    assert "provider token" not in resp.text


def test_chat_too_many_messages_rejected(client):
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(51)]
    resp = client.post("/chat", json={"messages": msgs})
    assert resp.status_code == 422  # exceeds max_length=50


def test_chat_red_flag_escalation(client):
    from app.triage.safety import EMERGENCY_HOME_ADVICE

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
    assert body["answer"].startswith("⚠️")  # leads with the emergency message
    assert body["prediction"]["homeAdvice"] == EMERGENCY_HOME_ADVICE
    assert body["needsClarification"] is False  # an emergency, not a clarification
    # Rolling summary still advances (prior summary + new message), no Gemini needed.
    assert "limping" in body["symptomSummary"].lower()
    assert "seizure" in body["symptomSummary"].lower()


def test_chat_red_flag_in_prior_summary_forces_emergency(client):
    resp = client.post(
        "/chat",
        json={
            "symptomSummary": "The cat had a seizure and collapsed moments ago.",
            "messages": [{"role": "user", "content": "What should I do now?"}],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["mode"] == "emergency"
    assert resp.json()["prediction"]["urgency"] == "EMERGENCY"


def test_chat_low_confidence_sets_clarification(client, fake_predictor):
    fake_predictor.confidence = 0.2  # below ABSTAIN_THRESHOLD (0.40)
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


def test_wellness_partial_data_below_reliability_gate_is_explicit(client):
    # Healthy activity alone is not enough for a reliable numeric score.
    resp = client.post(
        "/wellness",
        json={
            "pet": {"species": "dog", "weightKg": 28.5},
            "activity": {"avgStepsPerDay": 8000, "avgSleepHoursPerDay": 12, "daysTracked": 7},
            "previousScore": 82,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["breakdown"]["diet"]["availability"] == "MISSING"
    assert body["breakdown"]["diet"]["included"] is False
    assert body["breakdown"]["preventiveCare"]["availability"] == "MISSING"
    assert body["scoreStatus"] == "INSUFFICIENT_DATA"
    assert body["wellnessScore"] is None
    assert body["band"] is None


def test_wellness_species_only_returns_insufficient_data_contract(client):
    resp = client.post("/wellness", json={"pet": {"species": "cat"}})

    assert resp.status_code == 200
    body = resp.json()
    assert body["scoreStatus"] == "INSUFFICIENT_DATA"
    assert body["wellnessScore"] is None
    assert body["dataCoverage"] == 0


def test_wellness_rejects_implausible_numeric_values(client):
    resp = client.post(
        "/wellness",
        json={
            "pet": {"species": "dog", "weightKg": -4},
            "activity": {"avgSleepHoursPerDay": 25},
            "feeding": {"consistencyDays": 8},
        },
    )

    assert resp.status_code == 422
    fields = {error["loc"][-1] for error in resp.json()["detail"]}
    assert {"weightKg", "avgSleepHoursPerDay", "consistencyDays"} <= fields


def test_feeding_summary_batches_multiple_pets(client):
    resp = client.post(
        "/feeding-summary",
        json={
            "pets": [
                {
                    "petId": "pet-1",
                    "species": "dog",
                    "breed": "Labrador",
                    "weightKg": 28.5,
                    "products": [{"name": "kibble", "calories": 900}],
                },
                {
                    "petId": "pet-2",
                    "species": "cat",
                    "weightKg": 4.5,
                    "products": [],
                },
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert [r["petId"] for r in body["results"]] == ["pet-1", "pet-2"]
    assert body["results"][1]["status"] == "EXTREME_UNDER_TARGET"  # no food logged at all
    assert body["results"][1]["actualCalories"] == 0
    assert "disclaimer" in body


def test_feeding_summary_status_bands(client):
    # weight 28.5kg dog -> RER=70*28.5^0.75≈828.9, MER=*1.6≈1326.3
    resp = client.post(
        "/feeding-summary",
        json={
            "pets": [
                {
                    "petId": "on-target",
                    "species": "dog",
                    "weightKg": 28.5,
                    "products": [{"name": "kibble", "calories": 1326}],
                },
                {
                    "petId": "over-target",
                    "species": "dog",
                    "weightKg": 28.5,
                    "products": [{"name": "treats", "calories": 2000}],
                },
                {
                    "petId": "extreme-over-target",
                    "species": "dog",
                    "weightKg": 1,  # target=112 kcal -> 20000 actual is wildly over
                    "products": [{"name": "mega bag", "calories": 20000}],
                },
            ]
        },
    )

    assert resp.status_code == 200
    results = {r["petId"]: r for r in resp.json()["results"]}
    assert results["on-target"]["status"] == "ON_TARGET"
    assert results["over-target"]["status"] == "OVER_TARGET"
    assert results["extreme-over-target"]["status"] == "EXTREME_OVER_TARGET"
    assert results["extreme-over-target"]["deviationPct"] > 100


def test_feeding_summary_rejects_empty_pet_list(client):
    resp = client.post("/feeding-summary", json={"pets": []})

    assert resp.status_code == 422


def test_feeding_summary_juvenile_gets_growth_factor_not_adult_factor(client):
    # 1kg dog, no age -> adult factor 1.6 -> target=70*1^0.75*1.6=112
    # 1kg puppy, age 3mo -> juvenile factor 2.5 -> target=70*1^0.75*2.5=175
    resp = client.post(
        "/feeding-summary",
        json={
            "pets": [
                {"petId": "adult", "species": "dog", "weightKg": 1, "products": []},
                {
                    "petId": "puppy",
                    "species": "dog",
                    "weightKg": 1,
                    "ageMonths": 3,
                    "products": [],
                },
            ]
        },
    )

    assert resp.status_code == 200
    results = {r["petId"]: r for r in resp.json()["results"]}
    assert results["adult"]["targetCalories"] == 112.0
    assert results["puppy"]["targetCalories"] == 175.0


def test_auth_enforced_when_api_key_set(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")  # read per-request by api_key_auth
    payload = {"text": "my dog is itchy"}

    assert client.post("/predict", json=payload).status_code == 403
    ok = client.post("/predict", json=payload, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
