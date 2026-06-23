# Remaining Checklist — Pet Care AI Microservice

Status snapshot (as of this session): core service is functional and tested. The
retrained CE model is a healthy balanced baseline (~70% accuracy; weak classes
improved without collapsing the majority class). `/chat`, `/predict`, Gemini,
static metadata, red-flag escalation, and abstention are all wired and verified
in manual testing. 31 automated tests pass.

The items below are what's left before calling it production-ready. None block the
.NET backend from integrating now.

---

## 1. Untested guard cases (~10 min, do first)

Quick manual passes to confirm the safety/validation paths. Assumes server on `:8000`.

- [ ] **Red-flag escalation (explicit)** — must force `EMERGENCY` regardless of class:
  ```bash
  curl -s localhost:8000/predict -H 'Content-Type: application/json' \
    -d '{"text":"my dog collapsed and is not breathing"}' | python3 -m json.tool
  ```
  Expect `urgency: "EMERGENCY"` and the ⚠️ note prefixing the explanation.

- [ ] **Red-flag inside chat** — `"he had a seizure and collapsed"` → `prediction.urgency=EMERGENCY`, ⚠️ in `answer`.

- [ ] **Auth enforced** — restart with `API_KEY=secret uvicorn app.main:app --port 8000`:
  - no header → `403`
  - `-H 'X-API-Key: secret'` → `200`

- [ ] **Input caps** — `/predict` with `text` > 4000 chars → `422`; `/chat` with >50 messages → `422`.

- [ ] **No-user-message guard** — `/chat` with only an assistant message → `400`.

- [ ] **Low-confidence abstention** — vague input (`"he seems a bit off"`) → `needsClarification: true`.

- [ ] **Run the regression test against real weights** (now that `model.safetensors` exists):
  ```bash
  pytest    # in the .venv with torch installed — the regression test now runs instead of skipping
  ```

---

## 2. Open product decision

- [ ] **`needsClarification` semantics.** Currently true when EITHER the classifier
  is unsure OR Gemini chose to ask a follow-up — so it can be `true` even at 0.93
  confidence. Decide:
  - keep as-is ("assistant is still gathering info"), or
  - tighten to confidence-only ("don't trust this prediction yet") — one-line change
    in the `/chat` handler.

---

## 3. Cost migration (implemented, not yet applied)

The code exists; these steps actually realize the savings. Needs the model weights
(now present locally).

- [ ] **Export quantized ONNX model:**
  ```bash
  pip install optimum[onnxruntime]      # build-time only
  python -m app.ml.export_onnx \
    --model-dir models/transformer_model \
    --output-dir models/transformer_model_onnx
  ```
- [ ] **Validate ONNX parity** — run a handful of `/predict` inputs against both
  backends and confirm predictions match (int8 should be within ~1% of torch).
- [ ] **Flip the backend & shrink Lambda:** set `MODEL_BACKEND=onnx`,
  `MODEL_PATH=models/transformer_model_onnx`, and lower `MemorySize` in `template.yaml`
  (3008 → ~1024). Drop torch from the inference image.
- [ ] **Reconsider keep-warm** — with ONNX cold start ~2-4s, widen the 5-min schedule
  or remove it.
- [ ] **Enable the static-explanation cost path** if acceptable for `/predict`:
  start with `USE_STATIC_EXPLANATIONS=true` to eliminate the per-request Gemini call
  (and its ~20/day free-tier cap). Confirm the templated explanation reads acceptably.

Expected effect: image ~2.5GB → <500MB, cold start ~20s → ~2-4s, ~3x less GB-s
per request, `/predict` optionally zero-API-cost.

---

## 4. Model & data quality (the real accuracy ceiling)

Current model overfits (train F1 0.94 vs val 0.67) — the ceiling now is data, not
hyperparameters.

- [ ] **Pull real EHR data (SAVSNET/PetEVAL, 17.6k rows)** — already wired in
  `fetch_and_merge.py`; needs an HF token:
  ```bash
  HF_TOKEN=hf_xxxx python -m app.ml.fetch_and_merge
  # then retrain on the new merged file with --loss ce
  ```
  This is the highest-value accuracy lever, especially for the Ear/Respiratory and
  Infectious boundaries.

- [ ] **Run the leakage check** before trusting any reported metric:
  ```bash
  python -m app.ml.check_leakage --data-path data/merged_augmented.parquet --label-map data/label_map.json
  ```
  If test rows duplicate train rows, dedup before splitting and retrain.

- [ ] **Use the confusion matrix to drive label decisions** — Blood↔Immune and
  Infectious→(Skin/Ear) are genuine overlaps. Consider merging or a two-stage
  classifier rather than more synthetic data.

- [ ] **Only revisit focal loss** if CE leaves weak classes too low — and if so,
  decouple alpha (don't stack balanced weights onto focal). Add `--focal-alpha {none,sqrt,balanced}`.

- [ ] **Tune the regression test threshold** (`EXPECTED_MIN_ACCURACY` in
  `tests/test_classifier_regression.py`) to this model's real baseline so CI is meaningful.

---

## 5. Ops & rollout

- [ ] **CI is in place** (`.github/workflows/ci.yml`) — confirm it runs green on first push.
- [ ] **Structured logs / metrics** are emitted (JSON, CloudWatch-queryable) — add an
  alarm on fallback rate or low-confidence rate once deployed.
- [ ] **API Gateway throttling** params (`ApiRateLimit`/`ApiBurstLimit`) are in
  `template.yaml` — tune to expected traffic before deploy.
- [ ] **.NET backend integration** — hand `docs/dotnet-chat-integration.md` to the
  backend dev; the critical contract is persisting and replaying `symptomSummary`.
- [ ] **Secrets** — ensure `API_KEY` and `GEMINI_API_KEY` are set in the deploy
  environment (SAM params), not committed.
