# Pet Care AI Microservice

Architecture and accepted design decisions are documented in
[`docs/architecture.md`](docs/architecture.md) and [`docs/adr/`](docs/adr/).
Contributor, security, privacy, and operational procedures are in
[`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md),
[`docs/privacy-data-handling.md`](docs/privacy-data-handling.md), and
[`docs/runbooks/`](docs/runbooks/).

FastAPI microservice providing three AI-powered endpoints for pet health assessment, general pet-care Q&A, and wellness scoring. Deployed on AWS Lambda via AWS SAM.

---

## Architecture

```
Client
  │
  ├─ POST /predict   → local transformer classifier (current bundle: DistilBERT)
  │                    + Gemini explanation + home advice
  │
  ├─ POST /chat      → emergency rules + local classifier
  │                    + Gemini general/health routing and answer
  │
  └─ POST /wellness  → Rule-based scoring across 6 dimensions
                       + Gemini narrative + recommendations
```

**Design principles:**
- The **classifier** is the sole decision-maker for condition prediction — Gemini cannot override it.
- **Gemini** generates human-friendly text only (explanations, advice, narratives).
- Every response includes a medical/wellness disclaimer.
- Missing data is reported explicitly through weighted coverage and assessment status;
  partial scores remain proportional only after the minimum reliability gate is met.

---

## Project Structure

```
app/
  main.py                        # FastAPI app, lifespan, all route handlers
  schemas.py                     # All Pydantic request/response models + enums
  lambda_handler.py              # AWS Lambda entry point (Mangum wrapper)
  ml/
    train.py                     # Fine-tune a configured Hugging Face classifier
    predictor.py                 # Load model + run inference
    onnx_predictor.py            # Quantized ONNX inference backend
    condition_metadata.py        # Condition → urgency / specialist / category / advice
    fetch_and_merge.py           # Fetch external HF datasets + merge
    generate_synthetic.py        # Generate synthetic training data via Gemini
    prepare_dataset.py           # Deduplicate, rebalance, and build owner holdout
    evaluate.py                  # Evaluate Torch or ONNX model on a holdout
  services/
    gemini_service.py            # Explanation + home advice generation
    chat_context.py              # Bounded classifier and emergency context
    triage_safety.py             # Deterministic emergency and abstention rules
    wellness_service.py          # Wellness scoring engine + Gemini narrative
data/
  merged_pet_dataset.parquet     # Base training data
  merged_augmented.parquet       # Augmented (after fetch_and_merge)
  label_map.json                 # Label consolidation map (23 → 16 classes)
models/
  transformer_model/             # Fine-tuned model (HuggingFace format)
Dockerfile                       # Cloud Run / local container
Dockerfile.lambda                # AWS Lambda container (python:3.11-slim + awslambdaric)
template.yaml                    # AWS SAM deployment template
requirements.txt                 # Runtime dependencies (API server + inference)
requirements-train.txt           # Training-only dependencies (scikit-learn, datasets)
.env.example
```

---

## Local Development

### Setup

Recommended reproducible setup with uv:

```bash
uv sync --extra dev --extra torch-runtime  # local API development + full local tests
uv sync --extra onnx-runtime               # ONNX-only runtime
uv sync --extra train                       # training pipeline
uv sync --extra export                      # ONNX export tooling
```

`uv.lock` pins all profiles. Use `uv lock --upgrade` only in a dedicated
dependency-update change and run the relevant test/model workflow afterward.

Legacy pip setup remains available while Dockerfiles migrate to lock exports:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment

Copy `.env.example` to `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here   # from aistudio.google.com/apikey
GEMINI_MODEL=gemini-2.5-flash-lite         # free: 20 req/day | paid: gemini-2.5-flash
API_KEY=your_secret_api_key_here           # X-API-Key header auth (leave empty to disable)
MODEL_PATH=models/transformer_model        # path to fine-tuned model directory
MODEL_BACKEND=torch                        # torch or onnx
LOW_CONFIDENCE_THRESHOLD=0.65              # below this, appends low-confidence warning
USE_STATIC_EXPLANATIONS=false              # skip Gemini for /predict when true
```

### Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs

### Quality commands

Install the development profile, then use the same entry points as CI:

```bash
make format       # apply Black and isort
make lint         # Ruff + formatting checks
make typecheck    # mypy
make contracts    # OpenAPI fingerprint + SAM template validation
make test         # full locally available test suite
make quality      # lint + typecheck + contracts + tests
make pre-commit   # run all pre-commit hooks against the repository
```

---

## Security

- Set `API_KEY` to require `X-API-Key` header on all `POST` endpoints.
- If `API_KEY` is unset, auth is skipped (useful for local development).
- `GET /health` is always open (required for load balancer health checks).
- Auth uses `hmac.compare_digest` to prevent timing attacks.

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_api_key_here" \
  -d '{"text": "My cat has been scratching her ears"}'
```

---

## API Reference

### `GET /health`

```json
{ "status": "ok" }
```

---

### `POST /predict`

Classifies pet symptoms into one of 16 conditions using the fine-tuned transformer. Gemini generates the explanation and home-care advice.

**Request:**
```json
{ "text": "My dog has been vomiting and not eating for 2 days" }
```

**Response:**
```json
{
  "predictedCondition": "Digestive Issues",
  "confidence": 0.88,
  "explanation": "The symptoms are most consistent with a digestive condition...",
  "disclaimer": "This is an AI-assisted pre-assessment and not a veterinary diagnosis.",
  "urgency": "CONSULT_SOON",
  "specialist": "general_vet",
  "diseaseCategory": "GASTROINTESTINAL",
  "homeAdvice": [
    "Withhold food for 12-24 hours (water only) to rest the stomach.",
    "Offer small portions of bland food: boiled chicken and plain rice.",
    "Feed 2-3 small meals per day instead of one large meal.",
    "Ensure fresh water is always available.",
    "Watch for blood in vomit or stool — seek emergency care if present."
  ]
}
```

**Enum values:**

`urgency` — `MONITOR` | `CONSULT_SOON` | `URGENT` | `EMERGENCY`

`specialist` — `general_vet` | `dermatologist` | `neurologist` | `cardiologist` | `oncologist` | `ophthalmologist` | `internist` | `surgeon` | `emergency_vet`

`diseaseCategory` — `INFECTIOUS` | `METABOLIC` | `STRUCTURAL` | `NEOPLASTIC` | `IMMUNE` | `NEUROLOGICAL` | `CARDIOVASCULAR` | `DERMATOLOGICAL` | `GASTROINTESTINAL` | `RESPIRATORY` | `OPHTHALMIC` | `UROGENITAL` | `TRAUMA` | `HEMATOLOGICAL` | `REPRODUCTIVE` | `EAR`

**Low confidence** (below `LOW_CONFIDENCE_THRESHOLD`): appends a caution note to `explanation`.

---

### `POST /chat`

Unified stateless conversation endpoint for general pet-care questions and health
triage. The response `mode` is `general`, `health`, or `emergency`; callers persist
and replay `symptomSummary` between turns. The final message must be a non-empty
user message. See [the API reference](docs/api-reference.md#post-chat) and
[the .NET integration guide](docs/dotnet-chat-integration.md) for the complete contract.

---

### `POST /wellness`

Wellness indicator (0–100) derived from tracked activity, feeding, and care data. Designed to be called automatically by the backend using aggregated database records — no manual user input required.

**Scoring dimensions:**

| Dimension | Max | Source |
|---|---|---|
| Activity | 20 | `ActivityDailies` — steps + active minutes vs species norms |
| Sleep | 15 | `ActivityDailies` — sleep hours vs species norms |
| Diet | 20 | `FeedingLogs` — meal consistency, food variety, calorie fit |
| Symptoms | 25 | Optional free text → transformer classifier |
| Preventive care | 10 | `PetEvents` + reminder runs — vet visit, vaccinations, medication adherence |
| Baseline | 10 | Pet age + weight-history stability |

Missing dimensions are scaled out — partial data is always accepted. The response
uses `scoreStatus` to distinguish `COMPLETE`, `PARTIAL`, and `INSUFFICIENT_DATA`.
`dataCoverage` is the sum of maxima for `AVAILABLE` dimensions divided by the sum
of maxima for all `AVAILABLE` or `MISSING` dimensions, rounded to four decimal
places. `NOT_APPLICABLE` dimensions are excluded from both sides.

Each breakdown item includes:

- `availability`: `AVAILABLE`, `MISSING`, or `NOT_APPLICABLE`.
- `included`: compatibility field; true exactly when availability is `AVAILABLE`.
- `reasonCodes`: stable uppercase machine-readable explanations.
- `evidence`: only allowlisted scalar values used by the deterministic rule.

Evidence never includes raw symptom text, medication names, or free-form behavioral
notes. No symptom text means the symptom dimension is `NOT_APPLICABLE`; users are
not asked to invent symptoms to complete the score.

A numeric score is returned only when all of these reliability conditions are met:

- `dataCoverage >= 0.60`.
- At least three foundational dimensions are `AVAILABLE` among Activity, Sleep,
  Diet, Preventive care, and Baseline.
- Diet and PreventiveCare-or-Baseline are available.
- Activity-or-Sleep is available when either applies to the species. Species for
  which both are `NOT_APPLICABLE` are not blocked by that group.

If the gate is not met, the request still succeeds but returns
`INSUFFICIENT_DATA` and null score/band fields. Symptoms remain optional.

Medication adherence is calculated as completed doses divided by scheduled doses
and contributes 20% of the preventive-care dimension when dose totals are supplied.
Weight stability compares the oldest and newest `weightHistory` measurements:
changes up to 3% receive full credit, up to 5% receive 80%, up to 10% receive 40%,
and larger changes receive no stability credit.

Active chronic conditions cap the maximum possible score:
- Serious conditions (cancer, heart failure): max 65
- Moderate conditions (diabetes, kidney disease): max 75
- Mild conditions (arthritis, allergies): max 85

**Score bands:**

| Score | Band | Label |
|---|---|---|
| 90–100 | `EXCELLENT` | Excellent |
| 75–89 | `GOOD` | Good |
| 60–74 | `FAIR` | Fair |
| 40–59 | `CONCERNING` | Concerning |
| 0–39 | `CRITICAL` | Critical |

**Request:**
```json
{
  "pet": {
    "species": "dog",
    "breed": "Labrador",
    "ageMonths": 36,
    "weightKg": 28.5
  },
  "activity": {
    "avgStepsPerDay": 8500,
    "avgActiveMinutesPerDay": 45,
    "avgSleepHoursPerDay": 13.0,
    "daysTracked": 7
  },
  "feeding": {
    "avgMealsPerDay": 2.0,
    "avgCaloriesPerDay": 980,
    "foodTypes": ["dry_kibble", "wet_food"],
    "consistencyDays": 7
  },
  "activeConditions": [
    { "name": "hip dysplasia", "typeLabel": "musculoskeletal" }
  ],
  "activeMedications": [
    {
      "name": "Carprofen",
      "frequency": "daily",
      "scheduledDoses": 14,
      "completedDoses": 10
    }
  ],
  "weightHistory": [
    { "weightKg": 28.7, "measuredAt": "2026-06-30T08:00:00Z" },
    { "weightKg": 28.5, "measuredAt": "2026-07-30T08:00:00Z" }
  ],
  "preventiveCare": {
    "recentVetVisit": true,
    "vaccinationsUpToDate": true
  },
  "routineCare": [
    { "type": "Bathing", "lastDoneAt": "2026-07-20" },
    { "type": "NailTrimming", "lastDoneAt": "2026-05-02" }
  ],
  "evaluationWindow": {
    "startDate": "2026-07-24",
    "endDate": "2026-07-30"
  },
  "currentSymptoms": "slightly lethargic lately",
  "previousScore": 80
}
```

**Response:**
```json
{
  "wellnessScore": 85,
  "band": "GOOD",
  "bandLabel": "Good",
  "scoreStatus": "COMPLETE",
  "dataCoverage": 1.0,
  "calculationVersion": "2.0.0",
  "evaluatedAt": "2026-07-31T08:45:12.123456Z",
  "evaluationWindow": {
    "startDate": "2026-07-24",
    "endDate": "2026-07-30"
  },
  "trend": "STABLE",
  "breakdown": {
    "activity": {
      "score": 20.0,
      "maxScore": 20.0,
      "availability": "AVAILABLE",
      "included": true,
      "reasonCodes": ["ACTIVITY_TARGET_MET"],
      "evidence": {
        "avgStepsPerDay": 8500.0,
        "stepTarget": 8000,
        "avgActiveMinutesPerDay": 45.0,
        "activeMinuteTarget": 45
      }
    },
    "sleep": {
      "score": 15.0,
      "maxScore": 15.0,
      "availability": "AVAILABLE",
      "included": true,
      "reasonCodes": ["SLEEP_WITHIN_RANGE"],
      "evidence": {
        "avgSleepHoursPerDay": 13.0,
        "healthyMinimumHours": 12.0,
        "healthyMaximumHours": 14.0
      }
    },
    "diet": {
      "score": 20.0,
      "maxScore": 20.0,
      "availability": "AVAILABLE",
      "included": true,
      "reasonCodes": ["DIET_TRACKING_STRONG"],
      "evidence": {
        "consistencyDays": 7,
        "foodTypeCount": 2,
        "avgMealsPerDay": 2.0,
        "avgCaloriesPerDay": 980.0,
        "weightKg": 28.5,
        "calorieTargetPerDay": 997.5,
        "calorieRatio": 0.9825
      }
    },
    "symptoms": {
      "score": 15.3,
      "maxScore": 25.0,
      "availability": "AVAILABLE",
      "included": true,
      "reasonCodes": ["SYMPTOM_RESULT_AVAILABLE"],
      "evidence": {
        "confidence": 0.9,
        "urgency": "CONSULT_SOON"
      }
    },
    "preventiveCare": {
      "score": 9.4,
      "maxScore": 10.0,
      "availability": "AVAILABLE",
      "included": true,
      "reasonCodes": ["PREVENTIVE_CARE_CURRENT"],
      "evidence": {
        "recentVetVisit": true,
        "vaccinationsUpToDate": true,
        "medicationAdherence": 0.7143
      }
    },
    "baseline": {
      "score": 10.0,
      "maxScore": 10.0,
      "availability": "AVAILABLE",
      "included": true,
      "reasonCodes": ["BASELINE_STABLE"],
      "evidence": {
        "ageMonths": 36,
        "weightMeasurementCount": 2,
        "weightStability": 1.0
      }
    }
  },
  "conditionCap": 85,
  "classifierCondition": "Musculoskeletal Conditions",
  "narrative": "Your Labrador is doing well overall with excellent activity and sleep scores...",
  "recommendations": [
    "Continue current exercise routine with gentle low-impact activity.",
    "Monitor for increased stiffness after rest — note timing and duration."
  ],
  "reminders": [
    {
      "reminder": "Medication",
      "text": "Check the existing medication schedule recorded for your pet."
    }
  ],
  "trackingRecommendations": [
    {
      "dimension": "Activity",
      "text": "Keep tracking daily activity to maintain reliable wellness trends.",
      "requiredInputs": [
        "activity.avgStepsPerDay",
        "activity.avgActiveMinutesPerDay"
      ],
      "suggestedReminderTypes": ["Activity"]
    },
    {
      "dimension": "Diet",
      "text": "Keep logging meals consistently to maintain reliable wellness trends.",
      "requiredInputs": [
        "feeding.avgMealsPerDay",
        "feeding.consistencyDays"
      ],
      "suggestedReminderTypes": ["Feeding"]
    },
    {
      "dimension": "PreventiveCare",
      "text": "Keep vaccination and veterinary-visit records current for reliable wellness trends.",
      "requiredInputs": [
        "preventiveCare.vaccinationsUpToDate",
        "preventiveCare.recentVetVisit"
      ],
      "suggestedReminderTypes": ["Vaccination", "VetVisit"]
    }
  ],
  "disclaimer": "This wellness indicator is based on tracked activity, feeding, and care data. It is not a clinical assessment and does not replace a veterinary examination."
}
```

**Partial response example:**

```json
{
  "wellnessScore": 100,
  "band": "EXCELLENT",
  "bandLabel": "Excellent",
  "scoreStatus": "PARTIAL",
  "dataCoverage": 0.6667,
  "trackingRecommendations": [
    {
      "dimension": "Sleep",
      "text": "Track daily sleep duration to include sleep in the wellness score.",
      "requiredInputs": ["activity.avgSleepHoursPerDay"],
      "suggestedReminderTypes": []
    },
    {
      "dimension": "PreventiveCare",
      "text": "Record vaccination and recent veterinary-visit status to include preventive care.",
      "requiredInputs": [
        "preventiveCare.vaccinationsUpToDate",
        "preventiveCare.recentVetVisit"
      ],
      "suggestedReminderTypes": ["Vaccination", "VetVisit"]
    }
  ]
}
```

This example assumes Activity, Diet, and Baseline are available. The partial score
is normalized only across available dimensions after the reliability gate passes.
Consumers should show the coverage/status alongside it rather than presenting it
as a complete assessment.

**Insufficient-data response behavior:**

```json
{
  "wellnessScore": null,
  "band": null,
  "bandLabel": null,
  "scoreStatus": "INSUFFICIENT_DATA",
  "dataCoverage": 0.2667,
  "narrative": "There is not enough tracked data to calculate a wellness score yet. Record the suggested activity, feeding, preventive-care, or baseline details and request a new assessment.",
  "recommendations": [],
  "reminders": [],
  "trackingRecommendations": [
    {
      "dimension": "Activity",
      "text": "Track daily steps or active minutes to include activity in the wellness score.",
      "requiredInputs": [
        "activity.avgStepsPerDay",
        "activity.avgActiveMinutesPerDay"
      ],
      "suggestedReminderTypes": ["Activity"]
    }
  ]
}
```

This example represents otherwise strong feeding data without enough independent
wellness context. Gemini and score-derived reminders are skipped when status is
`INSUFFICIENT_DATA`. `trackingRecommendations` is deterministic, ordered by
dimension, and intentionally separate from health `recommendations` and care
`reminders`.

Each tracking item also returns `suggestedReminderTypes` using the C# backend's
exact `ReminderType` wire values. These values let the client offer a separate
“Create reminder” action after explicit user confirmation:

| Tracking dimension | Suggested reminder types |
|---|---|
| Activity | `Activity` |
| Sleep | none — no matching backend type |
| Diet | `Feeding` |
| PreventiveCare | `Vaccination`, `VetVisit` |
| Baseline | `Weighing` |
| RoutineCare | `Bathing`, `Brushing`, `NailTrimming`, `EarCleaning`, `PawCare`, `TeethCleaning` |

`RoutineCare` is tracking-only — it is never scored and never changes `dataCoverage`.
It is returned whenever the backend sends no `routineCare` record for a grooming
activity, or the record is older than 30 days (anchored to `evaluationWindow.endDate`
when provided, otherwise today). A recent unspecified `Grooming` record counts for
every grooming label. `Deworming` and other parasite products are never suggested.

The microservice does not create or schedule reminders and does not return repeat
frequency, time, or notification settings. Those values must be collected by the
client/backend reminder flow. `Medication` is never suggested for wellness tracking.

For complete `GOOD` or `EXCELLENT` assessments, `trackingRecommendations` contains
positive maintenance guidance for Activity, Diet, and PreventiveCare. It tells the
owner what to keep tracking even though the underlying data is already present.
Complete `FAIR`, `CONCERNING`, and `CRITICAL` assessments omit maintenance guidance
so health recommendations remain prominent. Suggested reminder types already
present in `reminders` are removed to avoid duplicate calls to action.

**Calculation and consumer compatibility:**

- `calculationVersion` starts at `2.0.0`. Changes to scoring, coverage,
  availability, or reason-code meaning require a version increment.
- `evaluatedAt` is always a timezone-aware UTC timestamp.
- `evaluationWindow` is optional, inclusive, echoed unchanged, and rejected when
  `startDate` is after `endDate`.
- Version 2 is breaking: C# DTOs must make `WellnessScore`, `Band`, and
  `BandLabel` nullable; add status, coverage, calculation metadata, availability,
  reason-code, evidence, and tracking-recommendation fields; and handle
  `INSUFFICIENT_DATA` without mapping null to zero.
- The C# tracking-recommendation DTO should deserialize
  `SuggestedReminderTypes` as a collection of the existing `ReminderType` enum.
  An empty collection means the current backend enum has no safe semantic match.
- C# and mobile consumers must not infer score eligibility from the presence of
  any single breakdown item; `scoreStatus` is authoritative and sparse but
  evaluable requests can intentionally return a null score.
- C# consumers should treat reason-code strings as an extensible contract and
  tolerate unknown future values. `included` remains a compatibility field but
  should be derived from or checked against `availability`.

Each item in `reminders` combines actionable text with the C# backend's exact
`ReminderType` wire value:
`Feeding`, `Activity`, `Medication`, `Vaccination`, `ParasiteTreatment`,
`VetVisit`, and `Grooming`. Scheduling values such as `Daily`, `Weekly`,
`Monthly`, and `Once` belong to the backend's separate `RepeatType` enum and
are never returned here. Low diet/activity scores produce `Feeding`/`Activity`;
adherence below 90% produces `Medication`; incomplete vaccinations produce
`Vaccination`; and an overdue visit or concerning symptom score produces
`VetVisit`. Recommendation text does not repeat actions already represented by
`reminders`, and medication, dosing, parasite-product, and supplement advice is
excluded. Medication names are never copied into generated output; adherence can
only produce a generic reminder about an existing backend-recorded schedule.
When no medication schedule exists, a classifier result with `CONSULT_SOON`,
`URGENT`, or `EMERGENCY` urgency can only produce a `VetVisit` reminder to discuss
whether clinical treatment is needed—it never proposes a drug.
Weight tracking guidance remains in `recommendations`.

---

## Label Classes (16 conditions)

| Condition | Urgency | Specialist | Category |
|---|---|---|---|
| Digestive Issues | CONSULT_SOON | general_vet | GASTROINTESTINAL |
| Infectious and Parasitic Diseases | URGENT | general_vet | INFECTIOUS |
| Musculoskeletal Conditions | CONSULT_SOON | surgeon | STRUCTURAL |
| Skin Conditions | MONITOR | dermatologist | DERMATOLOGICAL |
| Ear Conditions | MONITOR | general_vet | EAR |
| Neoplasms | CONSULT_SOON | oncologist | NEOPLASTIC |
| Neurological and Behavioural Disorders | URGENT | neurologist | NEUROLOGICAL |
| Metabolic and Endocrine Disorders | CONSULT_SOON | internist | METABOLIC |
| Eye Conditions | CONSULT_SOON | ophthalmologist | OPHTHALMIC |
| Respiratory Conditions | URGENT | general_vet | RESPIRATORY |
| Cardiovascular Conditions | URGENT | cardiologist | CARDIOVASCULAR |
| Immune System Disorders | CONSULT_SOON | internist | IMMUNE |
| Genitourinary Conditions | CONSULT_SOON | internist | UROGENITAL |
| Injury and Poisoning | EMERGENCY | emergency_vet | TRAUMA |
| Blood Disorders | URGENT | internist | HEMATOLOGICAL |
| Reproductive Conditions | CONSULT_SOON | general_vet | REPRODUCTIVE |

---

## Training Pipeline

Install training dependencies:
```bash
pip install -r requirements.txt -r requirements-train.txt
```

### Step 1 — Generate synthetic data for weak classes

```bash
python -m ml_pipeline.generate_synthetic --samples-per-class 100
```

Output: `data/synthetic_data.parquet`

### Step 2 — Merge all data sources

```bash
python -m ml_pipeline.fetch_and_merge
```

Combines: local base data + VetPetCare (free HF dataset) + synthetic data.
Output: `data/merged_augmented.parquet`

### Step 3 — Train

```bash
python -m ml_pipeline.train \
  --data-path data/merged_augmented.parquet \
  --label-map data/label_map.json \
  --model-dir models/transformer_model \
  --base-model emilyalsentzer/Bio_ClinicalBERT \
  --epochs 8 \
  --batch-size 16 \
  --lr 2e-5
```

**Mac Apple Silicon:** MPS is auto-detected, no flags needed.

**Optional upgrade — PetBERT** (gated, needs HuggingFace account):
```bash
HF_TOKEN=hf_xxxx python -m ml_pipeline.train \
  --base-model SAVSNET/PetBERT \
  --epochs 8 --batch-size 16 --lr 1e-5
```

---

## Deployment — AWS Lambda (Free Tier)

**Free tier:** 1M requests/month + 400K GB-seconds compute/month (permanent).

### Prerequisites

```bash
# AWS CLI
brew install awscli && aws configure

# AWS SAM CLI
brew install aws-sam-cli

# Docker (must be running)
docker info
```

### Deploy

```bash
# First time — interactive, saves settings to samconfig.toml
sam build && sam deploy --guided

# Subsequent deploys after code changes or retraining
sam build && sam deploy
```

### View logs

```bash
sam logs --name PetCareAiFunction --stack-name pet-care-ai --tail
```

### Tear down

```bash
sam delete --stack-name pet-care-ai
```

### Cost estimate

| Usage | Cost |
|---|---|
| Idle (no traffic) | $0 — scales to zero |
| 1,000 requests/day, ~2s/request, 3 GB | ~$0 — within free tier |
| Free tier limits | 1M req + 400K GB-seconds/month (permanent) |

---

## Improving Accuracy

1. **More data for rare classes** — Blood Disorders (33 rows), Immune (114), Genitourinary (91) are the bottleneck. PetEVAL adds 17,600 rows.
2. **Compare domain-specific base models** — evaluate PetBERT or Bio_ClinicalBERT
   against the current DistilBERT bundle on the locked owner-language holdout;
   promote only measured improvements.
3. **More epochs** — if val F1 is still rising at the end, increase `--epochs`.
4. **Increase `--patience`** — set to 3 or 4 to let the model recover from temporary plateaus.

---

## Project authorship and licensing

This repository is part of **Smart Pet Care App**, a collaborative student diploma project
consisting of a mobile client, a backend API, AI services, designs and project documentation.

See [`PROJECT_NOTICE.md`](PROJECT_NOTICE.md) and [`LICENSE`](LICENSE) in this repository, and
the project agreement and contributor documentation in the canonical project repository
(https://github.com/Gmust/smart-pet-care-app), for authorship and internal project usage terms.
