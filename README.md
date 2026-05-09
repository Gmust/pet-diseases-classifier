# Pet Care AI Microservice

FastAPI microservice providing three AI-powered endpoints for pet health assessment, general pet-care Q&A, and wellness scoring. Deployed on AWS Lambda via AWS SAM.

---

## Architecture

```
Client
  │
  ├─ POST /predict   → Bio_ClinicalBERT classifier (fine-tuned, local)
  │                    + Gemini explanation + home advice
  │
  ├─ POST /ask       → Gemini (general pet-care Q&A, guardrailed)
  │
  └─ POST /wellness  → Rule-based scoring across 6 dimensions
                       + Gemini narrative + recommendations
```

**Design principles:**
- The **classifier** is the sole decision-maker for condition prediction — Gemini cannot override it.
- **Gemini** generates human-friendly text only (explanations, advice, narratives).
- Every response includes a medical/wellness disclaimer.
- Missing data never blocks a response — dimensions are scaled proportionally.

---

## Project Structure

```
app/
  main.py                        # FastAPI app, lifespan, all route handlers
  schemas.py                     # All Pydantic request/response models + enums
  lambda_handler.py              # AWS Lambda entry point (Mangum wrapper)
  ml/
    train.py                     # Fine-tune Bio_ClinicalBERT / PetBERT
    predictor.py                 # Load model + run inference
    condition_metadata.py        # Condition → urgency / specialist / category / advice
    fetch_and_merge.py           # Fetch external HF datasets + merge
    generate_synthetic.py        # Generate synthetic training data via Gemini
  services/
    gemini_service.py            # Explanation + home advice generation
    ask_service.py               # General Q&A with species + topic guardrails
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
LOW_CONFIDENCE_THRESHOLD=0.65              # below this, appends low-confidence warning
```

### Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs

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

### `POST /ask`

General pet-care Q&A: breeds, diet, grooming, training, behaviour, housing.

**Supported species:** `dog` | `cat` | `rabbit` | `hamster` | `guinea_pig` | `bird` | `fish` | `turtle`

Medical/symptom questions are automatically redirected to `/predict`. Unsupported species receive a "not covered" response without consuming Gemini quota.

**Request:**
```json
{
  "question": "How often should I brush a Persian cat?",
  "petType": "cat"
}
```

`petType` is optional — if omitted, Gemini infers the species from the question.

**Response:**
```json
{
  "answer": "Persian cats have long, dense coats that mat easily. Daily brushing with a wide-tooth comb is recommended...",
  "relatedTopics": ["grooming", "persian", "long-hair breeds"],
  "disclaimer": "General pet care information — not a substitute for professional veterinary advice."
}
```

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
| Preventive care | 10 | `PetEvents` — vet visit + vaccinations |
| Baseline | 10 | Pet age + weight tracking |

Missing dimensions are scaled out — partial data is always accepted.

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
    { "name": "Carprofen", "frequency": "daily" }
  ],
  "preventiveCare": {
    "recentVetVisit": true,
    "vaccinationsUpToDate": true
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
  "trend": "STABLE",
  "breakdown": {
    "activity":      { "score": 20.0, "maxScore": 20.0 },
    "sleep":         { "score": 15.0, "maxScore": 15.0 },
    "diet":          { "score": 20.0, "maxScore": 20.0 },
    "symptoms":      { "score": 15.1, "maxScore": 25.0 },
    "preventiveCare":{ "score": 10.0, "maxScore": 10.0 },
    "baseline":      { "score": 10.0, "maxScore": 10.0 }
  },
  "conditionCap": 85,
  "classifierCondition": "Musculoskeletal Conditions",
  "narrative": "Your Labrador is doing well overall with excellent activity and sleep scores...",
  "recommendations": [
    "Continue current exercise routine with gentle low-impact activity.",
    "Monitor for increased stiffness after rest — note timing and duration.",
    "Consider adding an omega-3 supplement for joint support."
  ],
  "disclaimer": "This wellness indicator is based on tracked activity, feeding, and care data. It is not a clinical assessment and does not replace a veterinary examination."
}
```

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
python -m app.ml.generate_synthetic --samples-per-class 100
```

Output: `data/synthetic_data.parquet`

### Step 2 — Merge all data sources

```bash
python -m app.ml.fetch_and_merge
```

Combines: local base data + VetPetCare (free HF dataset) + synthetic data.
Output: `data/merged_augmented.parquet`

### Step 3 — Train

```bash
python -m app.ml.train \
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
HF_TOKEN=hf_xxxx python -m app.ml.train \
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
2. **Use PetBERT** — switching `--base-model SAVSNET/PetBERT` from Bio_ClinicalBERT typically gives +3–5% F1 on vet text.
3. **More epochs** — if val F1 is still rising at the end, increase `--epochs`.
4. **Increase `--patience`** — set to 3 or 4 to let the model recover from temporary plateaus.
