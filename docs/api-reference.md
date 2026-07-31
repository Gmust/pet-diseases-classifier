# API Reference — Pet Care AI Microservice

Backend integration reference for all endpoints. For the **stateful chat flow**
(session storage, rolling-summary persistence, C# DTOs), see the companion guide
`dotnet-chat-integration.md` — this document is the general endpoint reference.

- **Base URL (Lambda):** `https://{api-id}.execute-api.{region}.amazonaws.com/Prod/`
- **Base URL (local):** `http://localhost:8000/`
- **Content type:** `application/json` on all `POST` endpoints.
- **Interactive docs:** `/docs` (Swagger UI) when running the service.

---

## Authentication

All `POST` endpoints require an API key **when** the service is deployed with the
`API_KEY` environment variable set. `GET /health` is always open.

```
X-API-Key: <your secret>
```

If `API_KEY` is unset/empty on the server, auth is skipped (local dev). Comparison
uses `hmac.compare_digest` (constant-time).

---

## Common error responses

| Status | Meaning | Typical cause |
|---|---|---|
| `400` | Bad Request | Semantically invalid input (e.g. empty classifier text, `/chat` with no user message). |
| `403` | Forbidden | Missing/incorrect `X-API-Key` (only when auth is enabled). |
| `422` | Unprocessable Entity | Schema validation failed (missing field, wrong type, exceeds length caps). |
| `500` | Internal Server Error | Unexpected inference failure. |

Error bodies include `{ "detail": "<message>", "requestId": "<id>" }`. For
`422`, `detail` is a list of field-level validation errors.

> **All field names are camelCase** in both requests and responses. The service
> also accepts the snake_case internal names, but camelCase is the contract.

---

## `GET /health`

Liveness probe (used by the load balancer). No auth.

**Response `200`**
```json
{ "status": "ok" }
```

## `GET /health/live`

Process liveness probe. Returns `200` with `{ "status": "ok" }` independently
of model readiness.

## `GET /health/ready`

Model readiness probe. Returns `200` with non-sensitive `backend`,
`modelVersion`, and `labelCount` metadata after successful model validation, or
`503` with `reason=model_not_loaded` before services are ready.

---

## `POST /predict`

One-shot classification of a free-text symptom description, with a generated
explanation and home-care advice. Stateless — use this for single assessments;
use `/chat` for conversations.

**Request**

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | Symptom description. 1–4000 chars. |

```json
{ "text": "My dog has been vomiting and won't eat for 2 days" }
```

**Response `200`**

| Field | Type | Notes |
|---|---|---|
| `predictedCondition` | string | One of the 16 condition classes (see enums). |
| `confidence` | number | 0–1. |
| `explanation` | string | Cautious explanation. May be prefixed with a ⚠️ emergency note or suffixed with a low-confidence note (see Behavior). |
| `disclaimer` | string | Always present. |
| `urgency` | string | `MONITOR \| CONSULT_SOON \| URGENT \| EMERGENCY`. |
| `specialist` | string | Recommended specialist (see enums). |
| `diseaseCategory` | string | Broad category (see enums). |
| `homeAdvice` | string[] | Practical owner tips. |

```json
{
  "predictedCondition": "Digestive Issues",
  "confidence": 0.92,
  "explanation": "The described symptoms may be consistent with digestive issues. ...",
  "disclaimer": "This is an AI-assisted pre-assessment and not a veterinary diagnosis.",
  "urgency": "CONSULT_SOON",
  "specialist": "general_vet",
  "diseaseCategory": "GASTROINTESTINAL",
  "homeAdvice": ["Withhold food for 12-24 hours (water only) ...", "..."]
}
```

---

## `POST /chat`

**Unified conversational endpoint.** Send any user message; the service decides
under the hood whether it's a general pet-care question or a health concern, and
returns a `mode` so you branch on the response. (This replaces the old separate
`/ask` endpoint.) **Your backend owns all session state** — it stores the message
history and the rolling `symptomSummary`, and replays them on each call. The
service stores nothing. Full integration detail (persistence, DTOs, sequence) is
in `dotnet-chat-integration.md`.

`mode` values:

| `mode` | When | Response shape |
|---|---|---|
| `general` | General-care question (diet, grooming, behaviour…). | `prediction` is `null`; `relatedTopics` set. |
| `health` | Symptom/health concern. | `prediction` populated; `symptomSummary` updated. |
| `emergency` | A red-flag phrase was detected. | `prediction` with `urgency=EMERGENCY`; emergency `answer` + first-aid `homeAdvice`. |

**Request**

| Field | Type | Required | Notes |
|---|---|---|---|
| `messages` | array | yes | Recent turns, oldest-first. Last entry must be the new user message. Max 50. |
| `messages[].role` | string | yes | `"user"` or `"assistant"`. |
| `messages[].content` | string | yes | 1–4000 chars. |
| `symptomSummary` | string | no | Rolling summary returned by the previous turn. Omit/`null` on turn 1. |
| `sessionId` | string | no | Opaque id, used only for logs/telemetry. |
| `petType` | string | no | Species hint (see enums). |

```json
{
  "sessionId": "f1c2…",
  "petType": "cat",
  "symptomSummary": "The cat is sneezing.",
  "messages": [
    { "role": "user", "content": "my cat is sneezing" },
    { "role": "assistant", "content": "Runny nose or watery eyes?" },
    { "role": "user", "content": "yes, runny nose and watery eyes, a bit of coughing" }
  ]
}
```

**Response `200`**

| Field | Type | Notes |
|---|---|---|
| `mode` | string | `general` \| `health` \| `emergency` — branch on this. |
| `answer` | string | Conversational reply for the user. |
| `symptomSummary` | string | **Persist this and send it back next turn** — it's the conversation's memory. |
| `prediction` | object \| null | **Null when `mode=general`.** Populated for `health`/`emergency`. |
| `relatedTopics` | string[] | Keyword tags when `mode=general`; empty otherwise. |
| `needsClarification` | bool | True when the reply is asking a follow-up (low confidence or assistant gathering detail). |
| `disclaimer` | string | Always present. |
| `prediction.predictedCondition` | string | Current best class. May change between turns as symptoms evolve. |
| `prediction.confidence` | number | 0–1. |
| `prediction.topK` | array | `[{ condition, confidence }]`, highest first. |
| `prediction.urgency` | string | See enums. |
| `prediction.specialist` | string | See enums. |
| `prediction.diseaseCategory` | string | See enums. |
| `prediction.homeAdvice` | string[] | Tips for the current condition. |

```json
{
  "answer": "These symptoms might suggest a respiratory condition. Any difficulty breathing?",
  "symptomSummary": "The cat is sneezing, has a runny nose, watery eyes, and a bit of coughing.",
  "needsClarification": true,
  "disclaimer": "This is an AI-assisted pre-assessment and not a veterinary diagnosis.",
  "prediction": {
    "predictedCondition": "Respiratory Conditions",
    "confidence": 0.93,
    "topK": [
      { "condition": "Respiratory Conditions", "confidence": 0.93 },
      { "condition": "Infectious and Parasitic Diseases", "confidence": 0.01 }
    ],
    "urgency": "URGENT",
    "specialist": "general_vet",
    "diseaseCategory": "RESPIRATORY",
    "homeAdvice": ["Keep your pet resting in a calm, well-ventilated room ...", "..."]
  }
}
```

---

## `POST /wellness`

Rule-based wellness assessment across six dimensions, with a generated narrative
and recommendations when enough data is available. Designed to be called by the
backend from aggregated DB records — no manual user input required. Missing
dimensions are scaled out once the reliability requirements are met. A
species-only request returns `200` with `scoreStatus: "INSUFFICIENT_DATA"` and
null score and band fields; absence of observations is not positive wellness
evidence.

**Request** (all sub-objects optional except `pet`)

| Field | Type | Notes |
|---|---|---|
| `pet.species` | string | Required. e.g. `"dog"`. |
| `pet.breed` / `pet.ageMonths` / `pet.sex` / `pet.weightKg` / `pet.behavioralNotes` | mixed | Optional. |
| `activity.avgStepsPerDay` / `avgActiveMinutesPerDay` / `avgSleepHoursPerDay` / `daysTracked` | number | Optional — from activity tracking. |
| `feeding.avgMealsPerDay` / `avgCaloriesPerDay` / `foodTypes[]` / `consistencyDays` | mixed | Optional — from feeding logs. |
| `activeConditions[]` | array | `{ name, typeLabel? }` — active chronic conditions (cap the max score). |
| `activeMedications[]` | array | `{ name, frequency? }`. |
| `preventiveCare` | object | `{ recentVetVisit, vaccinationsUpToDate }`. |
| `currentSymptoms` | string | Optional free text — passed through the classifier. |
| `previousScore` | int | 0–100. Used to compute `trend`. |

```json
{
  "pet": { "species": "dog", "breed": "Labrador", "ageMonths": 36, "weightKg": 28.5 },
  "activity": { "avgStepsPerDay": 8000, "avgActiveMinutesPerDay": 60, "avgSleepHoursPerDay": 12, "daysTracked": 7 },
  "feeding": { "avgMealsPerDay": 2, "consistencyDays": 7 },
  "activeConditions": [],
  "preventiveCare": { "recentVetVisit": true, "vaccinationsUpToDate": true },
  "previousScore": 82
}
```

**Response `200`**

| Field | Type | Notes |
|---|---|---|
| `wellnessScore` | int? | 0–100, or `null` when `scoreStatus` is `INSUFFICIENT_DATA`. |
| `band` | string? | `EXCELLENT \| GOOD \| FAIR \| CONCERNING \| CRITICAL`, or `null` when no score is available. |
| `bandLabel` | string? | Human-readable band label, or `null` when no score is available. |
| `scoreStatus` | string | `COMPLETE \| PARTIAL \| INSUFFICIENT_DATA`. |
| `dataCoverage` | number | Weighted foundational-data coverage from 0–1. |
| `trend` | string? | `IMPROVING \| STABLE \| DECLINING` (only if `previousScore` provided). |
| `breakdown` | object | Per-dimension `{ score, maxScore }` for `activity, sleep, diet, symptoms, preventiveCare, baseline`. |
| `conditionCap` | int? | Max score allowed given active chronic conditions, if any. |
| `classifierCondition` | string? | Condition detected from `currentSymptoms`, if provided. |
| `narrative` | string | Generated summary. |
| `recommendations` | string[] | Actionable suggestions. |
| `disclaimer` | string | Always present. |

---

## `POST /feeding-summary`

Daily per-pet feeding summary, meant to be called once/day (batched across all
pets in one request) by a scheduler in the backend, to drive a feeding
notification. Fully deterministic (RER/MER calorie-target formula) — no
classifier, no Gemini, no per-pet API cost, so it stays cheap at any batch size.

**Request**

| Field | Type | Notes |
|---|---|---|
| `pets[]` | array | Required, 1–1000 entries. |
| `pets[].petId` | string | Required. |
| `pets[].species` | string | Default `"dog"`. |
| `pets[].breed` | string? | Optional. |
| `pets[].weightKg` | number | Required, `0 < weightKg <= 500`. |
| `pets[].ageMonths` | int? | Optional. Applies a growth-energy multiplier for juveniles (< 12mo) instead of the adult maintenance factor. |
| `pets[].products[]` | array | `{ name, calories }` — logged food items for the day. Max 100. |

```json
{
  "pets": [
    {
      "petId": "pet-123",
      "species": "dog",
      "weightKg": 28.5,
      "ageMonths": 36,
      "products": [{ "name": "Kibble A", "calories": 450 }]
    }
  ]
}
```

**Response `200`**

| Field | Type | Notes |
|---|---|---|
| `results[]` | array | One entry per input pet. |
| `results[].petId` | string | Echoes the input `petId`. |
| `results[].status` | string | `EXTREME_UNDER_TARGET \| UNDER_TARGET \| ON_TARGET \| OVER_TARGET \| EXTREME_OVER_TARGET`. No baked-in text — the frontend renders notification copy per-locale from `status` + the numeric fields below. |
| `results[].targetCalories` | number | Computed RER/MER daily target. |
| `results[].actualCalories` | number | Sum of logged `products[].calories`. |
| `results[].deviationPct` | number | `actualCalories` vs `targetCalories`, as a percentage deviation. |
| `disclaimer` | string | Always present. |

---

## Behavior notes (important for the backend)

- **Red-flag escalation.** If the input text contains emergency phrasing (collapse,
  not breathing, seizure, bloat, blocked urination, toxin ingestion, severe
  bleeding, blue gums, hit-by-car, dystocia), `urgency` is forced to `EMERGENCY`
  regardless of the predicted class, and the `explanation`/`answer` is prefixed
  with a ⚠️ note. Surface this prominently in the UI.
- **Low-confidence note.** On `/predict`, if confidence < `LOW_CONFIDENCE_THRESHOLD`
  (default 0.65), a caution sentence is appended to `explanation`.
- **Abstention / `needsClarification`.** On `/chat`, very low confidence (< 0.40)
  forces a clarifying reply and `needsClarification: true`. Note the assistant may
  also set this when gathering more detail even at high confidence.
- **Prediction can change between chat turns** as symptoms accumulate — render the
  current `prediction` each turn; don't lock it after turn 1.
- **Gemini fallback.** If the Gemini key is unset or the call fails, the service
  still returns a valid response using local templated text (explanation/answer +
  static home advice). Responses are always well-formed.
- **Disclaimer is always present** — display it. This is a triage aid, not a diagnosis.

---

## Enum reference

**Condition classes** (`predictedCondition`, 16): `Blood Disorders`,
`Cardiovascular Conditions`, `Digestive Issues`, `Ear Conditions`, `Eye Conditions`,
`Genitourinary Conditions`, `Immune System Disorders`,
`Infectious and Parasitic Diseases`, `Injury and Poisoning`,
`Metabolic and Endocrine Disorders`, `Musculoskeletal Conditions`, `Neoplasms`,
`Neurological and Behavioural Disorders`, `Reproductive Conditions`,
`Respiratory Conditions`, `Skin Conditions`.

**`urgency`:** `MONITOR`, `CONSULT_SOON`, `URGENT`, `EMERGENCY`.

**`specialist`:** `general_vet`, `dermatologist`, `neurologist`, `cardiologist`,
`oncologist`, `ophthalmologist`, `internist`, `surgeon`, `emergency_vet`.

**`diseaseCategory`:** `INFECTIOUS`, `METABOLIC`, `STRUCTURAL`, `NEOPLASTIC`,
`IMMUNE`, `NEUROLOGICAL`, `CARDIOVASCULAR`, `DERMATOLOGICAL`, `GASTROINTESTINAL`,
`RESPIRATORY`, `OPHTHALMIC`, `UROGENITAL`, `TRAUMA`, `HEMATOLOGICAL`,
`REPRODUCTIVE`, `EAR`.

**`petType`:** `dog`, `cat`, `rabbit`, `hamster`, `guinea_pig`, `bird`, `fish`,
`turtle`, `other`. (Optional hint on `/chat`; used by `/wellness`.)

**`band`:** `EXCELLENT` (90–100), `GOOD` (75–89), `FAIR` (60–74),
`CONCERNING` (40–59), `CRITICAL` (0–39).

**`trend`:** `IMPROVING`, `STABLE`, `DECLINING`.

---

## Server configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | _(empty)_ | Enables `X-API-Key` auth on `POST` endpoints. Empty = auth disabled. |
| `GEMINI_API_KEY` | _(empty)_ | Enables Gemini text generation. Empty = local fallbacks. |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Model used for generation. |
| `LOW_CONFIDENCE_THRESHOLD` | `0.65` | Below this, `/predict` appends a low-confidence note. |
| `USE_STATIC_EXPLANATIONS` | `false` | If true, `/predict` skips Gemini and serves a templated explanation (zero API cost). |
| `MODEL_BACKEND` | `torch` | `torch` or `onnx` (quantized, cheaper). |
| `MODEL_PATH` | `models/transformer_model` | Model directory. |
| `LOG_LEVEL` | `INFO` | Log verbosity (structured JSON logs). |
| `ROOT_PATH` | _(empty)_ | Set to `/Prod` behind the API Gateway stage. |

---

## Quick curl examples

```bash
# health
curl -s localhost:8000/health

# predict (auth disabled)
curl -s localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"text":"my dog is vomiting"}'

# chat (auth enabled)
curl -s localhost:8000/chat \
  -H 'Content-Type: application/json' -H "X-API-Key: $API_KEY" \
  -d '{"messages":[{"role":"user","content":"my cat is sneezing"}],"petType":"cat"}'
```
