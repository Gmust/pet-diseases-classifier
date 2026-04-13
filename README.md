# Pet Care AI Microservice

FastAPI microservice for pet condition pre-assessment using a fine-tuned transformer classifier (DistilBERT / PetBERT) and Gemini-generated explanations.

## Architecture

- The **classifier** (transformer) predicts the `condition` — it is the sole decision-maker.
- **Gemini** only generates a human-friendly explanation; it cannot override the classifier.
- Every response includes a medical disclaimer.

## Project Structure

```text
app/
  main.py                   # FastAPI app
  schemas.py                # Request/response models
  ml/
    train.py                # Fine-tune transformer (DistilBERT / PetBERT)
    predictor.py            # Load model and run inference
    fetch_and_merge.py      # Fetch external HF datasets + merge
    merge_datasets.py       # Merge local CSV/parquet files
  services/
    gemini_service.py       # Gemini explanation generation
data/
  merged_pet_dataset.parquet   # Base training data
  merged_augmented.parquet     # Augmented (after fetch_and_merge)
  label_map.json               # Label consolidation map
models/
  transformer_model/           # Fine-tuned model directory (HuggingFace format)
.env.example
requirements.txt
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment

Copy `.env.example` to `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
MODEL_PATH=models/transformer_model
LOW_CONFIDENCE_THRESHOLD=0.65
```

---

## Full Training Pipeline

### Recommended: Fully Free Pipeline (no HuggingFace account needed)

#### Step 1 — Generate synthetic data for weak classes

Uses your existing `GEMINI_API_KEY` to generate realistic symptom descriptions for rare conditions. Costs only a few cents.

```bash
# Default: 100 examples per weak class (~800 new rows total)
python -m app.ml.generate_synthetic

# More examples for better coverage:
python -m app.ml.generate_synthetic --samples-per-class 150
```

Output: `data/synthetic_data.parquet`

#### Step 2 — Merge all sources

```bash
python -m app.ml.fetch_and_merge
```

Combines: local base data + VetPetCare (free HF dataset) + synthetic data.
Output: `data/merged_augmented.parquet`

#### Step 3 — Train with Bio_ClinicalBERT (free, no account needed)

Pre-trained on clinical notes (MIMIC-III) — much better than DistilBERT for medical text, freely available.

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

---

### Optional Upgrade: PetBERT (gated — needs HuggingFace account)

Pre-trained on 5.1M veterinary EHRs — best possible base model for this task.

1. Create a free account at https://huggingface.co
2. Accept conditions at https://huggingface.co/SAVSNET/PetBERT
3. Get your token at https://huggingface.co/settings/tokens

```bash
HF_TOKEN=hf_xxxx python -m app.ml.train \
  --data-path data/merged_augmented.parquet \
  --label-map data/label_map.json \
  --model-dir models/transformer_model \
  --base-model SAVSNET/PetBERT \
  --epochs 8 \
  --batch-size 16 \
  --lr 1e-5
```

Also add PetEVAL (17,600 real UK vet EHRs) to your training mix:
```bash
HF_TOKEN=hf_xxxx python -m app.ml.fetch_and_merge
```

#### Full config reference

```bash
python -m app.ml.train \
  --data-path data/merged_augmented.parquet \
  --label-map data/label_map.json \
  --model-dir models/transformer_model \
  --base-model distilbert-base-uncased \   # or SAVSNET/PetBERT
  --epochs 8 \
  --batch-size 16 \
  --lr 2e-5 \
  --max-length 256 \
  --min-samples 20 \       # drop classes with fewer than N samples
  --patience 2 \           # early-stopping patience (epochs)
  --warmup-ratio 0.1
```

#### Mac GPU (Apple Silicon M1/M2/M3/M4)

MPS is auto-detected — no flags needed. Verify it's active:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
# Should print: MPS: True
```

Training will log `Using device: mps` on the first line.

---

### Step 3 — Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /predict`

**Request:**
```json
{
  "text": "My dog has been vomiting and not eating for 2 days"
}
```

**Response:**
```json
{
  "predictedCondition": "Digestive Issues",
  "confidence": 0.88,
  "explanation": "The symptoms are most consistent with a digestive condition. Monitor closely and consult a vet if symptoms persist.",
  "disclaimer": "This is an AI-assisted pre-assessment and not a veterinary diagnosis."
}
```

**Low confidence (< `LOW_CONFIDENCE_THRESHOLD`):** appends an extra caution note to `explanation`.

### cURL example

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "My cat has been scratching her ears and shaking her head"}' \
  | python3 -m json.tool
```

---

## Label Classes (15 consolidated)

| Class | Description |
|---|---|
| Digestive Issues | Vomiting, diarrhea, bloating, GI obstruction |
| Infectious and Parasitic Diseases | Parasites, parvovirus, bacterial infections |
| Musculoskeletal Conditions | Hip dysplasia, arthritis, mobility problems |
| Skin Conditions | Itching, rashes, dermatitis, alopecia |
| Ear Conditions | Ear infections, discharge, scratching |
| Neoplasms | Tumours, cancers, growths |
| Neurological and Behavioural Disorders | Epilepsy, cognitive dysfunction, behaviour |
| Metabolic and Endocrine Disorders | Diabetes, hyperthyroidism |
| Eye Conditions | Vision problems, eye discharge |
| Respiratory Conditions | Coughing, breathing difficulties, pneumonia |
| Cardiovascular Conditions | Heart disease, heartworm |
| Immune System Disorders | Allergies, immune-mediated disease |
| Genitourinary Conditions | UTI, kidney disease, bladder stones |
| Injury and Poisoning | Trauma, wounds, toxic ingestion |
| Blood Disorders | Anaemia, clotting disorders |
| Reproductive Conditions | Pregnancy, perinatal, reproductive surgery |

---

## Improving Accuracy Further

1. **More data for rare classes** — Blood Disorders (33 rows), Immune (114), Genitourinary (91) are the bottleneck. PetEVAL adds 17,600 rows and covers all of these.
2. **Use PetBERT** — switching `--base-model SAVSNET/PetBERT` from DistilBERT typically gives +3–5% F1 on vet text.
3. **More epochs** — if val F1 is still rising at the end, increase `--epochs`.
4. **Increase `--patience`** — set to 3 or 4 to let the model recover from temporary plateaus.
