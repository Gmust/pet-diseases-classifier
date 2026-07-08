"""
Generate synthetic training examples for underrepresented condition classes
using the Gemini API (the same key already configured in your .env).

Why synthetic data?
-------------------
- Blood Disorders:        33 samples  → model struggles
- Immune System Disorders: 114 samples → low F1 (0.27)
- Genitourinary Conditions: 91 samples → low F1 (0.43)
- Cardiovascular Conditions: 81 samples → moderate F1 (0.61)
- Respiratory Conditions:   117 samples → moderate F1 (0.67)
- Reproductive Conditions:   39 samples → low F1 (0.40)

This script generates realistic owner-written symptom descriptions + clinical notes
for each target class and saves them as a parquet file ready for merge + retrain.

Usage
-----
# Generate with defaults (100 examples per weak class):
python -m app.ml.generate_synthetic

# Custom targets:
python -m app.ml.generate_synthetic \
    --output data/synthetic_data.parquet \
    --samples-per-class 150 \
    --classes "Blood Disorders" "Immune System Disorders" "Genitourinary Conditions"

# Dry run (print prompts only, no API calls):
python -m app.ml.generate_synthetic --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from app.ml.synthetic_quality import apply_quality_gates, quality_summary, tag_provenance

load_dotenv()


# ---------------------------------------------------------------------------
# Target classes and their generation prompts
# ---------------------------------------------------------------------------

# Each entry: condition_label -> (description_for_prompt, example_diseases)
CONDITION_SPECS: dict[str, dict] = {
    "Blood Disorders": {
        "description": "blood and haematological conditions",
        "examples": "anaemia, thrombocytopenia, haemophilia, blood clotting disorders, polycythaemia, leukopenia",
        "species_focus": "dogs and cats",
    },
    "Immune System Disorders": {
        "description": "immune-mediated and autoimmune conditions",
        "examples": "immune-mediated haemolytic anaemia (IMHA), lupus, immune-mediated thrombocytopenia, pemphigus, rheumatoid arthritis, vasculitis",
        "species_focus": "dogs and cats",
    },
    "Genitourinary Conditions": {
        "description": "genitourinary and kidney conditions",
        "examples": "urinary tract infection, bladder stones, chronic kidney disease, cystitis, urinary incontinence, renal failure, polycystic kidney disease",
        "species_focus": "dogs and cats",
    },
    "Cardiovascular Conditions": {
        "description": "heart and cardiovascular conditions",
        "examples": "dilated cardiomyopathy, mitral valve disease, congestive heart failure, heartworm disease, arrhythmia, pericardial effusion",
        "species_focus": "dogs and cats",
    },
    "Respiratory Conditions": {
        "description": "respiratory and lung conditions",
        "examples": "pneumonia, asthma, bronchitis, tracheal collapse, laryngeal paralysis, pleural effusion, feline upper respiratory infection",
        "species_focus": "dogs and cats",
    },
    "Reproductive Conditions": {
        "description": "reproductive and perinatal conditions",
        "examples": "pyometra, cryptorchidism, dystocia, false pregnancy, mammary tumours, prostatic hyperplasia, eclampsia",
        "species_focus": "dogs and cats",
    },
    "Eye Conditions": {
        "description": "eye and vision conditions",
        "examples": "cataracts, glaucoma, conjunctivitis, corneal ulcer, uveitis, retinal detachment, cherry eye, entropion",
        "species_focus": "dogs and cats",
    },
    "Injury and Poisoning": {
        "description": "traumatic injuries and toxic ingestion",
        "examples": "hit by car, lacerations, fractures, toxic plant ingestion, chemical burns, insect bites, foreign body ingestion",
        "species_focus": "dogs and cats",
    },
    "Metabolic and Endocrine Disorders": {
        "description": "metabolic, nutritional and hormonal conditions",
        "examples": "diabetes, hypothyroidism, hyperthyroidism, Cushing's disease, Addison's disease, obesity, pancreatitis",
        "species_focus": "dogs and cats",
    },
    "Neoplasms": {
        "description": "tumours, lumps and cancers",
        "examples": "lipoma, mast cell tumour, lymphoma, mammary tumour, melanoma, osteosarcoma, a new growing lump",
        "species_focus": "dogs and cats",
    },
    "Neurological and Behavioural Disorders": {
        "description": "neurological and behavioural conditions",
        "examples": "seizures, vestibular disease, cognitive dysfunction, anxiety, compulsive behaviour, head tilt, disorientation, tremors",
        "species_focus": "dogs and cats",
    },
    # These five are well-covered for dogs/cats but need specs so per-species
    # generation (--species) can extend them to rabbits, birds, etc.
    "Digestive Issues": {
        "description": "digestive and gastrointestinal conditions",
        "examples": "vomiting, diarrhoea, gastrointestinal upset, gut stasis (rabbits/guinea pigs), bloat, constipation, loss of appetite",
        "species_focus": "dogs and cats",
    },
    "Skin Conditions": {
        "description": "skin, coat and fur conditions",
        "examples": "itching, hair loss, mites, rash, sores, overgrooming, flaky skin, scabs, fur loss",
        "species_focus": "dogs and cats",
    },
    "Ear Conditions": {
        "description": "ear conditions",
        "examples": "ear infection, ear mites, head shaking, ear discharge, head tilt, scratching at ears",
        "species_focus": "dogs and cats",
    },
    "Musculoskeletal Conditions": {
        "description": "bone, joint and mobility conditions",
        "examples": "limping, stiffness, arthritis, splayed leg, overgrown nails affecting gait, reluctance to move, swollen joints",
        "species_focus": "dogs and cats",
    },
    "Infectious and Parasitic Diseases": {
        "description": "infectious and parasitic diseases",
        "examples": "fleas, mites, worms, coccidia, bacterial infection, viral infection, fever, lethargy from infection",
        "species_focus": "dogs and cats",
    },
}

# Classes that already have good data (>200 training samples, F1 > 0.80)
# can be skipped unless specifically requested
_WELL_REPRESENTED = {
    "Digestive Issues",
    "Musculoskeletal Conditions",
    "Skin Conditions",
    "Ear Conditions",
    "Neoplasms",
    "Infectious and Parasitic Diseases",
    "Neurological and Behavioural Disorders",
    "Metabolic and Endocrine Disorders",
}

_WEAK_CLASSES = [c for c in CONDITION_SPECS if c not in _WELL_REPRESENTED]

# Classes with ZERO pet-owner-voice training data (only clinical/external/synthetic
# coverage). These are where the model fails on real app input, because users type
# like owners, not clinicians. Generating OWNER-voice text here directly fills the
# serving-distribution gap surfaced by prepare_dataset.py.
_OWNER_DATA_CLASSES = {
    "Musculoskeletal Conditions",
    "Skin Conditions",
    "Digestive Issues",
    "Ear Conditions",
    "Infectious and Parasitic Diseases",
}
_OWNER_GAP_CLASSES = [c for c in CONDITION_SPECS if c not in _OWNER_DATA_CLASSES]


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------


def _get_gemini_client():
    try:
        from google import genai
        from google.genai import types  # noqa: F401
    except ImportError as exc:
        raise ImportError("google-genai not installed. Run: pip install google-genai") from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Add it to your .env file.")

    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# style → (system prompt, record_type, example line for the few-shot format)
SYSTEM_PROMPT = """You are a veterinary data annotation expert.
Generate realistic training examples for a pet condition classifier.
Each example must be a natural symptom description that a pet owner OR a veterinary clinician might write.
Vary between:
  - Owner perspective: informal, worried, describing what they observe at home
  - Clinical perspective: technical, using medical terminology, describing exam findings

CRITICAL RULES:
- Only describe symptoms and observations — never state the diagnosis directly in the text
- Mix species (mostly dogs and cats, occasionally rabbits or birds)
- Vary length: some short (1-2 sentences), some detailed (3-5 sentences)
- Use different writing styles — avoid repetitive phrasing
- Output ONLY a JSON array of strings, nothing else
"""

# Owner voice — this is the register the model is actually SERVED. Use this to fill
# the owner-language gap for classes that only have clinical/synthetic coverage.
OWNER_SYSTEM_PROMPT = """You are simulating how everyday PET OWNERS describe their pet's
symptoms when typing into a phone app — NOT how a vet writes.

Write in the owner's voice:
- First person, informal, worried pet-parent tone ("my dog", "she", "he", sometimes a name).
- Plain everyday words — NO medical jargon, NO abbreviations, NO vital signs or exam findings.
- Describe what they SEE or NOTICE at home (behaviour, appetite, energy, what looks wrong).
- Natural and a bit messy: varied length, casual phrasing, occasional run-on sentences.
- Some very short ("my cat keeps sneezing and won't eat"), some a few sentences.

CRITICAL RULES:
- Only describe symptoms/observations — NEVER name or guess the diagnosis in the text.
- Mostly dogs and cats; occasionally rabbits or birds.
- Avoid repetitive openings — do not start every example the same way.
- Output ONLY a JSON array of strings, nothing else.
"""

# style -> (system_prompt, record_type, few-shot example line)
_STYLES: dict[str, tuple[str, str, str]] = {
    "mixed": (
        SYSTEM_PROMPT,
        "Synthetic (Gemini)",
        '["My dog has been lethargic for 3 days and gums look pale.", "The patient presented with ...", ...]',
    ),
    "owner": (
        OWNER_SYSTEM_PROMPT,
        "Synthetic Owner (Gemini)",
        '["my dog seems really tired lately and his gums look kinda pale", "she hasnt been herself, keeps hiding and wont eat", ...]',
    ),
    "clinical": (
        SYSTEM_PROMPT,
        "Synthetic Clinical (Gemini)",
        '["Patient presents with pallor, lethargy and tachycardia on exam.", "O reports inappetence x3d; MM pale ...", ...]',
    ),
}


def _build_prompt(
    condition: str, spec: dict, n: int, style: str = "mixed", species: str | None = None
) -> str:
    _, _, example_line = _STYLES[style]
    focus_species = species or spec["species_focus"]
    voice = (
        "symptom descriptions, each written the way a worried pet OWNER would describe it at home,"
        if style == "owner"
        else "symptom description examples,"
    )
    species_rule = (
        f"\nEVERY example MUST be about a {species}. Use species-appropriate anatomy, behaviour and "
        f"husbandry details (e.g. cage/tank/hutch, not 'walks' for caged pets). Mention the species naturally."
        if species
        else ""
    )
    return f"""Generate {n} realistic {voice} for a pet with {spec['description']}.

Condition being described (DO NOT mention this explicitly in the text): {condition}
Typical diseases in this category: {spec['examples']}
Focus species: {focus_species}{species_rule}

Output a JSON array of exactly {n} strings. Example format:
{example_line}

Generate the {n} examples now:"""


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------


class QuotaExhaustedError(Exception):
    """Raised when the Gemini daily quota is exhausted — no point retrying."""

    pass


def _generate_batch(
    client: Any,
    model_name: str,
    condition: str,
    spec: dict[str, Any],
    n: int,
    style: str = "mixed",
    species: str | None = None,
) -> list[str]:
    """Generate a batch of synthetic examples using Gemini."""
    prompt = _build_prompt(condition, spec, n, style, species)
    system_prompt = _STYLES[style][0]
    raw = ""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.9,
                "max_output_tokens": 8192,
            },
        )
        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        examples: list = json.loads(raw)
        if not isinstance(examples, list):
            raise ValueError(f"Expected list, got {type(examples)}")

        return [str(e).strip() for e in examples if isinstance(e, str) and len(str(e).strip()) > 10]

    except (json.JSONDecodeError, ValueError) as exc:
        print(f"    [warn] JSON parse error for {condition}: {exc}")
        if raw:
            print(f"    Raw response snippet: {raw[:200]}")
        return []
    except Exception as exc:
        err_str = str(exc)
        # Detect daily quota exhaustion — retrying won't help until tomorrow
        if "429" in err_str and (
            "per_day" in err_str.lower()
            or "per_project_per_model" in err_str.lower()
            or "daily" in err_str.lower()
            or "free_tier" in err_str.lower()
        ):
            raise QuotaExhaustedError(
                "Daily Gemini quota exhausted. Partial results will be saved.\n"
                "Run again tomorrow, or upgrade to a paid plan for higher limits.\n"
                f"Original error: {exc}"
            ) from exc
        print(f"    [error] API error for {condition}: {exc}")
        return []


def _save_partial(rows: list[dict], output_path: str) -> None:
    """Save whatever was collected so far — called on quota exhaustion."""
    if not rows:
        print("Nothing generated yet — no file saved.")
        return
    df = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Append to existing file if present (resume support)
    if output.exists():
        existing = pd.read_parquet(output)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["text", "condition"]
        )
    df.to_parquet(output, index=False)
    print(f"Saved {len(df)} rows to {output} (including any previously saved data)")


def generate_synthetic_data(
    output_path: str = "data/synthetic_data.parquet",
    samples_per_class: int = 100,
    classes: list[str] | None = None,
    # gemini-2.5-flash: best quality, use with paid credits ($300 budget)
    gemini_model: str = "gemini-2.5-flash",
    # Large batches = fewer API calls = faster generation
    batch_size: int = 50,
    dry_run: bool = False,
    style: str = "mixed",
    owner_gap: bool = False,
    species: str | None = None,
) -> None:
    if style not in _STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from: {', '.join(_STYLES)}")
    record_type = _STYLES[style][1]
    if species:
        # Tag the source so coverage per species is traceable in the merged dataset.
        record_type = f"{record_type} ({species})"

    if owner_gap and classes is None:
        # The 11 classes with no owner-voice training data.
        target_classes = list(_OWNER_GAP_CLASSES)
    else:
        target_classes = classes or _WEAK_CLASSES

    # Validate requested classes
    unknown = [c for c in target_classes if c not in CONDITION_SPECS]
    if unknown:
        available = ", ".join(sorted(CONDITION_SPECS.keys()))
        raise ValueError(f"Unknown class(es): {unknown}\nAvailable: {available}")

    # Estimate API calls needed
    calls_needed = sum(max(1, (samples_per_class // batch_size) + 1) for _ in target_classes)
    print(f"Generating synthetic data for {len(target_classes)} classes  |  style: {style}")
    print(f"Target: {samples_per_class} examples per class  |  batch size: {batch_size}")
    print(f"Model: {gemini_model}  |  estimated API calls: ~{calls_needed}")
    print(f"Record type: {record_type}")
    print(f"Classes: {target_classes}\n")

    if dry_run:
        print("=== DRY RUN — showing prompts only ===\n")
        for condition in target_classes:
            spec = CONDITION_SPECS[condition]
            print(f"--- {condition} ---")
            print(_build_prompt(condition, spec, batch_size, style, species))
            print()
        return

    client = _get_gemini_client()
    all_rows: list[dict] = []

    for condition in target_classes:
        spec = CONDITION_SPECS[condition]
        print(f"[{condition}] Generating {samples_per_class} examples...")
        collected: list[str] = []
        attempts = 0
        max_attempts = (samples_per_class // batch_size + 1) * 2  # 2× retries budget

        try:
            while len(collected) < samples_per_class and attempts < max_attempts:
                needed = samples_per_class - len(collected)
                this_batch = min(batch_size, needed + 10)  # slight overshoot
                batch = _generate_batch(
                    client, gemini_model, condition, spec, this_batch, style, species
                )
                collected.extend(batch)
                attempts += 1

                if batch:
                    print(
                        f"  batch {attempts}: got {len(batch)} → total {len(collected)}/{samples_per_class}"
                    )
                else:
                    print(f"  batch {attempts}: empty response, retrying in 3s...")
                    time.sleep(3)

        except QuotaExhaustedError as exc:
            print(f"\n⚠️  {exc}")
            # Save whatever we have so far (including current partial class)
            for text in collected:
                all_rows.append({"text": text, "condition": condition, "record_type": record_type})
            _save_partial(all_rows, output_path)
            print("\nRun the script again tomorrow to continue (existing rows will be preserved).")
            return

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for ex in collected:
            if ex not in seen:
                seen.add(ex)
                unique.append(ex)

        final = unique[:samples_per_class]
        print(f"  → {len(final)} unique examples saved for '{condition}'\n")

        for text in final:
            all_rows.append(
                {
                    "text": text,
                    "condition": condition,
                    "record_type": record_type,
                }
            )

        time.sleep(0.5)  # gentle pause between classes

    if not all_rows:
        print("No data generated.")
        return

    df = pd.DataFrame(all_rows)
    df = tag_provenance(df, model_name=gemini_model)
    df = apply_quality_gates(df)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)

    summary = quality_summary(df)
    print(f"{'='*50}")
    print(f"Generated {len(df)} total synthetic examples")
    print(
        f"Quality gates: {summary['leaks_diagnosis']} leak the diagnosis, "
        f"{summary['near_duplicate']} are near-duplicates, "
        f"{summary['needs_review']} flagged for manual review"
    )
    print("\nBreakdown:")
    print(df["condition"].value_counts().to_string())
    print(f"\nSaved to: {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic pet-condition training data via Gemini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", default="data/synthetic_data.parquet", help="Output parquet path."
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=100, help="Target number of examples per class."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Specific classes to generate. Default: all weak classes.",
    )
    parser.add_argument(
        "--style",
        choices=["mixed", "owner", "clinical"],
        default="mixed",
        help="Writing register. 'owner' = how users actually type (fills the "
        "owner-language gap); 'clinical' = vet notes; 'mixed' = both.",
    )
    parser.add_argument(
        "--owner-gap",
        action="store_true",
        help="Target the 11 classes that have NO owner-voice data. Pair with "
        "--style owner to fill the serving-distribution gap.",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Generate examples for a specific species (e.g. rabbit, hamster, bird). "
        "Default: dogs and cats. Use to extend coverage to other pets.",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Gemini model name. Use gemini-2.5-flash with paid credits "
        "for best quality, or gemini-1.5-flash for free tier (1,500 req/day).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Examples to request per API call. Larger = fewer quota-consuming calls.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompts only, no API calls.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_synthetic_data(
        output_path=args.output,
        samples_per_class=args.samples_per_class,
        classes=args.classes,
        gemini_model=args.gemini_model,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        style=args.style,
        owner_gap=args.owner_gap,
        species=args.species,
    )
