"""
Static metadata derived from each predicted condition label.

Each entry maps one of the 16 trained label strings (as stored in
model.config.id2label) to the urgency level, recommended specialist type,
broad disease category, and default home-care advice.

The home_advice list is used as a fallback when Gemini is unavailable and
as a safety net when Gemini returns an empty advice list.

Update this file if the training label set changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas import DiseaseCategory, SpecialistType, UrgencyLevel


@dataclass(frozen=True)
class ConditionMetadata:
    urgency: UrgencyLevel
    specialist: SpecialistType
    disease_category: DiseaseCategory
    home_advice: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mapping — keys must match id2label values in the fine-tuned model config
# ---------------------------------------------------------------------------

CONDITION_METADATA: dict[str, ConditionMetadata] = {
    "Digestive Issues": ConditionMetadata(
        urgency=UrgencyLevel.CONSULT_SOON,
        specialist=SpecialistType.GENERAL_VET,
        disease_category=DiseaseCategory.GASTROINTESTINAL,
        home_advice=[
            "Withhold food for 12-24 hours (water only) to rest the stomach.",
            "After fasting, offer small portions of bland food: boiled chicken and plain rice.",
            "Feed 2-3 small meals per day instead of one large meal until symptoms resolve.",
            "Ensure fresh water is always available — dehydration worsens GI issues.",
            "Watch for blood in vomit or stool, severe lethargy, or bloating; seek emergency care if present.",
        ],
    ),
    "Infectious and Parasitic Diseases": ConditionMetadata(
        urgency=UrgencyLevel.URGENT,
        specialist=SpecialistType.GENERAL_VET,
        disease_category=DiseaseCategory.INFECTIOUS,
        home_advice=[
            "Isolate your pet from other animals to prevent spread.",
            "Keep your pet warm, dry, and resting in a quiet area.",
            "Ensure access to fresh water at all times.",
            "Do not give over-the-counter human medications — many are toxic to pets.",
            "Monitor temperature: normal range is 38-39.2°C (100.4-102.5°F) for dogs and cats.",
        ],
    ),
    "Musculoskeletal Conditions": ConditionMetadata(
        urgency=UrgencyLevel.CONSULT_SOON,
        specialist=SpecialistType.SURGEON,
        disease_category=DiseaseCategory.STRUCTURAL,
        home_advice=[
            "Restrict exercise — no running, jumping, or stairs until assessed.",
            "Provide a soft, supportive bed at ground level.",
            "Apply a cold pack (wrapped in cloth) to swollen joints for 10 minutes, 2-3 times daily.",
            "Maintain a healthy weight — excess body weight stresses joints significantly.",
            "Do not give human pain relievers (ibuprofen, paracetamol) — they are toxic to pets.",
        ],
    ),
    "Skin Conditions": ConditionMetadata(
        urgency=UrgencyLevel.MONITOR,
        specialist=SpecialistType.DERMATOLOGIST,
        disease_category=DiseaseCategory.DERMATOLOGICAL,
        home_advice=[
            "Use an e-collar (cone) to prevent scratching and licking the affected area.",
            "Bathe with a gentle, pet-safe hypoallergenic shampoo once a week.",
            "Avoid known allergens: new foods, household cleaners, synthetic fabrics, pollen.",
            "Keep the skin clean and dry — moisture worsens most dermatological conditions.",
            "Do not apply human creams, hydrocortisone, or essential oils without vet approval.",
        ],
    ),
    "Ear Conditions": ConditionMetadata(
        urgency=UrgencyLevel.MONITOR,
        specialist=SpecialistType.GENERAL_VET,
        disease_category=DiseaseCategory.EAR,
        home_advice=[
            "Gently wipe visible debris from the outer ear canal with a cotton ball — no Q-tips.",
            "Keep ears dry after baths or swimming; use a dry cotton ball to absorb moisture.",
            "Use an e-collar to prevent scratching, which can worsen infection.",
            "Do not insert anything deeper than the visible canal.",
            "Check both ears daily for discharge, odour, or redness.",
        ],
    ),
    "Neoplasms": ConditionMetadata(
        urgency=UrgencyLevel.CONSULT_SOON,
        specialist=SpecialistType.ONCOLOGIST,
        disease_category=DiseaseCategory.NEOPLASTIC,
        home_advice=[
            "Note the size, location, and texture of any lumps — photograph them weekly to track changes.",
            "Watch for rapid growth, ulceration, bleeding, or changes in eating behaviour.",
            "Keep your pet comfortable with a calm, stress-free environment.",
            "Maintain a nutritious diet; consult your vet about cancer-supportive nutrition.",
            "Do not attempt to remove or puncture any growth at home.",
        ],
    ),
    "Neurological and Behavioural Disorders": ConditionMetadata(
        urgency=UrgencyLevel.URGENT,
        specialist=SpecialistType.NEUROLOGIST,
        disease_category=DiseaseCategory.NEUROLOGICAL,
        home_advice=[
            "Keep the environment calm and free of hazards — padded surfaces around furniture corners.",
            "Do not restrain your pet during a seizure; move objects away and time the episode.",
            "Log episodes: date, time, duration, and behaviour before and after.",
            "Avoid triggering situations: loud noises, bright flashing lights, sudden movements.",
            "Ensure your pet cannot fall from heights or injure itself if balance is impaired.",
        ],
    ),
    "Metabolic and Endocrine Disorders": ConditionMetadata(
        urgency=UrgencyLevel.CONSULT_SOON,
        specialist=SpecialistType.INTERNIST,
        disease_category=DiseaseCategory.METABOLIC,
        home_advice=[
            "Stick to a consistent feeding schedule — 2 meals per day at the same times.",
            "Feed a low-fat, high-fibre diet unless your vet advises otherwise.",
            "Ensure constant access to fresh water, especially for diabetic or kidney-related conditions.",
            "Monitor weight weekly and log any significant changes.",
            "Avoid treats, table scraps, and high-sugar foods entirely.",
        ],
    ),
    "Eye Conditions": ConditionMetadata(
        urgency=UrgencyLevel.CONSULT_SOON,
        specialist=SpecialistType.OPHTHALMOLOGIST,
        disease_category=DiseaseCategory.OPHTHALMIC,
        home_advice=[
            "Gently wipe discharge from the eye with a clean, damp cotton pad — use a fresh pad for each eye.",
            "Use an e-collar to prevent pawing at the eye.",
            "Keep your pet out of bright sunlight and dusty environments.",
            "Do not use human eye drops unless specifically prescribed by a vet.",
            "Watch for sudden cloudiness, squinting, or vision loss — these require immediate attention.",
        ],
    ),
    "Respiratory Conditions": ConditionMetadata(
        urgency=UrgencyLevel.URGENT,
        specialist=SpecialistType.GENERAL_VET,
        disease_category=DiseaseCategory.RESPIRATORY,
        home_advice=[
            "Keep your pet resting in a calm, well-ventilated room with clean air.",
            "Run a humidifier nearby to ease breathing — avoid dry air.",
            "Avoid exposure to smoke, strong perfumes, dust, and cold air.",
            "Watch breathing rate at rest: normal is 15-30 breaths per minute for dogs and cats.",
            "If lips or gums turn blue/grey, this is an emergency — go to a vet immediately.",
        ],
    ),
    "Cardiovascular Conditions": ConditionMetadata(
        urgency=UrgencyLevel.URGENT,
        specialist=SpecialistType.CARDIOLOGIST,
        disease_category=DiseaseCategory.CARDIOVASCULAR,
        home_advice=[
            "Restrict strenuous exercise; short, calm walks on a lead only.",
            "Feed a low-sodium diet — avoid salty treats and processed foods.",
            "Keep your pet at a healthy weight to reduce strain on the heart.",
            "Monitor resting breathing rate daily (write it down); an increase may signal fluid build-up.",
            "Avoid stressful situations — stress significantly worsens cardiac conditions.",
        ],
    ),
    "Immune System Disorders": ConditionMetadata(
        urgency=UrgencyLevel.CONSULT_SOON,
        specialist=SpecialistType.INTERNIST,
        disease_category=DiseaseCategory.IMMUNE,
        home_advice=[
            "Minimise stress — it suppresses immune function and worsens flare-ups.",
            "Feed a balanced, high-quality diet to support immune health.",
            "Keep your pet away from sick animals and crowded environments.",
            "Monitor for new symptoms: unusual lumps, hair loss, lethargy, pale gums.",
            "Do not stop or change any prescribed medication without consulting a vet.",
        ],
    ),
    "Genitourinary Conditions": ConditionMetadata(
        urgency=UrgencyLevel.CONSULT_SOON,
        specialist=SpecialistType.INTERNIST,
        disease_category=DiseaseCategory.UROGENITAL,
        home_advice=[
            "Increase fresh water intake — add a water fountain or wet food to encourage drinking.",
            "Take your pet outside to urinate more frequently to flush the urinary tract.",
            "Feed a low-mineral, urinary-support diet if stones or crystals are suspected.",
            "Monitor urination: frequency, colour, and whether straining occurs.",
            "Seek emergency care if your pet cannot urinate at all — this is life-threatening.",
        ],
    ),
    "Injury and Poisoning": ConditionMetadata(
        urgency=UrgencyLevel.EMERGENCY,
        specialist=SpecialistType.EMERGENCY_VET,
        disease_category=DiseaseCategory.TRAUMA,
        home_advice=[
            "Stay calm and keep your pet still — movement can worsen injuries.",
            "For wounds: apply gentle pressure with a clean cloth to control bleeding.",
            "If poisoning is suspected: note the substance name and contact a vet or poison control immediately.",
            "Do NOT induce vomiting unless a vet specifically instructs you to.",
            "Transport your pet to an emergency clinic as quickly and calmly as possible.",
        ],
    ),
    "Blood Disorders": ConditionMetadata(
        urgency=UrgencyLevel.URGENT,
        specialist=SpecialistType.INTERNIST,
        disease_category=DiseaseCategory.HEMATOLOGICAL,
        home_advice=[
            "Keep your pet resting — physical exertion worsens anaemia and clotting disorders.",
            "Offer small, frequent meals of highly digestible, iron-rich food (lean meats).",
            "Check gums daily: they should be pink and moist — pale or white gums require immediate vet attention.",
            "Prevent injuries — avoid rough play, sharp edges, or situations that could cause bleeding.",
            "Do not give aspirin or any blood-thinning supplements without explicit vet guidance.",
        ],
    ),
    "Reproductive Conditions": ConditionMetadata(
        urgency=UrgencyLevel.CONSULT_SOON,
        specialist=SpecialistType.GENERAL_VET,
        disease_category=DiseaseCategory.REPRODUCTIVE,
        home_advice=[
            "Keep the genital area clean with gentle, warm water — do not use soap or disinfectants.",
            "Prevent licking with an e-collar.",
            "Monitor for abnormal discharge: colour, odour, or volume changes.",
            "Provide a quiet, warm, and stress-free resting area.",
            "If your pet is pregnant, ensure access to fresh water and a nutritious, increased-calorie diet.",
        ],
    ),
}

# Safe fallback for any label not in the table (e.g., after retraining
# with a new class before this file is updated).
_FALLBACK = ConditionMetadata(
    urgency=UrgencyLevel.CONSULT_SOON,
    specialist=SpecialistType.GENERAL_VET,
    disease_category=DiseaseCategory.GASTROINTESTINAL,
    home_advice=[
        "Keep your pet calm and resting in a comfortable, quiet space.",
        "Ensure access to fresh water at all times.",
        "Monitor for changes in appetite, behaviour, or any worsening of symptoms.",
        "Do not give human medications without veterinary guidance.",
    ],
)


def get_condition_metadata(condition: str) -> ConditionMetadata:
    """Return metadata for *condition*, falling back gracefully to safe defaults."""
    return CONDITION_METADATA.get(condition, _FALLBACK)
