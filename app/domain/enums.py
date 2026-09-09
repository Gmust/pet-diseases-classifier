"""
Vocabulary shared across every domain: urgency, specialist routing, disease
category, and species.

These live in `app.domain` rather than in a per-domain `schemas.py` because
triage, wellness, and the static condition map all speak them. Anything used by
exactly one domain belongs in that domain's own schemas module instead.
"""

from enum import StrEnum


class UrgencyLevel(StrEnum):
    """How urgently the owner should seek veterinary attention."""

    MONITOR = "MONITOR"  # Watch at home; vet visit only if it worsens
    CONSULT_SOON = "CONSULT_SOON"  # Book an appointment within 1-3 days
    URGENT = "URGENT"  # Same-day or next-morning vet visit recommended
    EMERGENCY = "EMERGENCY"  # Go to an emergency clinic immediately


class SpecialistType(StrEnum):
    """Type of veterinary specialist best suited for the predicted condition."""

    GENERAL_VET = "general_vet"
    DERMATOLOGIST = "dermatologist"
    NEUROLOGIST = "neurologist"
    CARDIOLOGIST = "cardiologist"
    ONCOLOGIST = "oncologist"
    OPHTHALMOLOGIST = "ophthalmologist"
    INTERNIST = "internist"
    SURGEON = "surgeon"
    EMERGENCY_VET = "emergency_vet"


class DiseaseCategory(StrEnum):
    """Broad biomedical category the predicted condition belongs to."""

    INFECTIOUS = "INFECTIOUS"
    METABOLIC = "METABOLIC"
    STRUCTURAL = "STRUCTURAL"
    NEOPLASTIC = "NEOPLASTIC"
    IMMUNE = "IMMUNE"
    NEUROLOGICAL = "NEUROLOGICAL"
    CARDIOVASCULAR = "CARDIOVASCULAR"
    DERMATOLOGICAL = "DERMATOLOGICAL"
    GASTROINTESTINAL = "GASTROINTESTINAL"
    RESPIRATORY = "RESPIRATORY"
    OPHTHALMIC = "OPHTHALMIC"
    UROGENITAL = "UROGENITAL"
    TRAUMA = "TRAUMA"
    HEMATOLOGICAL = "HEMATOLOGICAL"
    REPRODUCTIVE = "REPRODUCTIVE"
    EAR = "EAR"


class PetType(StrEnum):
    """Supported pet species (optional hint on /chat, used by /wellness)."""

    DOG = "dog"
    CAT = "cat"
    RABBIT = "rabbit"
    HAMSTER = "hamster"
    GUINEA_PIG = "guinea_pig"
    BIRD = "bird"
    FISH = "fish"
    TURTLE = "turtle"
    OTHER = "other"


SUPPORTED_PET_TYPES: frozenset[PetType] = frozenset(
    {
        PetType.DOG,
        PetType.CAT,
        PetType.RABBIT,
        PetType.HAMSTER,
        PetType.GUINEA_PIG,
        PetType.BIRD,
        PetType.FISH,
        PetType.TURTLE,
    }
)
