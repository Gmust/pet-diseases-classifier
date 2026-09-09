"""Request and response models for `POST /feeding-summary`."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import PetType


class FeedingSummaryStatus(StrEnum):
    """How a pet's logged intake compares to its computed daily calorie target.

    No free-text title/message here on purpose — the frontend owns wording/i18n
    and renders per-status copy in the user's locale from this enum + the
    numeric fields (targetCalories/actualCalories/deviationPct).
    """

    EXTREME_UNDER_TARGET = "EXTREME_UNDER_TARGET"  # < -50% of target
    UNDER_TARGET = "UNDER_TARGET"  # -50% to -10% of target
    ON_TARGET = "ON_TARGET"  # within ±10% of target
    OVER_TARGET = "OVER_TARGET"  # +10% to +100% of target
    EXTREME_OVER_TARGET = "EXTREME_OVER_TARGET"  # > +100% of target


class FeedingProduct(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    calories: float = Field(..., ge=0, le=20_000)


class FeedingSummaryPet(BaseModel):
    pet_id: str = Field(..., alias="petId", min_length=1, max_length=100)
    species: PetType = Field(default=PetType.DOG)
    breed: str | None = Field(default=None, examples=["Labrador"])
    weight_kg: float = Field(..., alias="weightKg", gt=0, le=500)
    age_months: int | None = Field(
        default=None,
        alias="ageMonths",
        description="Used to apply a growth-energy multiplier for juveniles (< 12mo) "
        "instead of the adult maintenance factor — a growing kitten/puppy legitimately "
        "needs far more kcal/kg than an adult at the same weight.",
        ge=0,
        le=600,
    )
    products: list[FeedingProduct] = Field(default_factory=list, max_length=100)

    model_config = ConfigDict(populate_by_name=True)


class FeedingSummaryRequest(BaseModel):
    pets: list[FeedingSummaryPet] = Field(..., min_length=1, max_length=1000)


class FeedingSummaryResult(BaseModel):
    pet_id: str = Field(..., alias="petId")
    status: FeedingSummaryStatus
    target_calories: float = Field(..., alias="targetCalories")
    actual_calories: float = Field(..., alias="actualCalories")
    deviation_pct: float = Field(..., alias="deviationPct")

    model_config = ConfigDict(populate_by_name=True)


class FeedingSummaryResponse(BaseModel):
    results: list[FeedingSummaryResult]
    disclaimer: str

    model_config = ConfigDict(populate_by_name=True)
