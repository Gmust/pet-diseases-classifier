"""
Dimension scorers exercised directly.

The point of these is the import list: each scorer is reachable without building
a WellnessService, without an LLM client, and without touching another
dimension. If one of these ever needs a service to run, the split has regressed.
"""

from __future__ import annotations

from app.wellness.schemas import (
    WellnessActivity,
    WellnessDimensionAvailability,
    WellnessFeeding,
    WellnessPet,
)
from app.wellness.scoring.activity import _score_activity
from app.wellness.scoring.diet import _score_diet
from app.wellness.scoring.sleep import _score_sleep


def test_activity_scorer_rewards_meeting_the_species_target():
    at_target = _score_activity(
        WellnessActivity(avg_steps_per_day=8000, avg_active_minutes_per_day=60),
        "dog",
    )
    well_under = _score_activity(
        WellnessActivity(avg_steps_per_day=500, avg_active_minutes_per_day=5),
        "dog",
    )

    assert at_target.availability == WellnessDimensionAvailability.AVAILABLE
    assert well_under.availability == WellnessDimensionAvailability.AVAILABLE
    assert at_target.score > well_under.score
    assert 0 <= well_under.score <= at_target.max_score


def test_activity_scorer_reports_missing_rather_than_zero():
    item = _score_activity(None, "dog")

    assert item.availability == WellnessDimensionAvailability.MISSING
    assert item.included is False
    assert item.score == 0


def test_sleep_scorer_penalises_both_directions_from_the_norm():
    in_norm = _score_sleep(WellnessActivity(avg_sleep_hours_per_day=12.5), "dog")
    too_little = _score_sleep(WellnessActivity(avg_sleep_hours_per_day=4.0), "dog")
    too_much = _score_sleep(WellnessActivity(avg_sleep_hours_per_day=22.0), "dog")

    assert in_norm.score > too_little.score
    assert in_norm.score > too_much.score


def test_diet_scorer_uses_pet_weight_to_derive_the_calorie_target():
    pet = WellnessPet(species="dog", weight_kg=20.0)
    on_target = _score_diet(
        WellnessFeeding(avg_calories_per_day=800, consistency_days=7),
        pet,
    )
    starved = _score_diet(
        WellnessFeeding(avg_calories_per_day=50, consistency_days=7),
        pet,
    )

    assert on_target.availability == WellnessDimensionAvailability.AVAILABLE
    assert on_target.score > starved.score


def test_diet_scorer_rewards_logging_consistency():
    pet = WellnessPet(species="dog", weight_kg=20.0)
    every_day = _score_diet(WellnessFeeding(consistency_days=7), pet)
    one_day = _score_diet(WellnessFeeding(consistency_days=1), pet)

    assert every_day.score > one_day.score


def test_scorers_are_independent_of_each_other():
    """Each scorer sees only its own slice of the request."""
    activity = WellnessActivity(avg_steps_per_day=8000, avg_active_minutes_per_day=60)

    # Sleep is absent from this activity payload, so sleep reports MISSING while
    # activity still scores normally — no shared state between them.
    assert _score_activity(activity, "dog").availability == WellnessDimensionAvailability.AVAILABLE
    assert _score_sleep(activity, "dog").availability == WellnessDimensionAvailability.MISSING
