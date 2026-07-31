from datetime import UTC, datetime, timedelta
from unittest import TestCase

from app.schemas import (
    ReminderType,
    WellnessActivity,
    WellnessFeeding,
    WellnessMedication,
    WellnessPet,
    WellnessPreventiveCare,
    WellnessRequest,
    WellnessScoreStatus,
    WellnessWeightMeasurement,
)
from app.services.wellness_service import WellnessService


class _FakeGeminiResponse:
    text = """
    {
      "narrative": "Several tracked dimensions need attention.",
      "recommendations": [
        "Schedule a veterinary appointment and start a flea prevention product.",
        "Review portion sizes and improve the current diet.",
        "Improve medication adherence and discuss changing the dose.",
        "Add an omega-3 supplement.",
        "Increase daily activity."
      ]
    }
    """


class _FakeGeminiModels:
    def generate_content(self, **_kwargs: object) -> _FakeGeminiResponse:
        return _FakeGeminiResponse()


class _FakeGeminiClient:
    models = _FakeGeminiModels()


class _ConsultSoonPrediction:
    predicted_condition = "Digestive Issues"
    confidence = 0.9


class _ConsultSoonPredictor:
    def predict(self, _text: str) -> _ConsultSoonPrediction:
        return _ConsultSoonPrediction()


class WellnessServiceTests(TestCase):
    def setUp(self) -> None:
        self.service = WellnessService(api_key=None)

    def test_only_feeding_data_is_insufficient_for_a_reliable_score(self) -> None:
        request = WellnessRequest(
            pet=WellnessPet(species="dog"),
            feeding=WellnessFeeding(
                avg_meals_per_day=2,
                food_types=["dry", "wet"],
                consistency_days=7,
            ),
        )

        response = self.service.score(request)

        self.assertIsNone(response.wellness_score)
        self.assertEqual(response.score_status, WellnessScoreStatus.INSUFFICIENT_DATA)
        self.assertTrue(response.breakdown.diet.included)
        self.assertFalse(response.breakdown.activity.included)
        self.assertFalse(response.breakdown.sleep.included)
        self.assertFalse(response.breakdown.symptoms.included)
        self.assertFalse(response.breakdown.preventive_care.included)
        self.assertFalse(response.breakdown.baseline.included)

    def test_medication_adherence_changes_score_and_adds_reminder_type(self) -> None:
        care = WellnessPreventiveCare(
            recent_vet_visit=True,
            vaccinations_up_to_date=True,
        )
        missed_request = WellnessRequest(
            pet=WellnessPet(species="dog", age_months=36),
            activity=WellnessActivity(
                avg_steps_per_day=8000,
                avg_active_minutes_per_day=45,
            ),
            feeding=WellnessFeeding(
                avg_meals_per_day=2,
                food_types=["dry", "wet"],
                consistency_days=7,
            ),
            preventive_care=care,
            active_medications=[
                WellnessMedication(
                    name="Carprofen",
                    frequency="daily",
                    scheduled_doses=10,
                    completed_doses=0,
                )
            ],
        )
        completed_request = WellnessRequest(
            pet=WellnessPet(species="dog", age_months=36),
            activity=WellnessActivity(
                avg_steps_per_day=8000,
                avg_active_minutes_per_day=45,
            ),
            feeding=WellnessFeeding(
                avg_meals_per_day=2,
                food_types=["dry", "wet"],
                consistency_days=7,
            ),
            preventive_care=care,
            active_medications=[
                WellnessMedication(
                    name="Carprofen",
                    frequency="daily",
                    scheduled_doses=10,
                    completed_doses=10,
                )
            ],
        )

        missed_response = self.service.score(missed_request)
        completed_response = self.service.score(completed_request)

        self.assertEqual(missed_response.wellness_score, 96)
        self.assertEqual(completed_response.wellness_score, 100)
        self.assertIn(
            ReminderType.MEDICATION,
            [reminder.reminder for reminder in missed_response.reminders],
        )
        medication_reminder = next(
            reminder
            for reminder in missed_response.reminders
            if reminder.reminder == ReminderType.MEDICATION
        )
        self.assertEqual(
            medication_reminder.text,
            "Check the existing medication schedule recorded for your pet.",
        )
        self.assertNotIn("Carprofen", medication_reminder.text)
        self.assertNotIn(
            ReminderType.MEDICATION,
            [reminder.reminder for reminder in completed_response.reminders],
        )

    def test_weight_history_changes_baseline_and_remains_text_recommendation(self) -> None:
        now = datetime.now(UTC)
        stable_history = [
            WellnessWeightMeasurement(weight_kg=10, measured_at=now - timedelta(days=30)),
            WellnessWeightMeasurement(weight_kg=10.2, measured_at=now),
        ]
        unstable_history = [
            WellnessWeightMeasurement(weight_kg=10, measured_at=now - timedelta(days=30)),
            WellnessWeightMeasurement(weight_kg=12, measured_at=now),
        ]

        stable_response = self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="dog"),
                activity=WellnessActivity(
                    avg_steps_per_day=8000,
                    avg_active_minutes_per_day=45,
                ),
                feeding=WellnessFeeding(
                    avg_meals_per_day=2,
                    food_types=["dry", "wet"],
                    consistency_days=7,
                ),
                preventive_care=WellnessPreventiveCare(
                    recent_vet_visit=True,
                    vaccinations_up_to_date=True,
                ),
                weight_history=stable_history,
            )
        )
        unstable_response = self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="dog"),
                activity=WellnessActivity(
                    avg_steps_per_day=8000,
                    avg_active_minutes_per_day=45,
                ),
                feeding=WellnessFeeding(
                    avg_meals_per_day=2,
                    food_types=["dry", "wet"],
                    consistency_days=7,
                ),
                preventive_care=WellnessPreventiveCare(
                    recent_vet_visit=True,
                    vaccinations_up_to_date=True,
                ),
                weight_history=unstable_history,
            )
        )

        self.assertEqual(stable_response.wellness_score, 100)
        self.assertEqual(unstable_response.wellness_score, 83)
        self.assertEqual(unstable_response.reminders, [])
        self.assertIn(
            "Continue recording weight regularly and monitor the trend for further changes.",
            unstable_response.recommendations,
        )

    def test_reminders_use_exact_backend_values_with_actionable_text(self) -> None:
        request = WellnessRequest(
            pet=WellnessPet(species="dog"),
            activity=WellnessActivity(
                avg_steps_per_day=100,
                avg_active_minutes_per_day=2,
            ),
            feeding=WellnessFeeding(
                avg_meals_per_day=0.5,
                food_types=[],
                consistency_days=0,
            ),
            active_medications=[
                WellnessMedication(
                    name="Medicine",
                    scheduled_doses=10,
                    completed_doses=5,
                )
            ],
            preventive_care=WellnessPreventiveCare(
                recent_vet_visit=False,
                vaccinations_up_to_date=False,
            ),
        )

        response = self.service.score(request)
        payload = response.model_dump(by_alias=True, mode="json")

        self.assertEqual(
            [reminder["reminder"] for reminder in payload["reminders"]],
            ["Feeding", "Activity", "Medication", "Vaccination", "VetVisit"],
        )
        self.assertTrue(
            all(reminder["text"] for reminder in payload["reminders"])
        )

    def test_backend_camel_case_request_contract_validates(self) -> None:
        request = WellnessRequest.model_validate(
            {
                "pet": {"species": "dog", "weightKg": 12.4},
                "activeMedications": [
                    {
                        "name": "Medicine",
                        "scheduledDoses": 7,
                        "completedDoses": 6,
                    }
                ],
                "weightHistory": [
                    {
                        "weightKg": 12.2,
                        "measuredAt": "2026-06-30T08:00:00Z",
                    },
                    {
                        "weightKg": 12.4,
                        "measuredAt": "2026-07-30T08:00:00Z",
                    },
                ],
            }
        )

        self.assertEqual(request.active_medications[0].scheduled_doses, 7)
        self.assertEqual(request.weight_history[-1].weight_kg, 12.4)

    def test_fish_sleep_is_not_scored(self) -> None:
        response = self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="fish"),
                activity=WellnessActivity(avg_sleep_hours_per_day=4),
            )
        )

        self.assertFalse(response.breakdown.sleep.included)

    def test_no_medication_schedule_suggests_vet_visit_not_a_drug(self) -> None:
        response = self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="dog", age_months=36),
                activity=WellnessActivity(
                    avg_steps_per_day=8000,
                    avg_active_minutes_per_day=45,
                ),
                feeding=WellnessFeeding(
                    avg_meals_per_day=2,
                    food_types=["dry", "wet"],
                    consistency_days=7,
                ),
                current_symptoms="vomiting and not eating",
            ),
            predictor=_ConsultSoonPredictor(),
        )

        self.assertNotIn(
            ReminderType.MEDICATION,
            [reminder.reminder for reminder in response.reminders],
        )
        self.assertIn(
            {
                "reminder": "VetVisit",
                "text": "Consult a veterinarian about whether clinical treatment is needed.",
            },
            response.model_dump(by_alias=True, mode="json")["reminders"],
        )

    def test_recommendations_exclude_reminder_duplicates_and_medication_advice(self) -> None:
        now = datetime.now(UTC)
        self.service.client = _FakeGeminiClient()
        request = WellnessRequest(
            pet=WellnessPet(species="dog"),
            activity=WellnessActivity(
                avg_steps_per_day=100,
                avg_active_minutes_per_day=2,
            ),
            feeding=WellnessFeeding(
                avg_meals_per_day=1,
                food_types=["dry"],
                consistency_days=3,
            ),
            active_medications=[
                WellnessMedication(
                    name="Carprofen",
                    scheduled_doses=7,
                    completed_doses=4,
                )
            ],
            preventive_care=WellnessPreventiveCare(
                recent_vet_visit=False,
                vaccinations_up_to_date=False,
            ),
            weight_history=[
                WellnessWeightMeasurement(
                    weight_kg=10,
                    measured_at=now - timedelta(days=30),
                ),
                WellnessWeightMeasurement(weight_kg=12, measured_at=now),
            ],
        )

        response = self.service.score(request)

        self.assertEqual(
            response.recommendations,
            [
                "Continue recording weight regularly and monitor the trend for further changes."
            ],
        )
        self.assertEqual(
            [reminder.reminder for reminder in response.reminders],
            [
                ReminderType.FEEDING,
                ReminderType.ACTIVITY,
                ReminderType.MEDICATION,
                ReminderType.VACCINATION,
                ReminderType.VET_VISIT,
            ],
        )
