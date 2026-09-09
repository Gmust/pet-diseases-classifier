from datetime import UTC, datetime, timedelta
from unittest import TestCase

from pydantic import ValidationError

from app.wellness.norms import CALCULATION_VERSION
from app.wellness.schemas import (
    ReminderType,
    WellnessActivity,
    WellnessDimension,
    WellnessDimensionAvailability,
    WellnessEvaluationWindow,
    WellnessFeeding,
    WellnessMedication,
    WellnessPet,
    WellnessPreventiveCare,
    WellnessRequest,
    WellnessRoutineCareEntry,
    WellnessScoreStatus,
    WellnessWeightMeasurement,
)
from app.wellness.service import WellnessService
from app.wellness.tracking import _ROUTINE_CARE_TYPES


class _NarrativeResponse:
    text = """
    {
      "narrative": "This report is based on the tracked dimensions.",
      "recommendations": ["Maintain the observed healthy routines."]
    }
    """


class _CapturingModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> _NarrativeResponse:
        self.calls.append(kwargs)
        return _NarrativeResponse()


class _CapturingClient:
    def __init__(self) -> None:
        self.models = _CapturingModels()


class _Prediction:
    predicted_condition = "Digestive Issues"
    confidence = 0.91


class _Predictor:
    def predict(self, _text: str) -> _Prediction:
        return _Prediction()


def _complete_request(species: str = "dog") -> WellnessRequest:
    now = datetime.now(UTC)
    return WellnessRequest(
        pet=WellnessPet(species=species, age_months=36, weight_kg=12),
        activity=WellnessActivity(
            avg_steps_per_day=8000,
            avg_active_minutes_per_day=45,
            avg_sleep_hours_per_day=13,
            days_tracked=7,
        ),
        feeding=WellnessFeeding(
            avg_meals_per_day=2,
            avg_calories_per_day=420,
            food_types=["dry", "wet"],
            consistency_days=7,
        ),
        preventive_care=WellnessPreventiveCare(
            recent_vet_visit=True,
            vaccinations_up_to_date=True,
        ),
        weight_history=[
            WellnessWeightMeasurement(
                weight_kg=12,
                measured_at=now - timedelta(days=30),
            ),
            WellnessWeightMeasurement(weight_kg=12.1, measured_at=now),
        ],
    )


def _schema_resolves_request_path(path: str) -> bool:
    schema = WellnessRequest.model_json_schema(by_alias=True)
    node: dict[str, object] = schema
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        return False

    for segment in path.split("."):
        reference = node.get("$ref")
        if isinstance(reference, str):
            definition_name = reference.rsplit("/", maxsplit=1)[-1]
            resolved = definitions.get(definition_name)
            if not isinstance(resolved, dict):
                return False
            node = resolved

        properties = node.get("properties")
        if not isinstance(properties, dict):
            return False
        child = properties.get(segment)
        if not isinstance(child, dict):
            return False
        node = child

        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            non_null = [
                option
                for option in any_of
                if isinstance(option, dict) and option.get("type") != "null"
            ]
            if len(non_null) == 1:
                node = non_null[0]
    return True


class WellnessReliabilityTests(TestCase):
    def setUp(self) -> None:
        self.service = WellnessService(api_key=None)

    def test_complete_response_has_full_coverage_and_metadata(self) -> None:
        request = _complete_request()
        request.evaluation_window = WellnessEvaluationWindow(
            start_date="2026-07-01",
            end_date="2026-07-07",
        )

        response = self.service.score(request)
        payload = response.model_dump(by_alias=True, mode="json")

        self.assertEqual(response.score_status, WellnessScoreStatus.COMPLETE)
        self.assertEqual(response.data_coverage, 1.0)
        self.assertEqual(response.calculation_version, CALCULATION_VERSION)
        self.assertEqual(response.evaluated_at.utcoffset(), timedelta(0))
        self.assertEqual(
            payload["evaluationWindow"],
            {"startDate": "2026-07-01", "endDate": "2026-07-07"},
        )
        self.assertEqual(
            [item.dimension for item in response.tracking_recommendations],
            [
                WellnessDimension.ACTIVITY,
                WellnessDimension.DIET,
                WellnessDimension.PREVENTIVE_CARE,
                WellnessDimension.ROUTINE_CARE,
            ],
        )
        self.assertEqual(
            response.breakdown.symptoms.availability,
            WellnessDimensionAvailability.NOT_APPLICABLE,
        )

    def test_weighted_partial_score_uses_only_available_dimensions(self) -> None:
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
            )
        )

        self.assertEqual(response.score_status, WellnessScoreStatus.PARTIAL)
        self.assertEqual(response.wellness_score, 100)
        self.assertEqual(response.data_coverage, 0.6667)

    def test_feeding_only_is_below_threshold_and_returns_no_score(self) -> None:
        response = self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="dog"),
                feeding=WellnessFeeding(
                    avg_meals_per_day=2,
                    food_types=["dry", "wet"],
                    consistency_days=7,
                ),
            )
        )

        self.assertEqual(response.data_coverage, 0.2667)
        self.assertEqual(
            response.score_status,
            WellnessScoreStatus.INSUFFICIENT_DATA,
        )
        self.assertIsNone(response.wellness_score)

    def test_exact_coverage_threshold_with_required_groups_is_eligible(self) -> None:
        response = self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="dog", age_months=36),
                activity=WellnessActivity(avg_sleep_hours_per_day=13),
                feeding=WellnessFeeding(
                    avg_meals_per_day=2,
                    food_types=["dry", "wet"],
                    consistency_days=7,
                ),
            )
        )

        self.assertEqual(response.data_coverage, 0.6)
        self.assertEqual(response.score_status, WellnessScoreStatus.PARTIAL)
        self.assertIsNotNone(response.wellness_score)

    def test_missing_health_context_blocks_score_despite_high_coverage(self) -> None:
        response = self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="dog"),
                activity=WellnessActivity(
                    avg_steps_per_day=8000,
                    avg_active_minutes_per_day=45,
                    avg_sleep_hours_per_day=13,
                ),
                feeding=WellnessFeeding(
                    avg_meals_per_day=2,
                    food_types=["dry", "wet"],
                    consistency_days=7,
                ),
            )
        )

        self.assertEqual(response.data_coverage, 0.7333)
        self.assertEqual(
            response.score_status,
            WellnessScoreStatus.INSUFFICIENT_DATA,
        )
        self.assertIsNone(response.wellness_score)

    def test_fish_inapplicable_dimensions_do_not_reduce_coverage(self) -> None:
        response = self.service.score(_complete_request(species="fish"))

        self.assertEqual(
            response.breakdown.activity.availability,
            WellnessDimensionAvailability.NOT_APPLICABLE,
        )
        self.assertEqual(
            response.breakdown.sleep.availability,
            WellnessDimensionAvailability.NOT_APPLICABLE,
        )
        self.assertEqual(response.score_status, WellnessScoreStatus.COMPLETE)
        self.assertEqual(response.data_coverage, 1.0)

    def test_available_zero_is_not_insufficient_data(self) -> None:
        now = datetime.now(UTC)
        response = self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="dog"),
                activity=WellnessActivity(
                    avg_steps_per_day=0,
                    avg_active_minutes_per_day=0,
                ),
                feeding=WellnessFeeding(avg_calories_per_day=1000),
                weight_history=[
                    WellnessWeightMeasurement(
                        weight_kg=10,
                        measured_at=now - timedelta(days=30),
                    ),
                    WellnessWeightMeasurement(weight_kg=12, measured_at=now),
                ],
            )
        )

        self.assertEqual(response.wellness_score, 0)
        self.assertIsNotNone(response.band)
        self.assertEqual(response.score_status, WellnessScoreStatus.PARTIAL)

    def test_no_evaluable_dimensions_returns_nullable_score_without_gemini(self) -> None:
        client = _CapturingClient()
        self.service.client = client

        response = self.service.score(WellnessRequest(pet=WellnessPet(species="dog")))

        self.assertEqual(
            response.score_status,
            WellnessScoreStatus.INSUFFICIENT_DATA,
        )
        self.assertIsNone(response.wellness_score)
        self.assertIsNone(response.band)
        self.assertIsNone(response.band_label)
        self.assertEqual(response.data_coverage, 0)
        self.assertIn("not enough tracked data", response.narrative.lower())
        self.assertEqual(response.reminders, [])
        self.assertEqual(response.recommendations, [])
        self.assertEqual(client.models.calls, [])

    def test_insufficient_data_preserves_classifier_safety_reminder(self) -> None:
        response = self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="dog"),
                current_symptoms="vomiting and not eating",
            ),
            predictor=_Predictor(),
        )

        self.assertEqual(
            response.score_status,
            WellnessScoreStatus.INSUFFICIENT_DATA,
        )
        self.assertEqual(
            response.model_dump(by_alias=True, mode="json")["reminders"],
            [
                {
                    "reminder": "VetVisit",
                    "text": (
                        "Consult a veterinarian about whether clinical treatment " "is needed."
                    ),
                }
            ],
        )

    def test_availability_included_reason_codes_and_evidence_are_consistent(self) -> None:
        request = _complete_request()
        request.current_symptoms = "private raw symptom phrase"
        request.active_medications = [
            WellnessMedication(
                name="PrivateMedicationName",
                scheduled_doses=7,
                completed_doses=5,
            )
        ]

        response = self.service.score(request, predictor=_Predictor())
        for _, item in response.breakdown:
            self.assertEqual(
                item.included,
                item.availability == WellnessDimensionAvailability.AVAILABLE,
            )
            self.assertTrue(item.reason_codes)
            for code in item.reason_codes:
                self.assertEqual(code.value, code.value.upper())

        evidence_dump = str(response.breakdown.model_dump(by_alias=True, mode="json"))
        self.assertNotIn("private raw symptom phrase", evidence_dump)
        self.assertNotIn("PrivateMedicationName", evidence_dump)

    def test_medication_only_keeps_twenty_percent_preventive_weight(self) -> None:
        response = self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="dog"),
                active_medications=[
                    WellnessMedication(
                        name="Medicine",
                        scheduled_doses=7,
                        completed_doses=7,
                    )
                ],
            )
        )

        self.assertTrue(response.breakdown.preventive_care.included)
        self.assertEqual(response.breakdown.preventive_care.score, 2)
        self.assertEqual(response.breakdown.preventive_care.max_score, 10)

    def test_mixed_timezone_weight_history_is_normalized_before_sorting(self) -> None:
        request = _complete_request()
        request.weight_history = [
            WellnessWeightMeasurement(
                weight_kg=10,
                measured_at=datetime(2026, 1, 1, 12),
            ),
            WellnessWeightMeasurement(
                weight_kg=10.2,
                measured_at=datetime(2026, 1, 2, 12, tzinfo=UTC),
            ),
        ]

        response = self.service.score(request)

        self.assertEqual(response.breakdown.baseline.evidence["weightStability"], 1)

    def test_calorie_fit_evidence_contains_inputs_and_derived_values(self) -> None:
        response = self.service.score(_complete_request())

        evidence = response.breakdown.diet.evidence
        self.assertEqual(evidence["avgCaloriesPerDay"], 420)
        self.assertEqual(evidence["weightKg"], 12)
        self.assertEqual(evidence["calorieTargetPerDay"], 420)
        self.assertEqual(evidence["calorieRatio"], 1)

    def test_evaluation_window_may_be_omitted_and_rejects_reverse_order(self) -> None:
        response = self.service.score(_complete_request())
        self.assertIsNone(response.evaluation_window)

        with self.assertRaises(ValidationError):
            WellnessRequest.model_validate(
                {
                    "pet": {"species": "dog"},
                    "evaluationWindow": {
                        "startDate": "2026-07-08",
                        "endDate": "2026-07-01",
                    },
                }
            )

    def test_partial_gemini_prompt_exposes_status_coverage_and_missing_dimensions(self) -> None:
        client = _CapturingClient()
        self.service.client = client

        self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="dog", age_months=36),
                activity=WellnessActivity(
                    avg_steps_per_day=8000,
                    avg_active_minutes_per_day=45,
                ),
                feeding=WellnessFeeding(
                    avg_meals_per_day=2,
                    consistency_days=7,
                ),
            )
        )

        self.assertEqual(len(client.models.calls), 1)
        prompt = client.models.calls[0]["contents"]
        self.assertIsInstance(prompt, str)
        self.assertIn("Assessment status: PARTIAL", prompt)
        self.assertIn("Weighted data coverage: 66.67%", prompt)
        self.assertIn(
            "Missing dimensions: Sleep, PreventiveCare",
            prompt,
        )
        self.assertNotIn("evidence", prompt.lower())

    def test_tracking_recommendations_are_ordered_safe_and_schema_valid(self) -> None:
        response = self.service.score(
            WellnessRequest(
                pet=WellnessPet(species="dog"),
                feeding=WellnessFeeding(
                    avg_meals_per_day=2,
                    consistency_days=7,
                ),
            )
        )

        self.assertEqual(
            [item.dimension for item in response.tracking_recommendations],
            [
                WellnessDimension.ACTIVITY,
                WellnessDimension.SLEEP,
                WellnessDimension.PREVENTIVE_CARE,
                WellnessDimension.BASELINE,
                WellnessDimension.ROUTINE_CARE,
            ],
        )
        self.assertEqual(
            len(response.tracking_recommendations),
            len({item.dimension for item in response.tracking_recommendations}),
        )
        for recommendation in response.tracking_recommendations:
            for path in recommendation.required_inputs:
                self.assertTrue(
                    _schema_resolves_request_path(path),
                    msg=f"Unknown WellnessRequest path: {path}",
                )

        text = " ".join(item.text.lower() for item in response.tracking_recommendations)
        for unsafe_term in (
            "diagnos",
            "treatment",
            "medication",
            "medicine",
            "dose",
            "drug",
            "schedule an appointment",
        ):
            self.assertNotIn(unsafe_term, text)
        self.assertNotIn(
            WellnessDimension.SYMPTOMS,
            [item.dimension for item in response.tracking_recommendations],
        )

    def test_tracking_recommendations_suggest_exact_backend_reminder_types(self) -> None:
        response = self.service.score(WellnessRequest(pet=WellnessPet(species="dog")))
        expected = {
            WellnessDimension.ACTIVITY: [ReminderType.ACTIVITY],
            WellnessDimension.SLEEP: [],
            WellnessDimension.DIET: [ReminderType.FEEDING],
            WellnessDimension.PREVENTIVE_CARE: [
                ReminderType.VACCINATION,
                ReminderType.VET_VISIT,
            ],
            WellnessDimension.BASELINE: [ReminderType.WEIGHING],
            WellnessDimension.ROUTINE_CARE: list(_ROUTINE_CARE_TYPES),
        }

        self.assertEqual(
            {
                item.dimension: item.suggested_reminder_types
                for item in response.tracking_recommendations
            },
            expected,
        )
        self.assertNotIn(
            ReminderType.MEDICATION,
            [
                reminder_type
                for item in response.tracking_recommendations
                for reminder_type in item.suggested_reminder_types
            ],
        )

        payload = response.model_dump(by_alias=True, mode="json")
        payload_by_dimension = {
            item["dimension"]: item for item in payload["trackingRecommendations"]
        }
        self.assertEqual(
            payload_by_dimension["Activity"]["suggestedReminderTypes"],
            ["Activity"],
        )
        self.assertEqual(
            payload_by_dimension["PreventiveCare"]["suggestedReminderTypes"],
            ["Vaccination", "VetVisit"],
        )
        for item in payload["trackingRecommendations"]:
            self.assertEqual(
                set(item),
                {
                    "dimension",
                    "text",
                    "requiredInputs",
                    "suggestedReminderTypes",
                },
            )

    def test_complete_excellent_score_returns_positive_maintenance_guidance(self) -> None:
        response = self.service.score(_complete_request())

        self.assertEqual(response.score_status, WellnessScoreStatus.COMPLETE)
        self.assertEqual(response.band.value, "EXCELLENT")
        self.assertEqual(
            {
                item.dimension: item.suggested_reminder_types
                for item in response.tracking_recommendations
            },
            {
                WellnessDimension.ACTIVITY: [ReminderType.ACTIVITY],
                WellnessDimension.DIET: [ReminderType.FEEDING],
                WellnessDimension.PREVENTIVE_CARE: [
                    ReminderType.VACCINATION,
                    ReminderType.VET_VISIT,
                ],
                WellnessDimension.ROUTINE_CARE: list(_ROUTINE_CARE_TYPES),
            },
        )
        for item in response.tracking_recommendations:
            self.assertIn("keep", item.text.lower())
            self.assertNotIn("missing", item.text.lower())

    def test_complete_lower_band_omits_maintenance_guidance(self) -> None:
        request = _complete_request()
        request.activity = WellnessActivity(
            avg_steps_per_day=0,
            avg_active_minutes_per_day=0,
            avg_sleep_hours_per_day=24,
            days_tracked=7,
        )
        request.feeding = WellnessFeeding(
            avg_meals_per_day=0.2,
            avg_calories_per_day=10000,
            consistency_days=0,
        )

        response = self.service.score(request)

        self.assertEqual(response.score_status, WellnessScoreStatus.COMPLETE)
        self.assertIn(response.band.value, {"FAIR", "CONCERNING", "CRITICAL"})
        # Routine care is independent of the band, so only it survives here.
        self.assertEqual(
            [item.dimension for item in response.tracking_recommendations],
            [WellnessDimension.ROUTINE_CARE],
        )

    def test_maintenance_suggestions_do_not_duplicate_problem_reminders(self) -> None:
        request = _complete_request()
        request.activity = WellnessActivity(
            avg_steps_per_day=5600,
            avg_active_minutes_per_day=31.5,
            avg_sleep_hours_per_day=13,
            days_tracked=7,
        )
        request.preventive_care = WellnessPreventiveCare(
            recent_vet_visit=True,
            vaccinations_up_to_date=False,
        )

        response = self.service.score(request)

        self.assertEqual(response.score_status, WellnessScoreStatus.COMPLETE)
        self.assertEqual(response.band.value, "GOOD")
        self.assertEqual(
            [item.reminder for item in response.reminders],
            [ReminderType.ACTIVITY, ReminderType.VACCINATION],
        )
        maintenance = {
            item.dimension: item.suggested_reminder_types
            for item in response.tracking_recommendations
        }
        self.assertNotIn(WellnessDimension.ACTIVITY, maintenance)
        self.assertEqual(
            maintenance[WellnessDimension.DIET],
            [ReminderType.FEEDING],
        )
        self.assertEqual(
            maintenance[WellnessDimension.PREVENTIVE_CARE],
            [ReminderType.VET_VISIT],
        )

    def _routine_care_request(
        self,
        entries: list[WellnessRoutineCareEntry],
    ) -> WellnessRequest:
        request = _complete_request()
        request.evaluation_window = WellnessEvaluationWindow(
            start_date="2026-07-01",
            end_date="2026-07-31",
        )
        request.routine_care = entries
        return request

    def _routine_care_item(self, request: WellnessRequest):
        return [
            item
            for item in self.service.score(request).tracking_recommendations
            if item.dimension == WellnessDimension.ROUTINE_CARE
        ]

    def test_missing_routine_care_suggests_every_grooming_reminder_type(self) -> None:
        items = self._routine_care_item(self._routine_care_request([]))

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].suggested_reminder_types, list(_ROUTINE_CARE_TYPES))
        self.assertEqual(items[0].required_inputs, ["routineCare"])
        self.assertNotIn(ReminderType.DEWORMING, items[0].suggested_reminder_types)
        self.assertNotIn(ReminderType.MEDICATION, items[0].suggested_reminder_types)

    def test_routine_care_older_than_a_month_is_still_suggested(self) -> None:
        request = self._routine_care_request(
            [
                WellnessRoutineCareEntry(type=ReminderType.BATHING, last_done_at="2026-06-15"),
                WellnessRoutineCareEntry(type=ReminderType.BRUSHING, last_done_at="2026-07-20"),
            ]
        )

        items = self._routine_care_item(request)

        self.assertEqual(len(items), 1)
        self.assertIn(ReminderType.BATHING, items[0].suggested_reminder_types)
        self.assertNotIn(ReminderType.BRUSHING, items[0].suggested_reminder_types)

    def test_recent_generic_grooming_record_covers_every_label(self) -> None:
        request = self._routine_care_request(
            [WellnessRoutineCareEntry(type=ReminderType.GROOMING, last_done_at="2026-07-10")]
        )

        self.assertEqual(self._routine_care_item(request), [])

    def test_fully_recorded_routine_care_produces_no_suggestion(self) -> None:
        request = self._routine_care_request(
            [
                WellnessRoutineCareEntry(type=reminder_type, last_done_at="2026-07-25")
                for reminder_type in _ROUTINE_CARE_TYPES
            ]
        )

        self.assertEqual(self._routine_care_item(request), [])
