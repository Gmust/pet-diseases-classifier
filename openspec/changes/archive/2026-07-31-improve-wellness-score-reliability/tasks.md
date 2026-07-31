## 1. Reliability Schema

- [x] 1.1 Add score-status, dimension-availability, evaluation-window, reason-code, and bounded-evidence schema types.
- [x] 1.2 Extend `WellnessRequest` with the optional validated `evaluationWindow`.
- [x] 1.3 Extend breakdown and response schemas with availability, evidence, coverage, calculation metadata, and nullable insufficient-data score fields.
- [x] 1.4 Add typed wellness-dimension and `trackingRecommendations` response models with machine-readable `requiredInputs`.
- [x] 1.5 Extend tracking recommendations with typed `suggestedReminderTypes` using exact backend wire values.

## 2. Rule Evaluation

- [x] 2.1 Refactor each wellness dimension scorer to return `AVAILABLE`, `MISSING`, or `NOT_APPLICABLE` with stable reason codes and allowlisted evidence.
- [x] 2.2 Implement weighted coverage and score-status calculation from dimension maxima and availability.
- [x] 2.3 Preserve partial-score normalization over available dimensions and distinguish an available zero score from no evaluable data.
- [x] 2.4 Add the `2.0.0` calculation version, UTC evaluation timestamp, and evaluation-window echo to response assembly.
- [x] 2.5 Implement the deterministic, ordered tracking-recommendation catalog for actionable missing dimensions and treat absent symptom text as not applicable.
- [x] 2.6 Gate numeric scores on 60% weighted coverage, three foundational dimensions, Diet, applicable Activity-or-Sleep, and PreventiveCare-or-Baseline availability.
- [x] 2.7 Add deterministic Activity, Diet, and PreventiveCare reminder-type suggestions while leaving unsupported dimensions empty.
- [x] 2.8 Add positive Activity, Diet, and PreventiveCare maintenance guidance for complete GOOD/EXCELLENT assessments with reminder-type deduplication.

## 3. Narrative and Insufficient-Data Behavior

- [x] 3.1 Return the deterministic nullable insufficient-data response with tracking recommendations and skip Gemini and score-derived reminders when the reliability gate is not met.
- [x] 3.2 Add partial status, coverage, and missing-dimension context to the Gemini prompt without exposing sensitive evidence.
- [x] 3.3 Preserve existing reminder deduplication and medication-safety behavior for complete and partial assessments.

## 4. Verification

- [x] 4.1 Add contract tests for complete, weighted partial, species-inapplicable, genuine zero, and insufficient-data responses.
- [x] 4.2 Add tests for availability/included consistency, stable reason codes, allowlisted evidence, and absence of raw symptoms or medication names.
- [x] 4.3 Add validation tests for valid, omitted, and reversed evaluation windows plus UTC timestamp and calculation-version serialization.
- [x] 4.4 Add tests proving Gemini is skipped for insufficient data and receives partial-assessment context otherwise.
- [x] 4.5 Add tests for deterministic tracking-recommendation ordering, deduplication, valid request-field paths, symptom non-applicability, and medical-safety wording.
- [x] 4.6 Add reliability-gate tests for feeding-only, below-threshold, exact-threshold, missing foundational groups, eligible partial, and eligible zero-score responses.
- [x] 4.7 Add contract tests for exact suggested reminder mappings, backend enum compatibility, unsupported dimensions, medication exclusion, and serialized wire values.
- [x] 4.8 Add tests for complete GOOD/EXCELLENT maintenance guidance, lower-band omission, positive wording, supported dimensions, and problem-reminder deduplication.

## 5. Documentation and Consumer Migration

- [x] 5.1 Update the wellness API reference with complete, partial, and insufficient-data examples, coverage math, and tracking-recommendation examples.
- [x] 5.2 Document the breaking nullable fields, calculation-version policy, reason-code contract, and required C# consumer migration.
- [x] 5.3 Run compilation, the complete wellness test suite, schema serialization checks, and `git diff --check`.
- [x] 5.4 Update API examples and migration guidance for the minimum reliable-score gate and rerun verification.
- [x] 5.5 Document reminder-type suggestions, user confirmation ownership, and absence of scheduling side effects, then rerun verification.
- [x] 5.6 Document complete-state maintenance recommendations and rerun the complete verification suite.
