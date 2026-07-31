## ADDED Requirements

### Requirement: Weighted wellness data coverage
The service SHALL return `dataCoverage` as the weighted fraction of applicable wellness dimensions that were evaluated. It SHALL include `AVAILABLE` dimension maxima in the numerator, include `AVAILABLE` and `MISSING` dimension maxima in the denominator, exclude `NOT_APPLICABLE` dimensions from both, and return a value from 0 through 1 rounded to four decimal places.

#### Scenario: Complete applicable data
- **WHEN** every applicable wellness dimension is available
- **THEN** `dataCoverage` SHALL equal `1.0`

#### Scenario: Partial weighted data
- **WHEN** at least one applicable dimension is available and at least one applicable dimension is missing
- **THEN** `dataCoverage` SHALL equal the sum of available dimension maxima divided by the sum of all applicable dimension maxima

#### Scenario: Species-inapplicable dimension
- **WHEN** a dimension is not applicable to the pet species
- **THEN** that dimension's maximum SHALL NOT contribute to either side of the coverage calculation

### Requirement: Wellness score status
The service SHALL return `scoreStatus` as `COMPLETE`, `PARTIAL`, or `INSUFFICIENT_DATA` based on dimension availability and minimum score reliability. A numeric score SHALL require `dataCoverage >= 0.60`, at least three available foundational dimensions, available Diet, available PreventiveCare-or-Baseline, and available Activity-or-Sleep when either behavior dimension applies to the species.

#### Scenario: Complete assessment
- **WHEN** the minimum reliability gate is met and every applicable dimension is available
- **THEN** `scoreStatus` SHALL be `COMPLETE`

#### Scenario: Partial assessment
- **WHEN** the minimum reliability gate is met and at least one applicable dimension is missing
- **THEN** `scoreStatus` SHALL be `PARTIAL`

#### Scenario: Reliability gate not met
- **WHEN** weighted coverage is below `0.60` or any required foundational group is unavailable
- **THEN** `scoreStatus` SHALL be `INSUFFICIENT_DATA`

#### Scenario: Boundary coverage is eligible
- **WHEN** weighted coverage equals `0.60` and all required foundational groups are available
- **THEN** the coverage threshold SHALL be considered met

#### Scenario: Symptoms remain optional
- **WHEN** no symptom text is reported but the reliability gate is otherwise met
- **THEN** the service SHALL expose the eligible score

#### Scenario: Behavior dimensions are inapplicable to species
- **WHEN** both Activity and Sleep are `NOT_APPLICABLE`, at least three other foundational dimensions are available, and all other reliability conditions are met
- **THEN** absence of a behavior dimension SHALL NOT prevent score eligibility

### Requirement: Insufficient-data response
The service SHALL distinguish data that does not meet minimum reliability from an eligible wellness score, including a genuine eligible score of zero.

#### Scenario: Minimum reliability is not met
- **WHEN** `scoreStatus` is `INSUFFICIENT_DATA`
- **THEN** `wellnessScore`, `band`, and `bandLabel` SHALL be `null`
- **AND** the narrative SHALL state that tracked data is insufficient
- **AND** Gemini narrative generation SHALL NOT be called
- **AND** score-derived reminders SHALL be empty

#### Scenario: Eligible dimensions score zero
- **WHEN** the minimum reliability gate is met and the normalized score is zero
- **THEN** `wellnessScore` SHALL be `0`
- **AND** the response SHALL retain the corresponding non-null wellness band

### Requirement: Dimension availability
Every wellness breakdown item SHALL return `availability` as `AVAILABLE`, `MISSING`, or `NOT_APPLICABLE`, and its compatibility field `included` SHALL be true only for `AVAILABLE`.

#### Scenario: Sufficient source data
- **WHEN** a dimension has enough valid source data for its rule
- **THEN** `availability` SHALL be `AVAILABLE`
- **AND** `included` SHALL be true

#### Scenario: Required source data absent
- **WHEN** a dimension applies but lacks enough source data for evaluation
- **THEN** `availability` SHALL be `MISSING`
- **AND** `included` SHALL be false

#### Scenario: Rule does not apply to species
- **WHEN** the calculation rules mark a dimension inapplicable for the pet species
- **THEN** `availability` SHALL be `NOT_APPLICABLE`
- **AND** `included` SHALL be false

#### Scenario: No symptoms reported
- **WHEN** the request contains no symptom text
- **THEN** the symptoms dimension SHALL be `NOT_APPLICABLE`
- **AND** absence of symptom text SHALL NOT reduce `dataCoverage`

### Requirement: Explainable dimension results
Every breakdown item SHALL return at least one stable uppercase `reasonCode` and structured `evidence` sufficient to explain the availability or principal scoring outcome.

#### Scenario: Available dimension
- **WHEN** a dimension is evaluated
- **THEN** its `reasonCodes` SHALL identify the principal rule outcomes
- **AND** its `evidence` SHALL contain the allowlisted scalar values used by those rules

#### Scenario: Missing dimension
- **WHEN** a dimension is missing
- **THEN** its `reasonCodes` SHALL identify the missing source category
- **AND** its `evidence` SHALL be empty or contain only non-sensitive availability metadata

#### Scenario: Sensitive input supplied
- **WHEN** the request contains symptom text, medication names, or behavioral notes
- **THEN** those raw values SHALL NOT appear in breakdown evidence

### Requirement: Calculation metadata
Every wellness response SHALL identify the calculation version and evaluation time.

#### Scenario: Successful wellness response
- **WHEN** the service returns any wellness response status
- **THEN** `calculationVersion` SHALL contain the semantic version of the scoring contract
- **AND** `evaluatedAt` SHALL contain a timezone-aware UTC timestamp

### Requirement: Evaluation window
The service SHALL accept an optional inclusive `evaluationWindow` with `startDate` and `endDate` and SHALL echo a valid supplied window in the response.

#### Scenario: Valid window supplied
- **WHEN** `startDate` is on or before `endDate`
- **THEN** the request SHALL be accepted
- **AND** the response SHALL contain the same evaluation window

#### Scenario: Invalid window supplied
- **WHEN** `startDate` is after `endDate`
- **THEN** request validation SHALL fail with a client error

#### Scenario: Window omitted
- **WHEN** the request does not contain `evaluationWindow`
- **THEN** the response `evaluationWindow` SHALL be `null`

### Requirement: Partial-score normalization
The service SHALL continue to normalize earned points only over `AVAILABLE` dimensions when `scoreStatus` is `PARTIAL`.

#### Scenario: Eligible partial assessment
- **WHEN** Diet, Activity, and Baseline are available, weighted coverage is at least `0.60`, and other applicable dimensions are missing
- **THEN** the numeric wellness score SHALL be normalized over those available dimensions only
- **AND** `scoreStatus` SHALL be `PARTIAL`

#### Scenario: Feeding-only assessment
- **WHEN** diet is the only available applicable dimension
- **THEN** `wellnessScore`, `band`, and `bandLabel` SHALL be `null`
- **AND** `scoreStatus` SHALL be `INSUFFICIENT_DATA`
- **AND** `dataCoverage` SHALL reflect the diet weight relative to all applicable dimensions

### Requirement: Partial-assessment narrative
The service SHALL make partial assessment limitations visible in generated narrative context.

#### Scenario: Gemini narrative for partial score
- **WHEN** `scoreStatus` is `PARTIAL` and Gemini narrative generation is available
- **THEN** the prompt SHALL include the partial status, weighted coverage, and missing dimensions
- **AND** generated text SHALL NOT describe the result as a complete assessment

### Requirement: Actionable tracking recommendations
The service SHALL return deterministic `trackingRecommendations` for actionable missing data and positive tracking maintenance. Each recommendation SHALL contain a stable dimension value, concise user-facing text, machine-readable `requiredInputs`, and zero or more exact backend `ReminderType` wire values in `suggestedReminderTypes`.

#### Scenario: Partial assessment has actionable missing dimensions
- **WHEN** `scoreStatus` is `PARTIAL` and one or more actionable dimensions are `MISSING`
- **THEN** `trackingRecommendations` SHALL contain one deduplicated item for each actionable missing dimension
- **AND** items SHALL follow the wellness breakdown dimension order

#### Scenario: Insufficient assessment
- **WHEN** `scoreStatus` is `INSUFFICIENT_DATA`
- **THEN** `trackingRecommendations` SHALL describe the applicable data the user can record to obtain a wellness score

#### Scenario: Complete good assessment
- **WHEN** `scoreStatus` is `COMPLETE` and the band is `GOOD` or `EXCELLENT`
- **THEN** `trackingRecommendations` SHALL contain positive maintenance guidance for available Activity, Diet, and PreventiveCare dimensions with compatible reminder types
- **AND** text SHALL describe continuing successful tracking rather than missing data or a health problem

#### Scenario: Complete lower-band assessment
- **WHEN** `scoreStatus` is `COMPLETE` and the band is `FAIR`, `CONCERNING`, or `CRITICAL`
- **THEN** `trackingRecommendations` SHALL be empty
- **AND** health recommendations and problem-driven reminders SHALL remain the priority

#### Scenario: Maintenance reminder already represented
- **WHEN** a maintenance suggested reminder type is already present in `reminders`
- **THEN** that type SHALL be removed from the maintenance suggestion
- **AND** the maintenance item SHALL be omitted when no suggested type remains

#### Scenario: Inapplicable dimension
- **WHEN** a dimension is `NOT_APPLICABLE`
- **THEN** the service SHALL NOT return a tracking recommendation for that dimension

#### Scenario: No symptoms reported
- **WHEN** no symptom text is reported
- **THEN** the service SHALL NOT ask the user to record or invent symptoms to improve wellness coverage

#### Scenario: Recommendation source and safety
- **WHEN** tracking recommendations are produced
- **THEN** they SHALL come from a deterministic allowlisted catalog rather than Gemini
- **AND** their user-facing `text` SHALL NOT contain diagnoses, treatment guidance, medication guidance, or instructions that imply a reminder was created

#### Scenario: Required input paths
- **WHEN** a tracking recommendation is returned
- **THEN** every value in `requiredInputs` SHALL identify a valid field or collection in the wellness request contract

#### Scenario: Reminder-type mappings
- **WHEN** Activity tracking guidance is returned
- **THEN** `suggestedReminderTypes` SHALL equal `["Activity"]`
- **WHEN** Diet tracking guidance is returned
- **THEN** `suggestedReminderTypes` SHALL equal `["Feeding"]`
- **WHEN** PreventiveCare tracking guidance is returned
- **THEN** `suggestedReminderTypes` SHALL equal `["Vaccination", "VetVisit"]`

#### Scenario: No compatible backend reminder type
- **WHEN** missing-data Sleep or Baseline tracking guidance is returned
- **THEN** `suggestedReminderTypes` SHALL be empty
- **AND** Sleep and Baseline SHALL NOT produce complete-state maintenance guidance

#### Scenario: Suggestions require user confirmation
- **WHEN** any `suggestedReminderTypes` are returned
- **THEN** the service SHALL NOT create or schedule a reminder
- **AND** the response SHALL NOT invent repeat frequency, time, or notification settings
- **AND** the client or backend SHALL require user confirmation before reminder creation

#### Scenario: Reminder wire-value compatibility
- **WHEN** a suggested reminder type is returned
- **THEN** it SHALL be a value from the C# backend `ReminderType` contract
- **AND** `Medication` SHALL NOT be suggested for missing wellness-tracking data
