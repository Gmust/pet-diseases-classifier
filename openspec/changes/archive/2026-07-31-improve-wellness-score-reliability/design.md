## Context

`POST /wellness` evaluates six weighted dimensions and normalizes the score over dimensions marked `included`. This prevents missing inputs from lowering a partial score, but the response does not tell consumers how much data was evaluated. When every dimension is unavailable, the current implementation returns `0`, `CRITICAL`, and a critical narrative even though no negative health evidence exists.

The .NET backend aggregates pet data and calls this stateless service. The mobile client presents the result. Both consumers need a stable, machine-readable distinction between a complete score, a partial score, and no score, plus evidence that explains the deterministic rule results.

## Goals / Non-Goals

**Goals:**

- Quantify weighted coverage of applicable wellness dimensions.
- Distinguish available, missing, and species-inapplicable dimensions.
- Make insufficient data different from a genuine score of zero.
- Expose stable reason codes and privacy-conscious evidence for rule results.
- Give users deterministic, actionable guidance for collecting missing wellness data.
- Version the calculation and identify when and over what window it ran.
- Preserve proportional normalization when the minimum reliable-score gate is met.

**Non-Goals:**

- Change dimension weights, dimension scoring thresholds, condition caps, reminder rules, or medication-safety rules.
- Add historical anomaly detection, predictive alerts, new data sources, or persistence.
- Retrain the classifier or allow Gemini to calculate scores.
- Move reminder scheduling or wellness-history storage into this service.

## Decisions

### Use explicit dimension availability

Each breakdown item will add `availability` with `AVAILABLE`, `MISSING`, or `NOT_APPLICABLE`.

- `AVAILABLE` means the rule had enough source data to evaluate the dimension.
- `MISSING` means the dimension applies to the pet but required source data was absent or evaluation failed.
- `NOT_APPLICABLE` means the rule does not apply to that species, such as fish sleep in the current formula.

The existing `included` field remains for compatibility and is derived as `availability == AVAILABLE`.

Alternative considered: infer all states from `included`. This was rejected because missing data and species-inapplicable data must affect the coverage denominator differently.

### Calculate weighted coverage from dimension maxima

`dataCoverage` will be:

`sum(maxScore for AVAILABLE dimensions) / sum(maxScore for AVAILABLE or MISSING dimensions)`

`NOT_APPLICABLE` dimensions are excluded from both sides. The value is rounded to four decimal places and constrained to `0..1`.

This preserves the formula's relative dimension importance. A simple count of dimensions was rejected because a missing 25-point symptom assessment is more material than a missing 10-point baseline dimension.

When no symptom text is reported, the symptoms dimension is `NOT_APPLICABLE`, not `MISSING`. If symptom text is supplied but the classifier cannot evaluate it, the dimension is `MISSING`. This prevents healthy users from being told to invent or report symptoms merely to complete a wellness score.

### Gate numeric scores on minimum reliability

A numeric wellness score is eligible only when both conditions are true:

- `dataCoverage` is at least `0.60`.
- At least three foundational dimensions are `AVAILABLE` among Activity, Sleep, Diet, Preventive care, and Baseline.
- Diet is `AVAILABLE`, at least one general-health-context dimension (Preventive care or Baseline) is `AVAILABLE`, and at least one daily-behavior dimension (Activity or Sleep) is `AVAILABLE` when either behavior dimension applies to the species.

The applicability exception prevents species such as fish, whose current Activity and Sleep rules are both `NOT_APPLICABLE`, from being permanently ineligible. Such species still need three available foundational dimensions. Symptoms remain optional and never need to be invented merely to unlock a score.

- `COMPLETE`: the reliability gate is met and every applicable dimension is `AVAILABLE`.
- `PARTIAL`: the reliability gate is met and at least one applicable dimension is `MISSING`.
- `INSUFFICIENT_DATA`: the reliability gate is not met, including when some dimensions are available.

Coverage alone was rejected because three similarly weighted but narrowly related inputs can omit nutrition or general health context. Dimension count alone was rejected because it ignores the formula weights. Combining a 60% weighted threshold with the three foundational groups prevents a feeding-only result from looking like a reliable 100 while retaining useful partial assessments.

### Represent no assessment with null, not zero

For `INSUFFICIENT_DATA`, including sparse but evaluable requests, `wellnessScore`, `band`, and `bandLabel` will be `null`. The endpoint will return a deterministic insufficient-data narrative, skip Gemini generation, and produce no score-derived reminders.

This is intentionally breaking. Returning zero was rejected because it conflates missing evidence with the worst possible wellness result. Returning a normalized score from sub-threshold data was rejected because values such as a feeding-only 100 remain misleading even when marked partial.

### Return stable reason codes and bounded evidence

Each dimension will return at least one uppercase `reasonCode` explaining availability or the principal scoring outcome. `evidence` will contain only JSON-safe numeric, boolean, date, or enum-like scalar values used by the rule.

Raw symptom text, medication names, free-form behavioral notes, and generated prose will not be copied into evidence. This keeps evidence useful for UI explanations without unnecessarily duplicating sensitive or token-heavy input.

### Generate deterministic tracking recommendations

The response will include `trackingRecommendations`, separate from health `recommendations` and care `reminders`. Each item will contain:

- `dimension`: a stable wellness-dimension enum value.
- `text`: concise user-facing guidance describing what to track.
- `requiredInputs`: machine-readable request-field paths the backend can use to deep-link or build a checklist.
- `suggestedReminderTypes`: zero or more exact backend `ReminderType` wire values that a client may offer when asking the user whether to create a reminder.

Missing-data recommendations are generated from an allowlisted mapping of actionable missing-input reason codes. They are returned for `PARTIAL` and `INSUFFICIENT_DATA`, omitted for `AVAILABLE` and `NOT_APPLICABLE` dimensions, deduplicated, and ordered by the existing dimension order.

For a `COMPLETE` assessment in the `GOOD` or `EXCELLENT` band, the same field returns positive maintenance guidance for available Activity, Diet, and Preventive care dimensions that have backend-compatible reminder types. Maintenance text says to keep tracking rather than implying missing or unhealthy data. Sleep and Baseline are omitted from maintenance guidance because no compatible reminder type exists. `FAIR`, `CONCERNING`, and `CRITICAL` complete assessments do not receive maintenance guidance so it cannot compete with health recommendations and problem-driven reminders.

Any maintenance `suggestedReminderTypes` already present in the response's problem-driven `reminders` are removed. A maintenance item is omitted if no suggested type remains. This prevents the client from showing duplicate reminder calls to action.

Initial mappings cover:

- Activity: daily steps or active minutes; suggests `Activity`.
- Sleep: daily sleep duration; suggests no reminder because the backend has no Sleep reminder type.
- Diet: meal frequency and feeding consistency; suggests `Feeding`.
- Preventive care: vaccination status and recent veterinary-visit status; suggests `Vaccination` and `VetVisit`.
- Baseline: age and at least two weight-history measurements; suggests no reminder because the backend has no Weight or Baseline reminder type.

The microservice only supplies suggestions. It does not create reminders, choose repeat schedules, or imply user consent. The backend or client must present a confirmation step and collect any required scheduling data. A list is used because one missing dimension can map to multiple valid reminder types, while an empty list is explicit when the C# enum has no semantic match.

No tracking recommendation is generated for absent symptom text because absence means `NOT_APPLICABLE`. Evaluation failures do not produce data-collection advice when the required input was already supplied.

Gemini will not create, rewrite, or expand these messages. Deterministic generation was chosen to keep the contract testable, prevent duplication, avoid medical suggestions, and reduce output-token usage.

### Echo an optional evaluation window

The request will accept an optional `evaluationWindow` containing `startDate` and `endDate`, inclusive. The service validates `startDate <= endDate` and echoes the window in the response. It will not infer a window from mixed inputs because activity, feeding, medication, and weight data may cover different periods.

### Use a semantic calculation version and UTC evaluation time

The response will include a constant `calculationVersion` beginning at `2.0.0` and an `evaluatedAt` UTC timestamp. Changes that can alter scores, coverage, availability, or reason-code semantics require a version increment.

API deployment version and model version remain separate concerns.

## Risks / Trade-offs

- [Breaking nullable score fields] → Coordinate deployment so the .NET backend and mobile client accept null before the microservice response changes.
- [Most current requests may be `PARTIAL`] → Treat this as truthful coverage, document missing inputs, and let the UI display a partial-state indicator.
- [Reason-code proliferation] → Centralize codes as enums/constants and cover every code with contract tests.
- [Evidence contract can accidentally expose input text] → Restrict evidence values to an allowlisted scalar schema and add tests proving medication names and raw symptoms are absent.
- [Coverage changes when applicability rules evolve] → Couple applicability changes to `calculationVersion`.
- [Client uses `included` instead of availability] → Keep `included` synchronized and document it as a compatibility field.
- [Tracking guidance becomes stale when request fields change] → Keep required-input paths in one typed mapping and test every path against the request schema.
- [Tracking suggestions duplicate reminders or health advice] → Use a separate response field and prohibit reminder actions, diagnosis, treatment, and medication language in the tracking-message catalog.
- [Positive maintenance guidance distracts from a concerning result] → Return it only for complete GOOD/EXCELLENT assessments and suppress types already covered by problem-driven reminders.
- [Client silently creates suggested reminders] → Name the field as suggestions, return no scheduling data, document explicit user confirmation, and test that the microservice has no reminder-creation side effect.

## Migration Plan

1. Add schema types and tests while preserving the existing response behind the current deployment.
2. Update the .NET client DTOs and mobile handling for new fields and nullable score/band values.
3. Deploy the microservice and verify complete, partial, and insufficient-data contract fixtures.
4. Monitor validation errors and client parsing failures.
5. Roll back to the previous microservice image if consumers cannot handle the contract; no data migration is required because the service is stateless.

## Open Questions

- Should the public API keep `included` indefinitely or deprecate it after all consumers adopt `availability`?
- Should the backend require `evaluationWindow` for production calls even though the microservice keeps it optional for backward compatibility?
