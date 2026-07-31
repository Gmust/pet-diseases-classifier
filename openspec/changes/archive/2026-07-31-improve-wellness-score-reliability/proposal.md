## Why

The wellness endpoint can currently return a confident-looking score from very little data, including 100 from feeding data alone or zero when no dimension is available. The API needs a minimum reliability gate plus explicit coverage, status, evidence, and version metadata so the backend and UI can distinguish a complete assessment from a reliable partial assessment or an unavailable one and explain every rule-based result.

## What Changes

- Add a weighted `dataCoverage` value describing how much applicable wellness data was evaluated.
- Add `scoreStatus` with `COMPLETE`, `PARTIAL`, and `INSUFFICIENT_DATA` states.
- Add `calculationVersion`, `evaluatedAt`, and an optional echoed `evaluationWindow`.
- Add dimension availability, stable reason codes, and structured evidence to every breakdown item.
- Add deterministic `trackingRecommendations` that tell users which missing data to record and, for complete GOOD/EXCELLENT assessments, how to maintain useful tracking habits.
- Add optional `suggestedReminderTypes` metadata to tracking recommendations so clients can offer user-confirmed reminder creation with exact backend enum values.
- Require at least 60% weighted coverage plus Diet, Activity-or-Sleep, and PreventiveCare-or-Baseline before exposing a numeric score.
- Preserve proportional scoring for partial data while making incompleteness explicit.
- **BREAKING**: return `null` for `wellnessScore`, `band`, and `bandLabel` when the minimum reliability gate is not met instead of reporting a misleading score from sparse data.
- Document the response contract, coverage calculation, status rules, reason codes, and backward-incompatible behavior.

## Capabilities

### New Capabilities

- `wellness-score-reliability`: Defines coverage, assessment status, version metadata, evaluation windows, dimension availability, evidence, actionable tracking recommendations, and insufficient-data behavior for `POST /wellness`.

### Modified Capabilities

None. This repository does not yet contain baseline OpenSpec capability specifications.

## Impact

- API contracts in `app/schemas.py`, including nullable insufficient-data fields, reliability metadata, and tracking-recommendation models.
- Rule evaluation and response assembly in `app/services/wellness_service.py`.
- `POST /wellness` documentation and examples in `README.md`.
- Wellness unit and serialization tests under `tests/`.
- The C# backend and mobile client must handle the new metadata and nullable score fields before deploying this breaking response change.
- No new runtime dependency, persistence layer, model retraining, or Gemini behavior is required.
