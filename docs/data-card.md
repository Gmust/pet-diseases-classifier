# Data Card

Generated from `configs/release-documentation.json`; do not edit by hand.

## Intended use

Training and evaluation of a pet-owner-language condition-category pre-assessment classifier; not epidemiology or diagnosis.

## Sources and licenses

- repository-local curated records: repository-controlled; verify upstream rights before release (tracked by dataset manifest)
- configured external adapters: recorded per source manifest (pinned by adapter configuration)

## Coverage

- Classes: All labels must have owner-language evaluation support before promotion; enforced by eval coverage and release gates.
- Species/register: Owner-language dog/cat symptom descriptions; unsupported species and clinical shorthand are not release claims.
- Synthetic contribution: Recorded by each immutable dataset manifest; synthetic rows require leakage, similarity, provenance, and manual-review gates.

## Transformations and exclusions

Text/labels are normalized centrally, content-deduplicated, schema-validated, and grouped before splitting. Invalid, duplicate, leakage-prone, and diagnosis-leaking rows are rejected with counts.

## Known gaps

No claim of population representativeness, clinical outcome validity, rare-species coverage, or multilingual coverage.
