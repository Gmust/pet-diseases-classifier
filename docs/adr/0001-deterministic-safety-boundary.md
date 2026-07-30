# ADR 0001: Keep safety policy deterministic

- Status: Accepted
- Date: 2026-07-07

## Decision

Emergency detection, escalation, abstention, and advice validation execute as
deterministic application policy around the classifier and generator. Emergency
responses never call the external generator.

## Consequences

Safety fixtures remain independently testable and available during provider
failure. Policy changes require reviewed fixtures and veterinary ownership.
Generated prose may improve presentation but cannot lower urgency.
