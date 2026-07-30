# ADR 0003: Keep chat stateless and provider-optional

- Status: Accepted
- Date: 2026-07-07

## Decision

Clients send bounded history and symptom summary on every chat request. This
service stores no conversation state. Gemini is a replaceable, timeout-bounded
presentation provider; reviewed local fallbacks remain valid behavior.

## Consequences

Scaling and privacy boundaries are simpler, but clients own conversation
continuity and payload limits. Provider degradation does not make deterministic
triage unavailable.
