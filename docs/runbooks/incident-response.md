# Incident Response Runbook

1. Declare severity/owner/time, preserve request ids and immutable model/image
   versions, and avoid copying symptom text into tickets or chat.
2. Contain: rotate Secrets Manager keys for auth compromise, enable static
   explanations for provider failure, or follow the immutable rollback runbook
   for model/safety regression.
3. Measure Lambda Errors/Throttles/duration, readiness version, fallback reasons,
   abstentions, and red-flag rates. Correlate by request id only.
4. Validate recovery with readiness plus `scripts/smoke_test.sh`; monitor through
   one full canary/traffic window.
5. Document scope, timeline, root cause, affected versions, data exposure,
   corrective tests, and owners. Update runbooks/ADRs and notify affected parties
   according to legal/privacy obligations.
