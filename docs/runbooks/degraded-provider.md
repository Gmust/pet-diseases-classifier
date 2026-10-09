# Degraded Provider Runbook

Gemini timeout, quota exhaustion, malformed output, or missing configuration must
leave deterministic triage available.

1. Confirm `generation_fallback` events and reason counts without inspecting raw
   prompts. Check provider status and quota separately.
2. If errors are sustained, set `USE_STATIC_EXPLANATIONS=true` and deploy the
   exact reviewed image digest. Emergency output already bypasses Gemini.
3. Smoke-test predict/chat/wellness responses and confirm valid bounded fallbacks,
   stable status codes, and normal classifier readiness.
4. Restore generation only after provider calls meet timeout/error thresholds;
   deploy the reviewed image and monitor fallback counts through the full
   post-deployment observation window (`docs/runbooks/release.md`, step 5).
