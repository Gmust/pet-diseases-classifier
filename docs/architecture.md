# Architecture

This document describes the deployed service as of 2026-07-07. Changes to a
boundary or safety/release decision must update this document and the matching
ADR in `docs/adr/` in the same pull request.

## Runtime request path

1. `app/api/` owns FastAPI transport: `deps.py` (authentication, services
   accessor), `errors.py` (stable HTTP errors, use-case error mapping), and one
   module per endpoint under `routes/`. `app/main.py` re-exports the app so
   `app.main:app` stays the entry point.
2. Each domain package owns its own request/response contract:
   `app/wellness/schemas.py` and `responses.py`, `app/triage/schemas.py`,
   `app/feeding/schemas.py`. Vocabulary shared across domains
   (urgency, specialist, disease category, species) lives in `app/domain/enums.py`.
3. Domain entry points orchestrate behavior without depending on FastAPI:
   `app/triage/predict.py` and `chat.py`, `app/wellness/entrypoint.py`,
   `app/feeding/summary.py`.
4. `app/triage/safety.py` applies deterministic emergency policy before any
   generated response. `app/triage/advice.py` validates generated/static advice.
5. Classifiers implement `app/inference/protocols.py`; Torch and ONNX adapters
   live in `app/inference/predictor.py` and `onnx_predictor.py`. Gemini
   infrastructure lives under `app/llm/`, and wellness scoring is split across
   `app/wellness/scoring/`, `norms.py`, `reminders.py`, `tracking.py`, and
   `narrative.py`, sequenced by `app/wellness/service.py`.
6. `app/observability.py` emits structured logs without symptom or prompt text.

Dependencies point one way: `app/api/` -> domain packages (`wellness`, `triage`,
`feeding`) -> shared packages (`inference`, `llm`, `domain`). Domain packages do
not import each other. The training pipeline lives outside `app/` in
`ml_pipeline/`, and nothing under `app/` may import it — both Lambda images copy
`app/` only. `tests/test_domain_boundaries.py` enforces all of this.

`app/lambda_handler.py` is the AWS entry point; Uvicorn imports `app.main:app`
for the server image. `app/app_services.py` is the composition container shared
by both entry points, built by `app/bootstrap.py` at Lambda INIT.

## Safety boundary

Probabilistic classifier and generator output cannot override deterministic
red-flag escalation, abstention, public-error, or advice-validation policy.
Emergency responses bypass Gemini. Provider failures return reviewed bounded
fallbacks. The service is a pre-assessment aid, not a diagnostic system.

## ML lifecycle

`dataset_schema.py`, source adapters, normalization/quality gates, and
`split_manifest.py` produce traceable leakage-resistant inputs. `train.py`
consumes validated configuration and emits metrics, run metadata, calibration,
and a model card. `evaluate.py` and `release_gates.py` produce model, safety,
and Torch/ONNX parity evidence.

`model_registry.py` publishes write-once version directories with hashes, sizes,
retention intent, and rollback metadata. Images verify `release-manifest.json`
during build; adapters verify it again at startup. Readiness exposes only backend,
version, and label count.

## Deployment boundary

`.github/workflows/ci.yml` handles fast/data evidence. `model-release.yml`
retrieves a run-ID-addressed immutable model artifact and gates containers and
promotion evidence. `template.yaml` deploys the ONNX Lambda image behind API
Gateway using Secrets Manager values, bounded concurrency, tracing, retained
logs, alarms, and canary rollback.

## Dependency direction

Transport depends on use cases; use cases depend on structural classifier/
generator contracts and deterministic policy; infrastructure adapters implement
those contracts. Offline ML tooling may depend on runtime model adapters, but
runtime request modules must not import training/data-pipeline modules.
