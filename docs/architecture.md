# Architecture

This document describes the deployed service as of 2026-07-07. Changes to a
boundary or safety/release decision must update this document and the matching
ADR in `docs/adr/` in the same pull request.

## Runtime request path

1. `app/main.py` owns FastAPI transport, authentication, stable HTTP errors,
   liveness/readiness, and dependency construction.
2. `app/schemas.py` validates the public camelCase request/response contract.
3. `app/use_cases/` orchestrates predict, chat, and wellness behavior without
   depending on FastAPI.
4. `app/services/triage_safety.py` applies deterministic emergency policy before
   any generated response. `advice_safety.py` validates generated/static advice.
5. Classifiers implement `app/ml/protocols.py`; Torch and ONNX adapters live in
   `predictor.py` and `onnx_predictor.py`. Gemini and wellness infrastructure
   adapters live under `app/services/`.
6. `app/observability.py` emits structured logs without symptom or prompt text.

`app/lambda_handler.py` is the AWS entry point; Uvicorn imports `app.main:app`
for the server image. `app/app_services.py` is the composition container shared
by both entry points.

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
