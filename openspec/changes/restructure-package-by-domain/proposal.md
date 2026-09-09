## Why

The service layout has drifted into a few oversized, multi-responsibility modules: `app/services/wellness_service.py` is 1332 lines covering species norm tables, six dimension scorers, the reliability gate, reminders/tracking recommendations and Gemini narrative generation; `app/schemas.py` is 623 lines holding four unrelated request/response domains. Separately, `app/ml/` mixes the four modules needed to serve requests with ~16 training, dataset and release-gate modules (~4.3k lines), so the training pipeline ships inside the Lambda image and is importable from serving code.

The layering itself (`routes → use_cases → services → ml`) is sound. This change keeps that direction and fixes module size and package boundaries before more endpoints are added.

## What Changes

- Split `app/ml/` by lifecycle: runtime inference (`protocols`, `predictor`, `onnx_predictor`) moves to `app/inference/`; training/dataset/evaluation/release modules move out of `app/` to a top-level `ml_pipeline/` package. Lambda images copy `app/` only.
- Move `app/ml/condition_metadata.py` to `app/domain/conditions.py` — it is business data (urgency, specialist, disease category, home advice per condition), not ML code, and is consumed by `services/` and `use_cases/`.
- Reorganize the request-handling code package-by-domain: `app/wellness/`, `app/triage/` (predict + chat), `app/feeding/`, each owning its own schemas, and shared `app/llm/` for the Gemini client, key rotation and generation policy.
- Decompose `wellness_service.py` into `norms.py`, `scoring/` (one module per dimension plus `aggregate.py`), `reminders.py`, `narrative.py`, and a thin `service.py` orchestrator. Target: no module over ~250 lines.
- Split `app/schemas.py` into per-domain schema modules, with `app/schemas.py` kept temporarily as a re-export shim so tests and docs keep importing successfully during migration, then removed.
- Split `app/main.py` into an `app/api/` package (app factory, `deps.py`, `errors.py`, `routes/`) plus `app/bootstrap.py` for `build_services`/`ensure_services`. Route-level `try/except` for `InvalidInputError`/`InferenceUnavailableError` is replaced by shared exception handlers. `app.main:app` stays importable as an alias for SAM, uvicorn and tests.
- No change to HTTP behavior: paths, request/response shapes, status codes, auth, and the classifier/Gemini separation of duties are all preserved.

## Capabilities

### New Capabilities
- `service-module-boundaries`: The internal package structure contract — which packages exist, the allowed dependency direction between them, the separation of runtime-serving code from training-pipeline code, and the module-size limit that keeps this change from regressing.

### Modified Capabilities
<!-- None. This is a structural refactor; no endpoint requirement changes. `wellness-score-reliability` behavior is preserved exactly and is the regression baseline. -->

## Impact

- **Code**: every module under `app/` moves or is split; `app/ml/` ceases to exist. Roughly 8.9k lines of `app/` code relocated, no logic rewritten.
- **Tests**: `tests/conftest.py` monkeypatches `Predictor.from_paths` by import path and resets `app.state.services`; `test_docs_sync.py` and `test_classifier_regression.py` reference module/doc paths. All break on move and must be updated in the same commit as each move.
- **Build/deploy**: `Dockerfile.lambda`, `Dockerfile.lambda.onnx`, `template.yaml` handler path, and `app/lambda_handler.py` INIT import all reference moved modules.
- **Tooling**: `pyproject.toml` coverage `omit` (currently `app/ml/train.py`) and the `fail_under = 85` branch-coverage gate shift when `ml_pipeline/` leaves the measured package; mypy target (`mypy app`) must gain `ml_pipeline`; `uv` extras (`train`, `export`) now map to the new top-level package.
- **Docs**: `CLAUDE.md`, `README.md`, `docs/api-reference.md` and `docs/dotnet-chat-integration.md` cite module paths that change.
- **External API consumers** (the .NET backend): unaffected.
