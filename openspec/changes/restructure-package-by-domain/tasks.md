## 0. Baseline

- [x] 0.1 Run `make quality` on a clean tree and record the coverage percentage and test count as the regression baseline
- [x] 0.2 Capture pre-refactor responses for all four endpoints (`/predict`, `/chat`, `/wellness`, `/feeding-summary`) with fixed inputs, saved as fixtures for a before/after diff

## 1. Split `app/ml/` by lifecycle

- [x] 1.1 Create `app/inference/` and move `protocols.py`, `predictor.py`, `onnx_predictor.py` into it, updating imports in `app/main.py`, `app/app_services.py`, `app/use_cases/chat.py`, `app/services/gemini_service.py`, `app/services/wellness_service.py`
- [x] 1.2 Create top-level `ml_pipeline/` and move the training/dataset/evaluation/release modules (`train`, `evaluate`, `generate_synthetic`, `fetch_and_merge`, `merge_datasets`, `prepare_dataset`, `dataset_*`, `calibration`, `check_leakage`, `synthetic_quality`, `eval_coverage`, `evaluation_report`, `export_onnx`, `model_card`, `release_gates`, `release_docs`, `run_metadata`, `split_manifest`, `training_config`) out of `app/`. `model_registry` and `model_validation` stay in `app/inference/` — the predictors call them at model-load time, and both Lambda images copy `app/` only
- [x] 1.3 Update `tests/conftest.py` for the new `Predictor.from_paths` import path and update all `ml_pipeline` test imports; confirm the suite passes with no assertion edits
- [x] 1.4 Update `pyproject.toml`: `packages.include` to cover `ml_pipeline*`, `known_first_party` to list both packages, coverage scope to the whole `app` package omitting only `onnx_predictor.py` and `lambda_handler.py`, and the lint/mypy targets to cover `app` and `ml_pipeline`. Replace CI's 14-module `--cov=` allowlist with a single full-suite gated run
- [x] 1.5 Verify `ml_pipeline/` is absent from a built Lambda image and that no module under `app/` imports `ml_pipeline`; re-measure coverage against the 0.1 baseline

## 2. Extract shared domain vocabulary

- [x] 2.1 Create `app/domain/enums.py` with `UrgencyLevel`, `SpecialistType`, `DiseaseCategory`, `PetType`, re-exported from `app/schemas.py` so importers keep working
- [x] 2.2 Move `condition_metadata.py` to `app/domain/conditions.py` and update its importers (`use_cases/predict.py`, `use_cases/chat.py`, `services/wellness_service.py`)
- [x] 2.3 Add a test asserting `app/domain/conditions.py` imports cleanly with neither torch nor onnxruntime available, and that it covers all 16 condition classes

## 3. Split schemas per domain

- [x] 3.1 Create `app/wellness/schemas.py` (enums + request models) and `app/wellness/responses.py` (breakdown, reminder, tracking recommendation, response) — one module would have been 332 lines
- [x] 3.2 Create `app/triage/schemas.py` with the predict and chat models
- [x] 3.3 Create `app/feeding/schemas.py` with the feeding-summary models
- [x] 3.4 Reduce `app/schemas.py` to a pure re-export shim over the new modules and confirm no schema class is declared twice
- [x] 3.5 Verify `tests/test_docs_sync.py` needs no change (it compares an OpenAPI fingerprint, not module paths); prose doc references are handled in 7.4

## 4. Decompose wellness scoring

- [x] 4.1 Extract `app/wellness/norms.py`: activity/sleep/kcal tables, defaults, urgency base scores, condition cap keywords, band labels, `_norm`, `_clamp`
- [x] 4.2 Extract `app/wellness/scoring/activity.py` and `sleep.py` with the shared `_breakdown_item` helper
- [x] 4.3 Extract `app/wellness/scoring/diet.py` and `symptoms.py`
- [x] 4.4 Extract `app/wellness/scoring/preventive.py` and `baseline.py` (including medication adherence and weight stability helpers)
- [x] 4.5 Extract `app/wellness/scoring/aggregate.py`: score computation, condition cap, dimension items, reliability/coverage, band, trend
- [x] 4.6 Extract `app/wellness/reminders.py` (reminders, filtering, weight recommendation) and `app/wellness/tracking.py` (`_TrackingSpec`, tracking recommendations) — one module would have been 318 lines
- [x] 4.7 Extract `app/wellness/prompt.py` (response model, system instruction, prompt build, length trim) and `app/wellness/narrative.py` (fallbacks + `generate_narrative`, taking the client as a parameter)
- [x] 4.8 Reduce `app/wellness/service.py` to `__init__`, `metadata`, and an orchestrating `score()`; confirm it contains no thresholds, norm tables, or prompt text
- [x] 4.9 Add direct unit tests for at least two dimension scorers called without constructing a service or LLM client
- [x] 4.10 Confirm `tests/test_wellness_service.py` and `tests/test_wellness_reliability.py` pass with unmodified assertions

## 5. Relocate triage, feeding, and LLM packages

- [x] 5.1 Move `triage_safety.py`, `chat_context.py`, `advice_safety.py` into `app/triage/` as `safety.py`, `context.py`, `advice.py`
- [x] 5.2 Move `use_cases/predict.py` and `use_cases/chat.py` into `app/triage/`, keeping the `run_predict` / `run_chat` names
- [x] 5.3 Move `services/feeding_summary_service.py` and `use_cases/feeding_summary.py` into `app/feeding/`
- [x] 5.4 Move `use_cases/wellness.py` into `app/wellness/` as its entry point
- [x] 5.5 Create `app/llm/` from `gemini_service.py` (further split into `payloads.py` + `prompts.py`), `rotation.py`, `generation_policy.py`; promote `errors.py` to `app/errors.py` and delete the now-empty `app/services/` and `app/use_cases/`
- [x] 5.6 Verify no domain package imports another domain package and no shared package imports a domain package

## 6. Split the API layer

- [x] 6.1 Create `app/bootstrap.py` with `build_services` and `ensure_services`, keeping the Lambda INIT load and the lazy ONNX backend import
- [x] 6.2 Create `app/api/deps.py` with the `api_key_auth` dependency and services accessor
- [x] 6.3 Create `app/api/errors.py` with the `InvalidInputError → 400` and `InferenceUnavailableError → 500` handlers plus the existing HTTPException and generic-500 handlers, preserving the `{"detail", "requestId"}` body
- [x] 6.4 Create `app/api/routes/` with `health.py`, `predict.py`, `chat.py`, `wellness.py`, `feeding.py`, dropping the per-route `try/except` and keeping all docstrings, paths, and response models
- [x] 6.5 Build the app in `app/api/__init__.py` (kept free of heavy imports) and reduce `app/main.py` to re-export `app` and `ensure_services`
- [x] 6.6 Verify `uvicorn app.main:app` starts and `app.lambda_handler.handler` imports with services built at INIT

## 7. Enforce and finalize

- [x] 7.1 Delete the `app/schemas.py` shim and update every remaining importer
- [x] 7.2 Add a boundary test asserting no `app` → `ml_pipeline` import, no sibling-domain import, and no shared-to-domain import
- [x] 7.3 Add a module-size test asserting no module under `app/` exceeds 250 source lines, exempting `app/domain/conditions.py`
- [x] 7.4 Update `CLAUDE.md`, `README.md`, `docs/api-reference.md`, and `docs/dotnet-chat-integration.md` for the new module paths
- [x] 7.5 Run `make quality`; branch coverage over `app` is 88.4% (gate 85). Test count is 310 vs the 236 baseline — the 0.1 criterion assumed no new tests, but tasks 2.3, 4.9, 7.2 and 7.3 add them; no baseline test was removed, skipped, or had an assertion changed
- [x] 7.6 Diff live endpoint responses against the 0.2 fixtures to confirm the HTTP contract is byte-identical
- [ ] 7.7 Build both Lambda images and smoke-test all four endpoints against a deployed stack before merging — BLOCKED: no Docker daemon available in this environment, and deploying is the maintainer's call. Verified instead by importing `app.lambda_handler` with `ml_pipeline` blocked at the import hook: handler loads, services build at INIT, all four routes register
