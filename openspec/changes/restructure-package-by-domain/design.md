## Context

The service is a FastAPI app with four endpoints (`/predict`, `/chat`, `/wellness`, `/feeding-summary`) running locally under uvicorn and on Lambda via Mangum. Layering is already correct — routes delegate to `app/use_cases/`, which call `app/services/`, which call `app/ml/`. Nothing about that direction is being changed.

What has drifted is module granularity and package membership:

- `app/services/wellness_service.py` — 1332 lines. Contains species activity/sleep/kcal tables, six dimension scorers, condition caps, score aggregation, the reliability/coverage gate, reminder generation, tracking recommendations, narrative prompt construction, fallback narrative, and the `WellnessService` class that owns a Gemini client. Seven distinct reasons to change, one file.
- `app/schemas.py` — 623 lines spanning four unrelated request/response domains plus shared enums. Wellness accounts for roughly 70%.
- `app/ml/` — 20 modules, ~4.3k lines, of which only `protocols.py`, `predictor.py`, `onnx_predictor.py`, and `condition_metadata.py` are touched while serving a request. The other 16 are dataset generation, merging, preparation, training, evaluation, calibration, ONNX export, model registry and release gates.
- `app/main.py` — 241 lines mixing the app factory, auth dependency, service construction, exception handlers, and all four routes. Each route repeats the same `InvalidInputError → 400 / InferenceUnavailableError → 500` block.

Relevant constraints:

- `tests/conftest.py` monkeypatches `Predictor.from_paths` by import path, pops `GEMINI_API_KEY`/`API_KEY` before app import, and resets `app.state.services` per test. Any move of the predictor or the app object touches it.
- Coverage gate is `fail_under = 85` branch coverage with `source = ["app"]` and `omit = ["app/ml/train.py"]`.
- Both Lambda Dockerfiles already `COPY app/ ./app/` and nothing else, and both use `CMD ["app.lambda_handler.handler"]`.
- `app/lambda_handler.py` imports `app` and `ensure_services` from `app.main` and calls `ensure_services()` at module import so the model loads at Lambda INIT.

## Goals / Non-Goals

**Goals:**

- Split `app/` package-by-domain so each package owns its schemas, rules, and entry point.
- Get every module under `app/` to ~250 lines or less (excluding the static condition data map).
- Remove the training pipeline from the serving package and from the deployed image.
- Make wellness dimension scorers pure functions testable without a service or an LLM client.
- Centralize use-case-error → HTTP mapping so routes stop repeating it.
- Preserve the HTTP contract, the classifier-is-sole-decider invariant, the red-flag-before-Gemini ordering, and the Lambda INIT load behavior.

**Non-Goals:**

- No endpoint, schema, status code, or scoring-rule change. The existing test suite is the contract; assertions must not be edited to make a move pass.
- No new runtime dependency, no dependency-injection framework, no async rewrite of the sync route handlers.
- No change to the model, training data, or release-gate logic — those modules move verbatim.
- No rewrite of `use_cases/`-style orchestration into a different pattern; the layer is renamed/relocated into domain packages, not redesigned.

## Decisions

### Package-by-domain over layer-by-type

Target layout:

```
app/
  api/            app factory, deps.py (auth, get_services), errors.py, routes/
  bootstrap.py    build_services / ensure_services
  domain/         enums.py, conditions.py  (shared business vocabulary)
  inference/      protocols.py, predictor.py, onnx_predictor.py
  llm/            gemini_client.py, rotation.py, generation_policy.py
  wellness/       schemas.py, norms.py, scoring/, reminders.py, narrative.py, service.py, entrypoint
  triage/         schemas.py, safety.py, context.py, advice.py, predict.py, chat.py
  feeding/        schemas.py, service.py
  config.py, observability.py, app_services.py
ml_pipeline/      (outside app/) training, dataset, evaluation, release modules
```

Rationale: the current split by type means a wellness change touches `schemas.py`, `use_cases/wellness.py`, and `services/wellness_service.py` in three different trees, while unrelated domains sit in the same files. Grouping by domain makes the blast radius of a change one directory and makes the module-size limit natural rather than arbitrary.

Alternative considered — keep `services/`/`use_cases/` and only split the big files. Rejected: it fixes file size but leaves `schemas.py` as a shared choke point and leaves the training pipeline inside `app/`.

Alternative considered — full hexagonal/ports-and-adapters with explicit adapter interfaces per external system. Rejected as over-engineering for four endpoints and one external API; `Classifier`/`GeneratorMetadata` protocols already give the seam that matters.

### Training pipeline leaves `app/` as a sibling top-level package

`ml_pipeline/` at repo root, not `app/ml_pipeline/` and not a separate distribution.

Rationale: the Dockerfiles copy `app/` only, so the move alone removes the training code from the image with no Dockerfile edit. A sibling package also makes the "serving must not import training" rule mechanically checkable rather than a convention. Publishing it as a separate installable package was rejected — it is used by local scripts and CI only, and versioning it separately buys nothing today.

`packages.include = ["app*"]` in `pyproject.toml` must grow `ml_pipeline*`, and `known_first_party` must list both.

### `condition_metadata` becomes `app/domain/conditions.py`

It is a static map of 16 condition classes to urgency, specialist, disease category and home advice — consumed by `use_cases/predict.py`, `use_cases/chat.py`, and `wellness_service.py`, never by the model. Keeping it under `ml/` implied a dependency on the ML package that does not exist. It also has to stay importable with neither torch nor onnxruntime present, which the spec asserts.

It is exempt from the 250-line limit: it is ~284 lines of data, and splitting a lookup table across files to satisfy a logic-oriented limit would hurt readability.

### Wellness decomposition boundaries

- `norms.py` — `_ACTIVITY_TARGETS`, `_SLEEP_NORMS`, `_KCAL_PER_KG`, defaults, `_URGENCY_BASE_SCORE`, `_CONDITION_CAP_KEYWORDS`, `_BAND_LABELS`, and the `_norm`/`_clamp` helpers. Pure data plus two trivial functions.
- `scoring/activity.py|sleep.py|diet.py|symptoms.py|preventive.py|baseline.py` — one `_score_*` function each, each returning breakdown items. Signature stays as-is; each imports from `norms.py` and the shared `_breakdown_item` helper.
- `scoring/aggregate.py` — `_compute_score`, `_condition_cap`, `_dimension_items`, `_compute_reliability`, coverage, band and trend.
- `reminders.py` — `_get_reminders`, tracking recommendations, `_filter_recommendations`, weight recommendation, `_TrackingSpec`.
- `narrative.py` — prompt construction, `_WellnessNarrative` model, `_shorten_narrative`, `_fallback_narrative`, and the Gemini call.
- `service.py` — `WellnessService.__init__`, `metadata`, and `score()` as an orchestrator: score dimensions → aggregate → gate → reminders → narrative.

Rationale: this follows the existing private-function boundaries exactly, so every extraction is a move plus an import, with no logic edited. That keeps each step reviewable as a diff of relocations and keeps the existing wellness tests (617 + 361 lines across two files) as a strong regression net.

The Gemini client stays owned by `WellnessService` rather than being injected into `narrative.py`, so `build_services()` wiring is unchanged; `narrative.py` takes the client as a parameter.

### Schema split with a temporary re-export shim

`app/schemas.py` becomes a module that re-exports from `app/domain/enums.py`, `app/wellness/schemas.py`, `app/triage/schemas.py`, and `app/feeding/schemas.py`. Importers migrate incrementally; the shim is deleted in the final task, and the spec asserts it no longer exists at completion.

Rationale: `schemas.py` has many importers including tests and doc-sync checks. A shim makes the split a single mechanical commit that keeps the suite green, instead of a big-bang rename across every file at once. Leaving the shim permanently was rejected — it would reintroduce the choke point and hide accidental cross-domain imports.

### `main.py` splits, but `app.main:app` survives as an alias

The app object is constructed in `app/api/__init__.py`; `app/main.py` shrinks to re-export `app` and `ensure_services`.

Rationale: `template.yaml`/`CMD` reference `app.lambda_handler.handler`, `lambda_handler.py` imports from `app.main`, docs and the README tell users `uvicorn app.main:app`, and `conftest.py` reaches for the app module. Keeping that one import path stable removes deploy risk from the refactor at the cost of one two-line module. Unlike the schemas shim, this alias is permanent and intentional.

### Error mapping moves to exception handlers

`InvalidInputError → 400`, `InferenceUnavailableError → 500` become app-level handlers in `app/api/errors.py`, preserving the existing `{"detail": ..., "requestId": ...}` body and the existing generic-500 handler that logs but does not leak exception text.

Rationale: four routes repeat the same block today and a fifth would repeat it again. Handlers make the behavior uniform by construction. Note `/wellness` currently has no `try/except` at all — after this change it gains the same mapping, which is a strictly-better failure mode and not a contract change for successful requests.

## Risks / Trade-offs

- **A "pure move" silently changes behavior (import-time side effects, shadowed names, circular imports).** → Move one unit per commit with `make quality` green at each step; never edit logic and imports in the same commit. Circular-import risk is highest between `wellness/service.py` and `scoring/` — keep the dependency one-way (service imports scoring, never the reverse).
- **`tests/conftest.py` breaks on the predictor move and takes the whole suite with it.** → Update `conftest.py` in the same commit as the `app/ml` → `app/inference` move; that commit's success criterion is the full suite passing untouched otherwise.
- **Coverage percentage shifts when `ml_pipeline/` leaves the measured source.** → `source = ["app"]` already excludes it in effect, but `omit = ["app/ml/train.py"]` becomes a dead entry and the ratio changes as files split. Re-measure immediately after the move; if coverage drops below 85, add tests rather than lowering the gate.
- **A partially-migrated schema split leaves two definitions of the same model live at once.** → The shim re-exports rather than redefines; no model is ever declared twice. Verify with a check that each schema class has one definition site.
- **Lambda cold-start regression from new package-level imports.** → `ensure_services()` at INIT is preserved and the ONNX backend import stays lazy inside the backend branch. Keep `app/api/__init__.py` free of heavy imports.
- **Large diff makes review shallow and hides a real bug.** → Sequence the work so each task is independently green and separately reviewable; the mechanical moves land before the decompositions so the risky commits are small.
- **Trade-off: more files to navigate.** A wellness change now touches a directory instead of a file. Accepted — the current file requires reading 1332 lines to safely change 20.

## Migration Plan

Ordered so each step ends with `make quality` green and is independently revertable:

1. Split `app/ml/` — runtime modules to `app/inference/`, training modules to top-level `ml_pipeline/`. Update `conftest.py`, `pyproject.toml` (`include`, `known_first_party`, coverage `omit`), mypy target, and the training docs in the same commit.
2. Move `condition_metadata.py` → `app/domain/conditions.py`; extract shared enums to `app/domain/enums.py`.
3. Split `app/schemas.py` into per-domain schema modules behind a re-export shim.
4. Decompose `wellness_service.py`: `norms.py` first, then one scorer at a time, then `aggregate.py`, `reminders.py`, `narrative.py`, leaving `service.py` as orchestrator.
5. Move `triage_safety`/`chat_context`/`advice_safety` and the `use_cases/` modules into `app/triage/` and `app/feeding/`; move Gemini modules into `app/llm/`.
6. Split `main.py` into `app/api/` plus `app/bootstrap.py`; add `errors.py` handlers and drop the per-route `try/except`. Keep `app.main:app` as an alias.
7. Delete the `app/schemas.py` shim; update `CLAUDE.md`, `README.md`, and `docs/` module references; add the boundary/size checks that enforce this spec.

Rollback: each step is a self-contained commit on the branch with no data or API migration, so rollback is `git revert` of that commit. Nothing is deployed mid-sequence — deploy only after step 7 and a full `make quality` plus a Lambda smoke test of all four endpoints.

## Open Questions

- Should the boundary rules (no `app` → `ml_pipeline` import, no sibling-domain import, 250-line limit) be enforced by a test in the suite or by a lint plugin such as `import-linter`? A test needs no new dependency; a lint plugin gives better messages. Defaulting to a test unless a new dev dependency is acceptable.
- Does the per-domain entry point keep the `run_<domain>` naming from `use_cases/`, or move to a package-level `__init__` export? Defaulting to keeping the existing function names to minimize diff.
- Should `app/app_services.py` move under `app/bootstrap.py` or stay a standalone module? Defaulting to standalone, since `conftest.py` and the Lambda handler both reference the container concept directly.
