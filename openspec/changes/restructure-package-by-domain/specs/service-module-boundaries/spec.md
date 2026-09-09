## ADDED Requirements

### Requirement: Runtime serving code is separated from the training pipeline
The `app/` package SHALL contain only code required to serve HTTP requests. Dataset generation, dataset preparation, training, evaluation, ONNX export, and release-gate modules SHALL live in a top-level `ml_pipeline/` package outside `app/`. No module under `app/` SHALL import from `ml_pipeline/`. Model-bundle validation and release-version resolution SHALL stay under `app/`, because the predictors invoke them while loading a model at request time; `ml_pipeline/` MAY import them.

#### Scenario: Model load needs nothing outside `app/`
- **WHEN** a predictor loads a model bundle in an environment where only `app/` is present
- **THEN** bundle validation and release-version resolution SHALL succeed without importing `ml_pipeline`

#### Scenario: Serving code imports no training module
- **WHEN** every module under `app/` is scanned for imports
- **THEN** no import SHALL resolve to `ml_pipeline` or to any training-only third-party dependency (the `train` or `export` extras)

#### Scenario: Lambda image excludes the training pipeline
- **WHEN** a Lambda image is built from `Dockerfile.lambda` or `Dockerfile.lambda.onnx`
- **THEN** the image SHALL NOT contain `ml_pipeline/`

#### Scenario: Training pipeline still runs
- **WHEN** the environment is synced with the `train` extra and a pipeline entry point is invoked
- **THEN** it SHALL execute with the same behavior as before the move, reading and writing the same `data/` artifacts

### Requirement: Dependency direction between packages
Packages SHALL depend in one direction only: `app/api/` → domain packages (`app/wellness/`, `app/triage/`, `app/feeding/`) → shared packages (`app/inference/`, `app/llm/`, `app/domain/`, `app/config.py`, `app/observability.py`). A shared package SHALL NOT import a domain package, and a domain package SHALL NOT import another domain package.

#### Scenario: No inward-to-outward import
- **WHEN** modules in `app/inference/`, `app/llm/`, or `app/domain/` are scanned
- **THEN** none SHALL import from `app.api`, `app.wellness`, `app.triage`, or `app.feeding`

#### Scenario: No sibling domain coupling
- **WHEN** modules in one domain package are scanned
- **THEN** none SHALL import from another domain package

#### Scenario: Routes hold no business logic
- **WHEN** a module under `app/api/routes/` is inspected
- **THEN** it SHALL only validate/deserialize input, call a domain entry point, and map errors to HTTP responses

### Requirement: Domain packages own their schemas
Each domain package SHALL define its own Pydantic request/response models in its own `schemas.py`. Enums shared across domains (urgency, specialist, disease category, pet type) SHALL live in `app/domain/`. The aggregate module `app/schemas.py` SHALL be removed once all importers are migrated.

#### Scenario: Wellness models live with wellness code
- **WHEN** a wellness request or response model is resolved
- **THEN** it SHALL be defined in `app/wellness/schemas.py`

#### Scenario: Shared enum has a single definition
- **WHEN** `UrgencyLevel`, `SpecialistType`, `DiseaseCategory`, or `PetType` is imported from any package
- **THEN** it SHALL resolve to a single definition under `app/domain/`

#### Scenario: Shim removed at completion
- **WHEN** the change is complete
- **THEN** `app/schemas.py` SHALL NOT exist and no module SHALL import from it

### Requirement: Condition metadata is domain data, not ML code
The static map from each condition class to urgency, specialist, disease category, and home advice SHALL live in `app/domain/conditions.py` and SHALL be importable without loading any inference backend.

#### Scenario: Import without a model backend
- **WHEN** `app/domain/conditions.py` is imported in an environment with neither torch nor onnxruntime installed
- **THEN** the import SHALL succeed

#### Scenario: Coverage of all condition classes
- **WHEN** the metadata map is checked against the classifier label list
- **THEN** it SHALL contain an entry for every one of the 16 condition classes

### Requirement: Wellness scoring is decomposed into single-responsibility modules
Wellness scoring SHALL be split so that species norm tables, each of the six dimension scorers, score aggregation and the reliability gate, reminders and tracking recommendations, and narrative generation each live in their own module. The wellness service module SHALL only orchestrate these parts.

#### Scenario: Dimension scorer is independently testable
- **WHEN** a single dimension scorer is imported and called
- **THEN** it SHALL produce its breakdown item without constructing a wellness service or any LLM client

#### Scenario: Narrative generation is isolated
- **WHEN** narrative generation fails or is unavailable
- **THEN** the deterministic score, breakdown, reliability fields, reminders, and tracking recommendations SHALL still be produced

#### Scenario: Orchestrator holds no scoring rules
- **WHEN** the wellness service module is inspected
- **THEN** it SHALL contain no scoring thresholds, species norm tables, or prompt text

### Requirement: Module size limit
No Python module under `app/` SHALL exceed 250 source lines, excluding the static condition-metadata data map in `app/domain/conditions.py`, which is data rather than logic.

#### Scenario: Size check over the serving package
- **WHEN** source lines are counted for every module under `app/`
- **THEN** every module except the exempted data map SHALL be at or under 250 lines

### Requirement: Externally referenced entry points are preserved
The refactor SHALL NOT change how the application is started or invoked. `app.main:app` SHALL remain an importable ASGI application, and the Lambda handler SHALL remain at its existing import path with model loading still performed at INIT.

#### Scenario: Local server start
- **WHEN** `uvicorn app.main:app` is run
- **THEN** the application SHALL start and serve all existing routes

#### Scenario: Lambda handler path unchanged
- **WHEN** the Lambda runtime imports the handler configured in `template.yaml`
- **THEN** the import SHALL succeed and services SHALL be built at INIT so warm containers skip the model load

#### Scenario: Service wiring stays centralized
- **WHEN** a new service is added
- **THEN** it SHALL be wired through `AppServices` and the central service builder, not constructed inside a route

### Requirement: HTTP contract is unchanged by the refactor
Every endpoint SHALL keep its existing path, request schema, response schema, status codes, error body shape, and authentication behavior. The classifier SHALL remain the sole decision-maker for predicted conditions, and the deterministic red-flag safety layer SHALL still short-circuit ahead of any LLM call.

#### Scenario: Endpoint responses match the pre-refactor baseline
- **WHEN** the existing API test suite runs against the restructured application
- **THEN** every test SHALL pass without modification to its assertions

#### Scenario: Error body shape preserved
- **WHEN** a request fails validation or inference
- **THEN** the response body SHALL retain the `{"detail": ..., "requestId": ...}` shape and the same status code as before the refactor

#### Scenario: Red flag still precedes generation
- **WHEN** a red-flag phrase is present in a `/predict` or `/chat` request
- **THEN** urgency SHALL be overridden, no LLM call SHALL be made, and the fixed emergency message plus home advice SHALL be returned

### Requirement: Error mapping is centralized
Translation from use-case errors to HTTP status codes SHALL be defined once in the API layer rather than repeated per route. `InvalidInputError` SHALL map to 400 and `InferenceUnavailableError` SHALL map to 500.

#### Scenario: New route inherits mapping
- **WHEN** a route is added that raises `InvalidInputError` without its own `try/except`
- **THEN** the response SHALL be HTTP 400 with the standard error body

#### Scenario: Unexpected error is not leaked
- **WHEN** an unhandled exception escapes a route
- **THEN** the response SHALL be HTTP 500 with a generic message, and the exception detail SHALL appear only in server-side logs

### Requirement: Quality gates cover the relocated packages
Lint, type-check, and coverage configuration SHALL be updated so the relocated packages remain gated. Lint and type checking SHALL cover both `app/` and `ml_pipeline/`. The coverage gate SHALL measure branch coverage over the whole `app/` package, omitting only modules that cannot execute in the offline test suite, and SHALL fail below 85. Coverage scope SHALL be declared once in `pyproject.toml` rather than as a per-module allowlist in CI, so that a newly added module is gated by default.

#### Scenario: Type check spans both packages
- **WHEN** the type-check gate runs
- **THEN** it SHALL analyze `app/` and `ml_pipeline/`

#### Scenario: Coverage gate still enforced
- **WHEN** the test gate runs after the restructure
- **THEN** branch coverage over `app/` SHALL be measured and SHALL fail below 85

#### Scenario: Only offline-impossible modules are exempt
- **WHEN** the coverage omit list is inspected
- **THEN** it SHALL contain only `app/inference/onnx_predictor.py` (needs onnxruntime and an exported model directory) and `app/lambda_handler.py` (needs a real Lambda invocation event)

#### Scenario: New module is gated without a CI edit
- **WHEN** a module is added under `app/`
- **THEN** it SHALL count toward the coverage gate with no change to the CI workflow
