# Repository Improvement Specification

## Overview

This specification defines a staged modernization of the Pet Care AI Microservice. It is based on a repository-wide review of the FastAPI runtime, Torch and ONNX predictors, Gemini integrations, deterministic safety logic, wellness scoring, dataset tooling, model training and evaluation, tests, documentation, containers, AWS SAM deployment, and developer tooling.

The current system is functional and has useful safety and fallback mechanisms. The target state is a reproducible, testable classifier platform with explicit module boundaries, versioned model and dataset contracts, measurable safety gates, consistent configuration, and a single documented developer workflow.

Baseline observed during this review:

- 25 Python modules under `app/` and 5 test modules under `tests/`.
- 44 tests pass locally with the repository virtual environment.
- CI installs only `requirements-dev.txt`; pandas-dependent dataset tests and Torch-dependent model regression tests therefore skip in CI.
- The shipped Torch model identifies `distilbert-base-uncased` as its base model and exposes 16 labels.
- The local owner holdout contains 300 rows but only 5 of the 16 model classes.
- Torch evaluation on that holdout produced top-1 accuracy 0.770 and top-3 accuracy 0.953. The current `classification_report` also includes zero-support predicted classes, making its macro average unsuitable as a release gate.
- Local ignored model artifacts occupy approximately 332 MB; no version manifest, dataset hash, evaluation report, or model card binds them to the code that serves them.
- `pyproject.toml`, Ruff, Black, isort, mypy, coverage configuration, pre-commit, and `.editorconfig` are absent.

## Current Project Assessment

### Current architecture

```text
HTTP / Lambda
     |
     v
app/main.py --------------------------------------------------+
  |             |                    |                        |
  v             v                    v                        v
Predictor    safety rules       GeminiService          WellnessService
  |             |                    |                        |
  +----- Torch or ONNX --------------+------------------------+
                    |
                    v
          condition_metadata.py

Offline pipeline:

raw/local/HF/Gemini data
        -> merge/fetch scripts
        -> prepare_dataset.py
        -> train.py
        -> model directory
        -> export_onnx.py
        -> evaluate.py / runtime image
```

The diagram is directionally sound, but most contracts are implicit. `app/main.py` directly coordinates concrete predictors, metadata, red flags, LLM generation, fallbacks, and response construction. Training and evaluation reach into predictor internals. Data scripts repeat label-map loading and text normalization with slightly different behavior. Model directories are treated as valid if they merely exist.

### Strengths

- `app/services/triage_safety.py` provides a deterministic emergency override in front of probabilistic classification.
- `app/ml/predictor.py` lazily imports heavy ML libraries, keeping lightweight tests importable.
- `app/ml/onnx_predictor.py` provides a materially smaller deployment backend with the same conceptual operations.
- `app/ml/prepare_dataset.py` recognizes that owner-language evaluation must be separate from clinical-text evaluation.
- `tests/conftest.py` makes API tests deterministic and offline through `FakePredictor`.
- Gemini integrations validate structured responses with Pydantic and degrade to local fallbacks.
- Runtime, training, and ONNX dependency files are already separated.
- The repository includes API, .NET integration, ONNX deployment, smoke-test, Docker, Lambda, and SAM documentation.

### Weaknesses and evidence

- `app/main.py` contains service construction plus all three endpoint workflows; `predict` and `chat` are 65 and 91 lines respectively.
- `AppServices.predictor` is typed as `Predictor`, although `build_services()` can assign `OnnxPredictor`; there is no shared protocol.
- `Predictor` and `OnnxPredictor` duplicate validation, top-k semantics, label lookup, and probability-to-result conversion.
- `app/main.py` returns `Prediction failed: {exc}` to clients, exposing internal error detail.
- `/health` is liveness-only and does not report model readiness, backend, or model version.
- Chat red-flag detection examines only the latest user message, not the carried symptom summary or relevant recent context.
- `latest_user_message()` accepts an earlier user message even if the final message is from the assistant, contradicting the documented request contract.
- `ChatTurnPayload.mode` is an unconstrained string. Unexpected model output is silently handled as health mode.
- `gemini_used` reflects whether a client was configured, not whether the Gemini call succeeded.
- Gemini calls have no repository-defined timeout, retry, or circuit-breaker policy.
- Prompt inputs are interpolated into instruction text without a clear untrusted-data boundary or post-generation medical-advice policy.
- `WellnessPet` and related schemas permit negative or implausible values for weight, age, steps, calories, sleep, tracking days, and meal consistency.
- A request containing only `pet.species` receives 20/25 symptom points and can yield a `GOOD` score despite almost no observations.
- `app/schemas.py` defines `ChatResponse.model_config` twice; the second definition contains a `/predict` response example unrelated to the chat schema.
- Static home advice in `condition_metadata.py` includes treatment-like guidance that requires veterinary/safety review and version control.
- Training produces console output and model files but not a machine-readable run configuration, metrics, split manifest, dataset fingerprint, environment snapshot, or model card.
- Confidence thresholds are applied to uncalibrated softmax scores. Legacy `--calibrate` and `--tune` options are accepted and ignored.
- `prepare_dataset.py` creates an owner holdout covering only the classes available in `Owner Observation`; the current artifact covers 5/16 classes.
- `evaluate.py` is sequential, uses the private `_id2label` member, does not validate `top_k`, and cannot emit JSON suitable for CI comparison.
- Synthetic generation relies mainly on prompt instructions; it lacks diagnosis-leakage detection, similarity filtering, provenance fields, review sampling, and deterministic run metadata.
- `fetch_and_merge.py` catches source-fetch failures and continues, so the same command can silently produce materially different datasets.
- Model and parquet files are ignored by Git, but there is no DVC/MLflow/object-store manifest or documented immutable retrieval path.
- `.github/workflows/ci.yml` runs only pytest with lightweight dependencies. It does not lint, type-check, enforce coverage, validate SAM, build an image, or run a model-quality job.
- `README.md` still advertises removed `/ask` and nonexistent `app/services/ask_service.py`.
- `docs/deploy-onnx.md` states Lambda defaults of 1024 MB and 30 seconds, while `template.yaml` currently defaults to 1769 MB and 60 seconds.
- `README.md` describes Bio_ClinicalBERT as the active classifier, while `models/transformer_model/config.json` identifies DistilBERT.
- The container images run without an explicit non-root application user; `Dockerfile.lambda*` applies executable permissions to all files with `chmod -R 755`.
- SAM accepts empty API authentication by default and injects long-lived secrets directly into function environment variables.

## Key Problems

1. **Safety claims exceed evaluation evidence.** Emergency rules help, but classifier confidence is uncalibrated, full-class owner-language coverage is absent, and LLM-generated advice is not governed by a tested safety policy.
2. **Training is not reproducible.** Code, data, label map, model, metrics, and deployment artifacts are not joined by immutable identifiers.
3. **Runtime orchestration is too centralized.** Concrete implementations and environment reads make behavior hard to type-check, substitute, and test independently.
4. **Data pipeline behavior can drift silently.** Similar helpers differ across scripts, external source failures are non-fatal, and synthetic inputs lack quality gates.
5. **CI gives incomplete assurance.** The passing workflow does not execute dataset tests or real-model regressions and has no static-analysis or coverage gates.
6. **Documentation is not a reliable source of truth.** Endpoint, architecture, active model, and deployment settings disagree with the repository.
7. **Developer setup is fragmented.** Multiple requirements files exist without central project metadata, task commands, editor defaults, or pre-commit enforcement.

## Goals

- Establish explicit, typed boundaries between API transport, use-case orchestration, inference backends, safety policy, LLM generation, and configuration.
- Make every model release traceable to source commit, dependency lock, dataset fingerprint, label map, training parameters, evaluation metrics, and ONNX parity results.
- Define safety-oriented evaluation gates across all 16 classes, owner language, supported species, emergency phrases, abstention behavior, and adversarial inputs.
- Make CI representative: fast checks on every PR and artifact-backed model/deployment checks on controlled workflows.
- Consolidate Python tooling and dependency metadata in `pyproject.toml` while retaining deploy-specific lock/export files where necessary.
- Make documentation generated or verified against code and deployment configuration.
- Preserve current API field names and the Torch/ONNX deployment options during migration.

## Non-Goals

- Replacing the current classifier architecture solely for novelty.
- Adding persistent chat storage to this microservice; session persistence remains the caller's responsibility.
- Turning the service into a diagnostic medical device or claiming clinical validation.
- Migrating away from FastAPI, AWS Lambda/SAM, Gemini, Torch, or ONNX in the first modernization phase.
- Committing large model or parquet binaries directly to Git.
- Redesigning the .NET consumer beyond correcting and versioning its documented contract.
- Implementing changes as part of this exploration document.

## Proposed Architecture

```text
                         +----------------------+
HTTP / Mangum ----------> API routers + schemas |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Application use cases|
                         | predict/chat/wellness|
                         +---+---------+--------+
                             |         |
                 +-----------+         +----------------+
                 v                                      v
       +-------------------+                  +-------------------+
       | ClassifierProtocol|                  | TextGenerator     |
       +---------+---------+                  | protocol          |
                 |                            +---------+---------+
          +------+-------+                              |
          v              v                       Gemini / fallback
      Torch impl      ONNX impl
                 |
                 v
       +-----------------------+
       | Safety policy service |
       | red flags, abstention,|
       | advice allow-list     |
       +-----------------------+

Cross-cutting: validated Settings, structured errors, request IDs,
model manifest, metrics, readiness, and dependency injection.

Offline:

source adapters -> canonical data contract -> validation/quality report
-> immutable split manifest -> training run -> model bundle
-> full evaluation + calibration -> ONNX export/parity -> release manifest
```

Key design decisions:

- Use Python `Protocol` types for inference and text-generation ports. Concrete Torch, ONNX, Gemini, and fallback implementations remain adapters.
- Keep deterministic safety policy independent from both the classifier and Gemini.
- Move endpoint behavior into application use-case classes/functions that return domain results; routers translate domain errors to stable HTTP errors.
- Treat a model as a validated bundle, not a directory. A required manifest identifies model version, labels, tokenizer limit, dataset fingerprint, training run, metrics, backend files, and checksums.
- Treat dataset preparation as a deterministic build that emits data plus a quality/provenance report and immutable split membership.
- Separate fast CI from model CI, but make skips explicit failures in the jobs that claim to validate data or model behavior.

## Proposed Folder Structure

```text
app/
  api/
    app.py                    # FastAPI construction and middleware
    dependencies.py           # dependency wiring
    errors.py                 # stable public error model/handlers
    routers/
      health.py
      predict.py
      chat.py
      wellness.py
    schemas/                  # transport-only Pydantic contracts
  application/
    predict.py                # prediction use case
    chat.py                   # chat turn use case
    wellness.py               # wellness use case
  domain/
    classification.py        # Prediction, label, top-k contracts
    safety.py                 # red flags, abstention, advice policy
    wellness.py               # deterministic score domain types
  infrastructure/
    config.py                 # validated Settings
    inference/
      protocol.py
      torch_predictor.py
      onnx_predictor.py
      model_manifest.py
    generation/
      protocol.py
      gemini.py
      fallback.py
    observability.py
  lambda_handler.py
ml/
  data/
    contract.py               # canonical row/schema and validation
    normalize.py              # shared text/label normalization
    sources/                  # local, HF, synthetic adapters
    build_dataset.py
    split_dataset.py
  training/
    config.py
    train.py
    metrics.py
  evaluation/
    evaluate.py
    parity.py
    safety_suite.py
  export/
    onnx.py
configs/
  training/
  evaluation/
model_bundles/                # ignored payloads; tracked manifest templates only
tests/
  unit/
  integration/
  contract/
  model/
  fixtures/
docs/
  architecture.md
  model-card.md
  data-card.md
  api-reference.md
  deployment.md
pyproject.toml
.pre-commit-config.yaml
.editorconfig
```

This structure is a target, not a prerequisite for every fix. Migration should use compatibility imports and small moves to avoid a high-risk rewrite.

## Required Changes

### RC-1: Introduce validated, centralized configuration

- **Problem:** `app/main.py`, `lambda_handler.py`, services, scripts, `.env.example`, Dockerfiles, and `template.yaml` read or define settings independently. Defaults already differ for Gemini models and deployment resources.
- **Recommended solution:** Add a frozen Pydantic Settings model with explicit runtime modes (`local`, `test`, `production`) and startup validation.
- **Affected files or areas:** `app/main.py`, new `app/infrastructure/config.py`, `.env.example`, `template.yaml`, Dockerfiles, tests.
- **Implementation details:** Parse `MODEL_BACKEND`, `MODEL_PATH`, Gemini settings, API auth, thresholds, root path, log level, timeouts, retry limits, and static-explanation mode once. Reject unsupported backends and invalid thresholds. Require API authentication or an explicit `ALLOW_UNAUTHENTICATED=true` override in production. Inject settings into service construction. Do not read environment variables inside request handlers.
- **Priority:** Critical
- **Expected benefit:** Deterministic startup, safer production defaults, testable configuration, and elimination of documentation/default drift.

### RC-2: Define inference and generation protocols

- **Problem:** Runtime code depends on concrete `Predictor`; ONNX is assigned to a Torch-typed field, and evaluation reads `_id2label` directly.
- **Recommended solution:** Define `ClassifierProtocol` and `TextGeneratorProtocol` with public metadata and typed results.
- **Affected files or areas:** `app/ml/predictor.py`, `app/ml/onnx_predictor.py`, `app/main.py`, `app/ml/evaluate.py`, `app/services/gemini_service.py`, tests.
- **Implementation details:** The classifier protocol should expose `predict`, `predict_top_k`, `labels`, `model_version`, and `backend`. Validate `k >= 1`, label/logit cardinality, contiguous label ids, and tokenizer/model files during construction. Move shared input validation and result conversion into reusable helpers. Model generated outcomes as discriminated success/fallback results so telemetry reports actual fallback use.
- **Priority:** High
- **Expected benefit:** Static type safety, backend parity, simpler fakes, and less duplicated inference logic.

### RC-3: Extract application use cases from route handlers

- **Problem:** `app/main.py` combines HTTP concerns, prediction, safety, metadata, Gemini, logging, and response assembly.
- **Recommended solution:** Add separate predict, chat, and wellness application services; keep routers thin.
- **Affected files or areas:** `app/main.py`, `app/services/*`, `app/schemas.py`, new `app/application/*`, API tests.
- **Implementation details:** Each use case receives protocols and settings through construction, performs one workflow, and returns a domain result. FastAPI dependencies resolve the use case. Add typed domain exceptions and centralized handlers. Preserve current JSON aliases and status codes unless this specification explicitly changes them.
- **Priority:** High
- **Expected benefit:** Smaller units, easier integration testing, clearer ownership, and reduced coupling to FastAPI globals.

### RC-4: Harden public errors and readiness

- **Problem:** Unexpected prediction exceptions are embedded in HTTP 500 responses, and `/health` does not prove model readiness.
- **Recommended solution:** Return stable error codes/messages, log exception details server-side, and split liveness from readiness.
- **Affected files or areas:** `app/main.py`, `app/observability.py`, API schemas, `docs/api-reference.md`, tests, container/SAM health configuration.
- **Implementation details:** Add a public error envelope with `code`, `message`, and `requestId`. Keep validation errors machine-readable. Add `/health/live` and `/health/ready`; readiness checks that settings and the model bundle loaded, and reports non-sensitive backend/model version metadata. Keep `/health` as a compatibility alias during migration.
- **Priority:** Critical
- **Expected benefit:** No internal information leakage, better operations, and actionable deployment health checks.

### RC-5: Strengthen safety policy and chat context handling

- **Problem:** Chat red flags only inspect the latest message; the code does not enforce that the final turn is from the user; generated home advice lacks a post-generation safety gate.
- **Recommended solution:** Centralize safety policy and evaluate the latest message plus bounded symptom context before any LLM call.
- **Affected files or areas:** `app/services/triage_safety.py`, `app/services/chat_context.py`, `app/main.py`, `app/services/gemini_service.py`, `app/ml/condition_metadata.py`, safety tests and docs.
- **Implementation details:** Validate that `messages[-1].role == user`. Scan the latest message and normalized carried summary while preventing repeated stale false positives through an explicit emergency-state rule. Add negation-aware test cases such as “is breathing normally” and “no seizure.” Validate generated advice against forbidden categories such as human medication, unsupported dosing, diagnosis claims, or delaying emergency care. Require veterinary review and version metadata for static advice. Keep emergency responses deterministic and LLM-free.
- **Priority:** Critical
- **Expected benefit:** Lower false-negative emergency risk, fewer contract ambiguities, and controlled medical-advice output.

### RC-6: Constrain LLM inputs, outputs, and failure policy

- **Problem:** Gemini prompts interpolate untrusted text into instructions, mode is a free string, calls lack explicit timeout/retry policy, and configured-client telemetry is mistaken for successful generation.
- **Recommended solution:** Model prompt data separately from instructions, constrain all structured outputs, and expose generation outcomes.
- **Affected files or areas:** `app/services/gemini_service.py`, `app/services/wellness_service.py`, configuration, observability, tests.
- **Implementation details:** Use `Literal["general", "health"]` or a shared enum for mode. Bound transcript, summary, topics, answer, recommendations, and advice lengths. Add timeout, limited retry with jitter for transient errors, no retry for validation/quota errors, and a short circuit after repeated failure. Return metadata such as `provider_used`, `fallback_reason`, and latency internally. Add prompt-injection regression fixtures and mocked malformed/empty/quota/timeout responses.
- **Priority:** Critical
- **Expected benefit:** Predictable latency, accurate telemetry, bounded cost, and more robust generated content.

### RC-7: Correct wellness validation and data sufficiency semantics

- **Problem:** Wellness inputs accept implausible numeric values, and an almost empty request can produce a confident `GOOD` score because absent symptoms receive 20/25 points.
- **Recommended solution:** Add domain bounds and explicit data sufficiency to the wellness result.
- **Affected files or areas:** `app/schemas.py`, `app/services/wellness_service.py`, API docs, tests.
- **Implementation details:** Reuse `PetType` for species or explicitly validate `other`. Constrain age, weight, steps, minutes, sleep, meals, calories, and tracking days to defensible ranges. Treat absent symptoms as an excluded dimension rather than positive evidence, or clearly define a neutral prior approved by product/clinical review. Return `dataCoverage` and a confidence/quality band; do not label sparse records `GOOD`/`EXCELLENT` unless a minimum set of dimensions is present. Replace keyword-only condition caps with normalized condition codes where available.
- **Priority:** Critical
- **Expected benefit:** Prevents misleading wellness scores and makes partial-data behavior explainable.

### RC-8: Fix and version API schemas

- **Problem:** `ChatResponse` defines `model_config` twice and contains the wrong example; several unbounded strings/lists can inflate request cost.
- **Recommended solution:** Split transport schemas by endpoint, correct examples, and add explicit limits and cross-field validators.
- **Affected files or areas:** `app/schemas.py`, new `app/api/schemas/*`, OpenAPI output, API docs, contract tests.
- **Implementation details:** Bound `sessionId`, `symptomSummary`, breed, notes, condition/medication names, lists, topics, and recommendations. Add model validators for final-user-message and coherent mode/prediction combinations. Snapshot the generated OpenAPI contract. Introduce an explicit API version policy before breaking existing aliases.
- **Priority:** High
- **Expected benefit:** Accurate generated docs, bounded resource use, and stable consumer contracts.

### RC-9: Establish a canonical dataset contract and shared preprocessing

- **Problem:** Label-map loading, whitespace cleanup, missing-label handling, and deduplication are repeated with different behavior across `merge_datasets.py`, `fetch_and_merge.py`, `prepare_dataset.py`, `check_leakage.py`, and `train.py`.
- **Recommended solution:** Create one canonical row schema and preprocessing library used by every data command.
- **Affected files or areas:** all `app/ml/*dataset*`, merge/fetch/leakage/train scripts, `data/label_map.json`, dataset tests.
- **Implementation details:** Define required `text`, canonical `condition`, `record_type`, `source_id`, `source_version`, `license`, `generated_by`, `generation_run_id`, and optional species fields. Centralize normalization and label-map validation. Reject unknown labels and sentinel labels according to configuration. Emit row counts and rejection reasons. Use content-based deduplication and optional semantic-similarity review before splitting.
- **Priority:** Critical
- **Expected benefit:** Consistent data behavior, auditable provenance, and fewer train/serve label mismatches.

### RC-10: Make external and synthetic data builds deterministic and fail-visible

- **Problem:** External source failures are swallowed, source revisions are not pinned, and synthetic data receives limited automated quality review.
- **Recommended solution:** Use explicit source adapters with strict/best-effort modes and emit a dataset build manifest.
- **Affected files or areas:** `app/ml/fetch_and_merge.py`, `app/ml/generate_synthetic.py`, `app/ml/merge_datasets.py`, docs, new configuration files.
- **Implementation details:** Pin Hugging Face dataset revisions and record accepted licenses. In strict mode, fail if a requested source cannot load; in best-effort mode, mark the build incomplete. For synthetic rows, record provider/model/prompt version/temperature/time, detect label-name leakage, reject duplicates and near-duplicates, enforce species/style constraints, and produce a stratified manual-review sample. Correct API call estimation and define append/replace behavior explicitly.
- **Priority:** High
- **Expected benefit:** Repeatable dataset builds, clearer licensing, and lower synthetic-data contamination risk.

### RC-11: Create immutable split and leakage policy

- **Problem:** `train.py` creates row-level random splits at training time; current duplicate checks do not prevent semantic or source-family leakage, and the external owner holdout covers only 5 classes.
- **Recommended solution:** Generate and persist split membership before training, grouped by normalized/semantic family and source entity.
- **Affected files or areas:** `app/ml/prepare_dataset.py`, `app/ml/check_leakage.py`, `app/ml/train.py`, data manifest, tests.
- **Implementation details:** Assign stable row ids. Group near-duplicates and generated variants before splitting. Preserve a locked test set that is not used for iteration. Build owner-language evaluation coverage for all supported classes or mark unsupported classes explicitly. Report distribution by class, species, source, and register. Fail builds on cross-split exact leakage and define a threshold/manual review for semantic similarity.
- **Priority:** Critical
- **Expected benefit:** Trustworthy metrics and comparable model iterations.

### RC-12: Make training reproducible and produce a model bundle

- **Problem:** Training saves weights/tokenizer only; important context exists only in console output. Reproducibility seeding is incomplete and focal-loss weighting is mathematically ambiguous.
- **Recommended solution:** Drive training from a validated config and emit a complete model bundle.
- **Affected files or areas:** `app/ml/train.py`, model directories, requirements, CI/model workflows, docs.
- **Implementation details:** Save training config, dataset and split fingerprints, label-map checksum, source commit, Python/dependency versions, random seed, metrics JSON, confusion matrix, calibration data, and model card. Enable deterministic algorithms where supported and record exceptions. Replace ignored legacy flags with explicit errors or implement them. Correct focal loss so class weighting is applied according to a documented formula; test it numerically. Save checkpoints to CPU or disk rather than cloning an entire device state indefinitely. Validate output bundle before publishing.
- **Priority:** Critical
- **Expected benefit:** Rebuildable releases, defensible experiments, and safer model promotion.

### RC-13: Calibrate confidence and define abstention/release thresholds

- **Problem:** `LOW_CONFIDENCE_THRESHOLD` and `ABSTAIN_THRESHOLD` operate on raw softmax confidence with no calibration evidence.
- **Recommended solution:** Calibrate on a dedicated validation set and version thresholds with the model.
- **Affected files or areas:** `app/ml/train.py`, `app/ml/evaluate.py`, predictors, model manifest, `triage_safety.py`, tests.
- **Implementation details:** Measure expected calibration error, Brier score, reliability curves, and selective accuracy/coverage. Compare temperature scaling or another simple held-out calibration method. Store calibration parameters and thresholds in the model bundle. Define threshold selection around safety and product costs, not a hard-coded global constant. Evaluate per-class and out-of-distribution abstention.
- **Priority:** Critical
- **Expected benefit:** Confidence values and clarification behavior become measurable rather than cosmetic.

### RC-14: Expand evaluation into a release gate

- **Problem:** `evaluate.py` prints a single report, runs one sample at a time, uses private state, and the current holdout is not representative of all labels/species.
- **Recommended solution:** Build a batch-capable evaluation suite that emits versioned JSON and Markdown reports and compares against a baseline.
- **Affected files or areas:** `app/ml/evaluate.py`, new `ml/evaluation/*`, test fixtures, CI/model workflow, model bundle.
- **Implementation details:** Report macro/weighted F1 using an explicit label set, per-class precision/recall, confusion matrix, top-k, calibration, abstention coverage, latency, and memory. Segment by source/register/species. Add emergency red-flag recall and false-positive tests separately from classifier metrics. Add Torch-vs-ONNX agreement and maximum metric-delta gates. Fail if required classes have zero support instead of silently averaging them.
- **Priority:** Critical
- **Expected benefit:** Comparable, safety-relevant release decisions and correct metric interpretation.

### RC-15: Version, validate, and distribute model artifacts

- **Problem:** Deployment copies ignored local directories with no immutable retrieval, checksum enforcement, or code/model compatibility check.
- **Recommended solution:** Introduce a model registry or object-store release process backed by a tracked manifest.
- **Affected files or areas:** `models/`, Dockerfiles, SAM/build scripts, new manifest schema, deployment docs.
- **Implementation details:** Each release gets a semantic or date-based version, checksums for every artifact, label/schema version, minimum runtime version, evaluation report link, and provenance. CI downloads by immutable version and verifies hashes before building. Runtime validates manifest compatibility and reports the version in readiness/logs. Keep large binaries out of Git.
- **Priority:** Critical
- **Expected benefit:** Reproducible deployments, rollback, tamper detection, and clear model ownership.

### RC-16: Improve observability and privacy controls

- **Problem:** JSON logging lacks request correlation, model version, fallback reason, and metrics; `sessionId` is documented for telemetry but unused.
- **Recommended solution:** Define a privacy-reviewed event schema and operational metrics.
- **Affected files or areas:** `app/observability.py`, use cases, Lambda/SAM, operational docs and tests.
- **Implementation details:** Add request id, endpoint, status, latency, backend, model version, confidence band, red-flag result, actual generation provider/fallback reason, and error code. Do not log symptom text, summaries, API keys, or raw prompts. Hash or omit session ids according to retention policy. Publish counters/histograms for fallbacks, abstentions, red flags, inference errors, and latency. Add log retention and alarms in infrastructure.
- **Priority:** High
- **Expected benefit:** Diagnosable production behavior without collecting sensitive pet-owner text.

### RC-17: Harden containers and AWS deployment

- **Problem:** Images are not built in CI, use broad file permissions, lack an explicit non-root user, and production auth can be disabled by an empty default.
- **Recommended solution:** Add reproducible container/SAM validation and secure defaults.
- **Affected files or areas:** `Dockerfile`, `Dockerfile.lambda`, `Dockerfile.lambda.onnx`, `template.yaml`, CI and deployment docs.
- **Implementation details:** Pin base images by digest on release branches, install from locked hashes, remove unused `torchvision`, use a non-root user where compatible, and set read-only model permissions rather than `chmod -R 755`. Add container health checks for the server image. Require production API auth or a stronger gateway authorizer. Prefer Secrets Manager/SSM references over long-lived plain environment values. Configure log retention, reserved concurrency, alarms, and tracing. Validate whether keep-warm remains justified with measured cold starts.
- **Priority:** High
- **Expected benefit:** Smaller attack surface, reproducible builds, and safer production deployment.

### RC-18: Remove legacy compatibility ambiguity

- **Problem:** Removed endpoints remain in docs, ignored training flags remain accepted, joblib artifacts/dependency references remain, and old/new model narratives coexist.
- **Recommended solution:** Inventory legacy behavior, define deprecation dates, then remove it deliberately.
- **Affected files or areas:** `README.md`, docs, `train.py`, predictor signatures, `requirements-train.txt`, ignored local joblib files, smoke tests.
- **Implementation details:** Remove `/ask` documentation and nonexistent modules immediately. Replace silently ignored CLI arguments and `**_ignored` constructor parameters with deprecation warnings followed by removal. Remove `joblib` if no supported path needs it. Document the active model from its manifest rather than hard-coding an architecture name.
- **Priority:** Medium
- **Expected benefit:** Less misleading surface area and fewer accidental no-op configurations.

## Documentation Improvements

### DI-1: Rewrite README as the authoritative entry point

- **Problem:** The README contains stale architecture, endpoint, module, model, and deployment information.
- **Recommended solution:** Reduce it to verified setup, architecture summary, supported endpoints, development commands, and links to deeper docs.
- **Affected files or areas:** `README.md`, `docs/*`.
- **Implementation details:** Remove `/ask` and `ask_service.py`; describe `/chat`; report the active model dynamically or refer to the model card; document Torch and ONNX profiles; distinguish runtime, dev, and training installation; include exact quickstart and test commands.
- **Priority:** High
- **Expected benefit:** Reliable onboarding and fewer integration errors.

### DI-2: Add architecture and decision records

- **Problem:** Important decisions are present only in comments and long module docstrings.
- **Recommended solution:** Add `docs/architecture.md` and concise ADRs.
- **Affected files or areas:** new `docs/architecture.md`, `docs/adr/*`, source docstrings.
- **Implementation details:** Document runtime components, data/model lifecycle, safety layering, stateless chat contract, backend selection, failure modes, and trust boundaries. Record decisions for deterministic safety, LLM role, model registry, owner-language evaluation, and ONNX deployment.
- **Priority:** Medium
- **Expected benefit:** Maintainers can understand why the design exists before changing it.

### DI-3: Add data card and model card

- **Problem:** Dataset composition and model limitations are scattered through scripts and a session checklist.
- **Recommended solution:** Generate versioned data/model cards from build and evaluation artifacts.
- **Affected files or areas:** new `docs/data-card.md`, `docs/model-card.md`, training/evaluation outputs.
- **Implementation details:** Include source, license, class/species/register coverage, synthetic proportions, exclusions, known gaps, intended use, prohibited use, training config, metrics, calibration, safety tests, ethical limitations, and rollback version.
- **Priority:** Critical
- **Expected benefit:** Transparent model governance and informed product use.

### DI-4: Keep API and deployment docs synchronized

- **Problem:** Handwritten API examples and Lambda defaults drift from schemas and `template.yaml`.
- **Recommended solution:** Generate OpenAPI JSON in CI and validate documented configuration values.
- **Affected files or areas:** `docs/api-reference.md`, `docs/deploy-onnx.md`, `docs/dotnet-chat-integration.md`, `template.yaml`, tests.
- **Implementation details:** Add contract snapshots and a script/test that compares documented environment/default values to settings/SAM. Correct current 1769 MB/60-second defaults. Mark examples as illustrative when values vary. Add API version and deprecation policy.
- **Priority:** High
- **Expected benefit:** Consumer and operator documentation remains accurate.

### DI-5: Replace the session checklist with maintained runbooks

- **Problem:** `docs/remaining-checklist.md` is a stale point-in-time status report; it says 31 tests pass and lists ONNX/SAM work already present.
- **Recommended solution:** Move durable items into production, model-release, and incident runbooks; track transient work in issues/OpenSpec changes.
- **Affected files or areas:** `docs/remaining-checklist.md`, new `docs/runbooks/*`.
- **Implementation details:** Add pre-release, deployment, rollback, degraded-Gemini, bad-model, quota-exhaustion, and data-rebuild procedures. Every command should be executable and every expected output measurable.
- **Priority:** Medium
- **Expected benefit:** Operational documentation remains actionable instead of becoming historical noise.

### DI-6: Add contributor and security documentation

- **Problem:** There is no concise contribution workflow, supported Python policy, vulnerability reporting path, or data-handling policy.
- **Recommended solution:** Add `CONTRIBUTING.md`, `SECURITY.md`, and a privacy/data-handling section.
- **Affected files or areas:** repository root and docs.
- **Implementation details:** Document setup profiles, task commands, code/test standards, model-change review requirements, secret handling, prohibited production-data commits, responsible disclosure, and clinical-review ownership.
- **Priority:** Medium
- **Expected benefit:** Safer, faster onboarding and clearer governance.

## Tooling Improvements

### TI-1: Add `pyproject.toml` as the tooling source of truth

- **Problem:** Tool configuration is fragmented and most quality tools are absent.
- **Recommended solution:** Add PEP 621 project metadata plus centralized Ruff, Black, isort, mypy, and pytest configuration.
- **Affected files or areas:** new `pyproject.toml`, `pytest.ini`, requirements files.
- **Implementation details:** Set Python `>=3.11,<3.13` initially; define runtime, dev, train, and ONNX/export optional dependency groups; configure a 100-character line length; move pytest settings from `pytest.ini`; enable strict-enough mypy incrementally with per-module exceptions documented. Ruff should cover import order and replace standalone isort enforcement; Black remains the formatter.
- **Priority:** High
- **Expected benefit:** One discoverable configuration surface and reproducible local/CI behavior.

### TI-2: Lock dependencies by environment

- **Problem:** Exact runtime pins coexist with open-ended dev pins and unpinned transitive dependencies; builds are not hash-locked.
- **Recommended solution:** Use a lock/export workflow for runtime, dev, train, and ONNX images.
- **Affected files or areas:** `requirements*.txt`, `pyproject.toml`, Dockerfiles, CI.
- **Implementation details:** Choose `uv` or `pip-tools`; commit generated lock files with hashes and update them through an automated PR process. Keep CPU Torch index handling explicit. Add `pip check` and a vulnerability scanner. Remove runtime pandas/pyarrow if no runtime code imports them on the selected backend.
- **Priority:** High
- **Expected benefit:** Repeatable environments, smaller images, and controlled upgrades.

### TI-3: Add pre-commit and editor defaults

- **Problem:** Formatting, trailing whitespace, YAML validity, and secret checks depend on individual developers.
- **Recommended solution:** Add pre-commit hooks and `.editorconfig`.
- **Affected files or areas:** new `.pre-commit-config.yaml`, `.editorconfig`, contributor docs.
- **Implementation details:** Run Ruff fix/check, Black, basic file/YAML checks, end-of-file/whitespace checks, and a secret scanner. Exclude model/data binaries and generated artifacts explicitly.
- **Priority:** Medium
- **Expected benefit:** Fast local feedback and lower CI churn.

### TI-4: Provide stable task commands

- **Problem:** README commands are long and profiles are easy to install incorrectly.
- **Recommended solution:** Add a small `Makefile` or `justfile` wrapping environment-neutral commands.
- **Affected files or areas:** new task file, README, CI.
- **Implementation details:** Include `setup-dev`, `format`, `lint`, `typecheck`, `test-fast`, `test-data`, `test-model`, `evaluate`, `export-onnx`, `parity`, `serve`, `smoke`, `sam-validate`, and `ci` targets. Targets must delegate to Python modules rather than duplicate logic.
- **Priority:** Medium
- **Expected benefit:** Consistent commands for humans and automation.

### TI-5: Expand CI into explicit quality jobs

- **Problem:** The single test job gives no indication that data/model tests skip and performs no static or deployment validation.
- **Recommended solution:** Create fast, data, model, and deployment jobs with explicit dependencies.
- **Affected files or areas:** `.github/workflows/ci.yml`, optional scheduled/release workflows.
- **Implementation details:** On every PR run format check, Ruff, mypy, unit/integration tests, coverage threshold, OpenAPI snapshot, and SAM validation. Install pandas in a data-test job and fail if tests unexpectedly skip. Run model regression, full evaluation, ONNX parity, and container builds only when model artifacts are available through authenticated immutable download or on a scheduled/release workflow. Pin actions to commit SHAs for release-sensitive workflows.
- **Priority:** Critical
- **Expected benefit:** CI claims match what is actually tested and model/deployment regressions become visible.

## Testing Strategy

### Test layers

| Layer | Scope | Required examples |
|---|---|---|
| Unit | Pure domain and preprocessing functions | thresholds, normalization, scoring bounds, mode validators, red-flag positive/negative/negated phrases |
| Component | Predictor and generator adapters | invalid manifests, missing labels, `k` bounds, malformed Gemini responses, timeouts, fallback metadata |
| API integration | FastAPI with injected fakes | auth modes, stable errors, readiness, final-user-message rule, sparse wellness data, request limits |
| Data contract | Dataset builds and split manifests | schema failures, unknown labels, provenance, strict source failure, exact/near-duplicate leakage |
| Model behavior | Versioned fixed fixtures and full holdouts | all 16 classes, owner/clinical registers, supported species, OOD/abstention, class-specific recall |
| Safety behavior | Classifier-independent safety suite | emergency recall, negation, adversarial phrasing, generated-advice forbidden content |
| Backend parity | Torch vs ONNX | label agreement, probability/metric tolerance, tokenizer truncation parity |
| Deployment | Container and SAM | image build, non-root/read-only behavior, startup, readiness, smoke tests |

### Coverage and gate policy

- Enforce at least 85% line coverage for application/domain code initially; exclude generated schemas and heavy third-party execution paths only with written rationale.
- Do not use aggregate coverage to excuse untested safety branches. Emergency, abstention, fallback, configuration failure, and public-error paths require direct tests.
- Fast PR tests must not depend on network or real Gemini credentials.
- Data and model jobs must fail on unexpected skips. Optional local skips remain acceptable only in the fast developer profile.
- A model release must meet versioned gates for full-class support, macro-F1, safety recall, calibration, selective accuracy, Torch/ONNX delta, and latency. Exact thresholds must be established from the first trusted baseline rather than invented in code.
- Regression fixtures must include counterexamples and ambiguous text, not only clean textbook symptoms.

### Immediate missing tests

- `ChatResponse` OpenAPI example and mode/prediction consistency.
- Trailing assistant message rejected when the contract requires the final message to be user-authored.
- Red flags present only in prior symptom context, plus negated emergency phrases.
- Gemini configured but failing: telemetry must report fallback, not Gemini success.
- Prompt injection and oversized summary/session/list inputs.
- Negative and implausible wellness values; species-only request must not receive a confident positive band.
- Torch/ONNX `predict_top_k` behavior for `k=0`, negative `k`, and `k > labels`.
- Missing/corrupt model files, noncontiguous labels, label/logit mismatch, and checksum mismatch.
- Full-class evaluation support and correct explicit-label macro-F1.
- Synthetic diagnosis leakage, duplicate/near-duplicate rejection, and strict source failure.
- Real model regression on CI with a fixed versioned artifact; correct `_weights_present` handling for both safetensors and PyTorch bin files.
- Docker/SAM build and readiness smoke test.

## Migration Plan

### Phase 0: Establish an evidence baseline

1. Freeze current API OpenAPI JSON, current model/data checksums, test result, and evaluation report.
2. Record the current 0.770 top-1 and 0.953 top-3 owner-holdout result with the caveat that only 5/16 classes are represented.
3. Create issues/OpenSpec changes for clinical advice review and the wellness sparse-data product decision.
4. Do not promote a new model until split provenance and evaluation output are machine-readable.

### Phase 1: Tooling, configuration, and safety correctness

1. Add `pyproject.toml`, locks, Ruff/Black/mypy/coverage, pre-commit, EditorConfig, and stable task commands.
2. Introduce validated settings and stable public errors.
3. Fix duplicate/wrong schema configuration and add request bounds.
4. Enforce final-user-message semantics, actual fallback telemetry, and red-flag context tests.
5. Correct sparse wellness behavior after product/clinical approval.
6. Update README and deployment/API docs to current reality.

### Phase 2: Runtime boundary refactor

1. Introduce protocols and manifest-backed classifier metadata without changing endpoints.
2. Extract predict/chat/wellness use cases behind compatibility routers.
3. Add readiness, model version logging, LLM timeout/retry policy, and advice validation.
4. Keep import shims temporarily; remove them only after tests and consumers migrate.

### Phase 3: Reproducible data and training

1. Introduce canonical dataset schema and source adapters.
2. Generate stable row ids, build manifests, provenance reports, and immutable grouped splits.
3. Expand owner-language evaluation to all intended classes/species or explicitly narrow product claims.
4. Make training config-driven and emit a complete bundle/model card.
5. Calibrate confidence and version abstention thresholds.

### Phase 4: Release automation

1. Publish immutable model bundles to a registry/object store.
2. Add model evaluation and ONNX parity workflows.
3. Build and scan containers in CI; verify checksums during build/startup.
4. Add SAM validation, alarms, log retention, stronger secret handling, and rollback automation.

### Phase 5: Cleanup

1. Remove legacy `/ask` documentation, ignored CLI flags, `**_ignored` compatibility, and unused joblib dependencies/artifacts.
2. Remove compatibility imports after one documented deprecation window.
3. Archive `remaining-checklist.md` after durable items live in runbooks and tracked work items.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Large folder refactor breaks imports/Lambda handler | Deployment outage | Introduce protocols/use cases in place first; move files behind compatibility imports; smoke-test both Uvicorn and Mangum. |
| Stricter validation breaks existing callers | 422 responses | Snapshot real payloads, add compatibility aliases, publish limits/versioning, and stage warnings before enforcement where safe. |
| Correcting sparse wellness semantics changes user-visible scores | Product confusion | Treat as a versioned scoring-policy change, backtest stored examples, expose coverage, and communicate the change. |
| Calibration lowers apparent confidence or increases abstention | More clarification flows | Optimize for measured selective accuracy and safety; track coverage as a product metric. |
| Full-class owner holdout is expensive to curate | Delayed model gate | Narrow supported claims temporarily; prioritize high-risk/confused classes; document unsupported segments. |
| Model registry adds operational complexity | Slower releases initially | Start with versioned object storage plus checksums and a simple manifest before adopting a larger platform. |
| Strict external-source mode reduces data availability | Failed builds | Support explicit strict and best-effort modes; never label an incomplete build as complete. |
| LLM safety filtering rejects useful advice | More generic fallbacks | Prefer reviewed deterministic advice, log only policy reason codes, and tune with a reviewed corpus. |
| Expanded CI becomes slow/expensive | Reduced developer throughput | Keep fast PR jobs lightweight; run model/container jobs only on relevant changes, schedule, or release. |
| Dependency consolidation disrupts Lambda image size | Cost/cold-start regression | Maintain separate locked runtime profiles and enforce image size/cold-start budgets. |
| Clinical guidance changes require expertise outside engineering | Safety work blocked | Assign named veterinary reviewer/owner and make approval an explicit release artifact. |

## Acceptance Criteria

### Architecture and runtime

- [ ] Route modules contain HTTP translation only; predict, chat, and wellness workflows are independently unit-testable.
- [ ] Torch and ONNX implementations satisfy one typed classifier protocol with no ignored type errors at the wiring boundary.
- [ ] Runtime settings are parsed once, validated at startup, and documented from the same defaults.
- [ ] Public 500 responses contain no internal exception text and include a request id.
- [ ] Readiness reports a loaded, checksum-validated model version and backend without exposing secrets or paths.
- [ ] Production startup fails when authentication is absent unless an explicit unauthenticated override is set.

### Safety and product behavior

- [ ] Chat validates that the last message is a non-empty user message.
- [ ] Emergency detection covers bounded relevant context and passes reviewed positive, negative, negation, and adversarial fixtures.
- [ ] Emergency responses remain deterministic and do not call the LLM.
- [ ] LLM output types are constrained; failures/timeouts produce bounded fallbacks and accurate fallback telemetry.
- [ ] Generated/static home advice passes a documented automated policy and named veterinary review.
- [ ] Sparse wellness requests return explicit insufficient-data/coverage semantics and cannot be labeled confidently healthy based on species alone.
- [ ] Wellness numeric inputs have documented, tested bounds.

### Data, model, and evaluation

- [ ] Every dataset build emits an immutable manifest with source revisions, licenses, transformations, counts, rejects, and fingerprints.
- [ ] Every row used for training/evaluation has a stable id and provenance.
- [ ] Split manifests prevent exact duplicate groups crossing train/validation/test and report semantic-similarity checks.
- [ ] The release evaluation set covers every claimed class and segment; zero-support required classes fail evaluation.
- [ ] Training emits config, dependency snapshot, commit, dataset/split/label hashes, metrics, calibration, and a model card.
- [ ] Confidence calibration and abstention thresholds are stored in the model bundle and evaluated with selective-accuracy metrics.
- [ ] Torch and ONNX releases meet defined agreement and metric-delta tolerances.
- [ ] Model startup verifies manifest compatibility and artifact checksums.

### Tooling, CI, and documentation

- [ ] `pyproject.toml` configures project metadata, Ruff, Black, import sorting, mypy, and pytest/coverage.
- [ ] Pre-commit and `.editorconfig` are documented and pass on the repository.
- [ ] Fast CI runs formatting, lint, type checking, offline tests, coverage, OpenAPI contract checks, and SAM validation.
- [ ] Data/model CI jobs fail on unexpected skips and publish their reports.
- [ ] Runtime, training, development, and ONNX dependencies are reproducibly locked.
- [ ] README contains no `/ask` or `ask_service.py` reference and does not hard-code an incorrect active model.
- [ ] Deployment documentation matches `template.yaml` defaults automatically.
- [ ] Data card, model card, architecture doc, contributor guide, security policy, and operational runbooks exist and are linked from README.

## Implementation Tasks

### Workstream A: Baseline and urgent correctness

- [ ] A1. Capture OpenAPI, model/data hashes, current test output, current Torch evaluation, and image size as baseline artifacts.
- [ ] A2. Fix `ChatResponse` duplicate `model_config` and replace its incorrect prediction example.
- [ ] A3. Enforce final user message, add summary-context/negation red-flag tests, and decide emergency-state carry-forward semantics.
- [ ] A4. Replace client-facing raw exception detail with stable errors and request ids.
- [ ] A5. Add wellness bounds and resolve species-only/sparse-data scoring with product and veterinary approval.
- [ ] A6. Correct README, ONNX deployment defaults, active-model description, and remaining-checklist drift.

### Workstream B: Developer platform

- [ ] B1. Add `pyproject.toml` with dependency groups and tool configuration.
- [ ] B2. Add reproducible lock/export workflow for dev, runtime Torch, runtime ONNX, and training.
- [ ] B3. Add Ruff, Black, mypy, pytest-cov, pre-commit, and `.editorconfig`.
- [ ] B4. Add stable local/CI task commands.
- [ ] B5. Split CI into fast, data, model, and deployment jobs; make expected skips explicit.

### Workstream C: Runtime architecture

- [ ] C1. Implement validated Settings and production auth safeguards.
- [ ] C2. Define classifier and generator protocols plus domain result/error types.
- [ ] C3. Add model manifest loading, validation, labels, version, and backend metadata.
- [ ] C4. Extract predict, chat, and wellness use cases.
- [ ] C5. Split routers/schemas and add centralized error handlers.
- [ ] C6. Add liveness/readiness endpoints and privacy-safe structured telemetry.
- [ ] C7. Add Gemini timeout/retry/circuit policy, constrained output enums, and actual fallback outcomes.
- [ ] C8. Add generated-advice safety validation and reviewed deterministic fallback catalog.

### Workstream D: Data platform

- [ ] D1. Define canonical dataset schema, stable row id, provenance, and label-map validation.
- [ ] D2. Consolidate text/label normalization and deduplication helpers.
- [ ] D3. Refactor local, VetPetCare, PetEVAL, and synthetic sources into pinned adapters.
- [ ] D4. Add strict/best-effort source behavior and dataset build reports.
- [ ] D5. Add synthetic leakage, quality, similarity, and manual-review sampling gates.
- [ ] D6. Generate immutable grouped split manifests and full leakage reports.
- [ ] D7. Curate or acquire owner-language evaluation examples for every claimed class/species segment.

### Workstream E: Training and evaluation

- [ ] E1. Move training parameters to validated versioned configuration.
- [ ] E2. Complete reproducibility controls and record any nondeterministic backend limitations.
- [ ] E3. Correct and numerically test focal/class-weight behavior; remove ignored legacy flags.
- [ ] E4. Emit machine-readable metrics, confusion matrices, environment snapshot, split/data hashes, and model card.
- [ ] E5. Implement calibrated confidence and model-versioned abstention thresholds.
- [ ] E6. Rewrite evaluation for batching, explicit labels, segment reports, JSON output, and baseline comparison.
- [ ] E7. Add full safety suite and Torch/ONNX parity gates.
- [ ] E8. Fix regression artifact detection for both safetensors and PyTorch binary formats and run it in model CI.

### Workstream F: Release and operations

- [ ] F1. Select object storage/model registry and define immutable model-bundle naming and retention.
- [ ] F2. Publish/download bundles with checksum verification and rollback metadata.
- [ ] F3. Harden Docker users, permissions, dependency layers, health checks, and image-size budgets.
- [ ] F4. Add container build/scan and SAM validation to CI.
- [ ] F5. Strengthen API auth/secrets, log retention, concurrency, tracing, alarms, and rollback in SAM.
- [ ] F6. Measure ONNX cold starts and remove or justify the keep-warm schedule.
- [ ] F7. Add production, model-release, rollback, degraded-provider, and incident runbooks.

### Recommended sequencing

Start with A and B because they expose current behavior and prevent further drift. Execute C and the safety portion of E before structural folder moves. Complete D before retraining or claiming metric improvements. Finish E before F promotes a new model bundle. Treat clinical advice review and sparse wellness semantics as release-blocking decisions, not cleanup tasks.
