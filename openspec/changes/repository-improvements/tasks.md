## 1. Baseline and immediate correctness

- [x] 1.1 Record current API, test, model, dataset, and evaluation baseline in a tracked report
- [x] 1.2 Fix the duplicate and incorrect `ChatResponse` schema configuration and add an OpenAPI regression test
- [x] 1.3 Enforce that the final chat message is from the user and add API/unit regression tests
- [x] 1.4 Replace client-facing internal prediction exceptions with stable errors and add failure-path tests
- [x] 1.5 Add bounded emergency-context evaluation with reviewed negation and prior-context tests
- [x] 1.6 Add wellness numeric bounds and explicit sparse-data semantics after resolving the documented product decision
- [x] 1.7 Correct README, API/deployment docs, active-model description, and stale checklist claims

## 2. Developer quality gates

- [x] 2.1 Add `pyproject.toml` with Python metadata and Ruff, Black, mypy, pytest, and coverage configuration
- [x] 2.2 Add `.editorconfig`, pre-commit hooks, and documented stable quality commands
- [x] 2.3 Create reproducible development, Torch runtime, ONNX runtime, training, and export dependency profiles
- [x] 2.4 Split CI into fast and data jobs with explicit no-unexpected-skip and coverage gates
- [x] 2.5 Add OpenAPI contract and SAM template validation to fast CI

## 3. Runtime boundaries and generation safety

- [x] 3.1 Introduce validated runtime Settings and production authentication safeguards
- [x] 3.2 Define typed classifier and generator protocols with public backend/model metadata
- [x] 3.3 Validate model labels, files, top-k bounds, manifest compatibility, and checksums
- [x] 3.4 Extract predict, chat, and wellness application use cases behind compatible routes
- [x] 3.5 Add stable public error handlers plus separate liveness and readiness endpoints
- [x] 3.6 Add generator timeout/retry policy, constrained modes, bounded payloads, and actual fallback telemetry
- [x] 3.7 Add generated/static advice safety validation and veterinary approval metadata

## 4. Reproducible data lifecycle

- [x] 4.1 Define the canonical dataset row schema, provenance fields, stable row ids, and label-map validation
- [x] 4.2 Consolidate text/label normalization, rejection accounting, and content-based deduplication
- [x] 4.3 Refactor local, VetPetCare, PetEVAL, and synthetic inputs into pinned strict/best-effort source adapters
- [x] 4.4 Add synthetic diagnosis-leakage, similarity, provenance, and manual-review quality gates
- [x] 4.5 Generate immutable grouped split manifests and fail on cross-split leakage
- [x] 4.6 Establish owner-language evaluation coverage for every class/species segment claimed by the product

## 5. Reproducible training and evaluation

- [x] 5.1 Drive training from validated configuration and emit commit/environment/data/split/label metadata
- [x] 5.2 Correct and numerically test focal/class-weight behavior and remove ignored legacy flags
- [x] 5.3 Emit machine-readable metrics, confusion matrices, bundle validation, and a generated model card
- [x] 5.4 Calibrate confidence and store versioned abstention thresholds with selective-accuracy evidence
- [x] 5.5 Rewrite evaluation for batching, explicit labels, segment metrics, JSON output, and baseline comparison
- [x] 5.6 Add full safety evaluation and Torch/ONNX parity release gates
- [x] 5.7 Fix model regression artifact detection and run real-model checks without skips in model CI

## 6. Artifact release, deployment, and operations

- [x] 6.1 Define and implement immutable model-bundle publishing, retrieval, retention, and rollback metadata
- [x] 6.2 Verify artifact checksums during image build/startup and expose model version in readiness and logs
- [x] 6.3 Harden container users, permissions, dependency layers, health checks, and image-size budgets
- [x] 6.4 Add model, parity, container build/scan, and release workflows backed by immutable artifacts
- [x] 6.5 Harden SAM authentication, secret references, log retention, concurrency, tracing, alarms, and rollback
- [x] 6.6 Measure ONNX cold starts and remove or justify the keep-warm schedule

## 7. Documentation and cleanup

- [x] 7.1 Add synchronized architecture and decision records
- [x] 7.2 Add generated data card and model card release documentation
- [x] 7.3 Add contribution, security, privacy/data-handling, release, rollback, degraded-provider, and incident guidance
- [x] 7.4 Add CI checks that detect API and deployment documentation drift
- [x] 7.5 Remove deprecated ignored CLI arguments, predictor compatibility kwargs, unused joblib dependencies, and stale checklist content after migration
