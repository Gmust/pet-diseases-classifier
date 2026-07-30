## Why

The service is functional, but safety-critical behavior, model reproducibility, and CI assurance rely on implicit conventions that are already drifting from documentation and deployment configuration. The repository needs staged, measurable improvements so runtime behavior, model releases, and developer workflows can be changed without weakening the current API or safety fallbacks.

## What Changes

- Harden request validation, emergency-context handling, public errors, readiness, generated-content fallbacks, and sparse wellness semantics.
- Introduce validated configuration and typed runtime boundaries for Torch/ONNX inference and text generation.
- Establish Python formatting, linting, type-checking, coverage, pre-commit, dependency locking, and representative CI jobs.
- Standardize dataset contracts, provenance, preprocessing, immutable splits, training metadata, calibrated confidence, and release evaluation.
- Version and validate model bundles, including Torch/ONNX parity and artifact checksums.
- Correct stale API, architecture, model, and deployment documentation and add maintained model/data cards and runbooks.
- Preserve existing endpoint paths and camelCase response contracts during staged migration; any future breaking API change requires a separate proposal.

## Capabilities

### New Capabilities

- `runtime-safety-contracts`: Validated API inputs, deterministic emergency handling, stable errors, readiness, bounded LLM behavior, and defensible wellness data-sufficiency semantics.
- `developer-quality-gates`: Central Python project configuration, formatting, linting, typing, coverage, pre-commit, dependency profiles, and explicit CI jobs.
- `reproducible-ml-lifecycle`: Canonical dataset provenance, immutable splits, reproducible training, calibrated evaluation, model manifests, and backend parity gates.
- `operational-documentation`: Synchronized API/deployment documentation, architecture decisions, model/data cards, contributor guidance, and operational runbooks.

### Modified Capabilities

No existing main specifications are present under `openspec/specs/`.

## Impact

- Runtime: `app/main.py`, `app/schemas.py`, `app/services/*`, `app/ml/predictor.py`, `app/ml/onnx_predictor.py`, `app/observability.py`, and Lambda wiring.
- ML lifecycle: dataset merge/preparation scripts, `app/ml/train.py`, `app/ml/evaluate.py`, ONNX export, ignored data/model artifacts, and release workflows.
- Tooling: dependency files, new `pyproject.toml`, pre-commit/editor configuration, tests, and `.github/workflows/ci.yml`.
- Operations: Dockerfiles, `template.yaml`, model artifact distribution, readiness, metrics, secret handling, and rollback procedures.
- Documentation: `README.md`, `docs/*`, generated OpenAPI, and the repository improvement specification used as planning input.
