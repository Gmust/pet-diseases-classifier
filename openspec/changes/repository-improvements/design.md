## Context

The current repository combines a FastAPI service, Torch/ONNX inference, deterministic triage rules, Gemini-generated content, rule-based wellness scoring, offline data preparation, transformer training/evaluation, and AWS Lambda deployment. These parts work, but their boundaries and release evidence are mostly implicit. The implementation must remain deployable while safety and reproducibility are strengthened incrementally.

Constraints include preserving current endpoint paths and camelCase contracts, retaining offline fallbacks, keeping large model/data binaries out of Git, supporting lightweight local tests without Torch, and avoiding a repository-wide move before compatibility tests exist.

## Goals / Non-Goals

**Goals:**

- Make safety-relevant behavior explicit, validated, and directly tested.
- Introduce typed runtime ports and validated configuration without a flag-day rewrite.
- Make data, training, evaluation, model export, and deployment artifacts traceable.
- Establish fast PR quality gates plus explicit data/model/release gates.
- Keep documentation synchronized with executable configuration and schemas.

**Non-Goals:**

- Replace FastAPI, Gemini, Torch, ONNX, AWS SAM, or the classifier architecture in the first increment.
- Add chat persistence to this service.
- Commit large model or dataset binaries to Git.
- Claim clinical validation or diagnostic use.
- Introduce breaking API changes under this change.

## Decisions

### Implement vertical, reversible slices

Each task SHALL change one observable behavior or one quality gate and include tests or validation. Safety and correctness fixes precede folder moves. Compatibility imports and current routes remain until equivalent contract tests pass.

Alternative considered: move immediately to the target folder structure. Rejected because a large mechanical diff would obscure safety behavior and make rollback harder.

### Separate transport, application, domain, and infrastructure over time

FastAPI routers will eventually translate HTTP only; application use cases will orchestrate classifier, safety, metadata, and generation ports; domain modules will own deterministic policy; infrastructure adapters will own Torch, ONNX, Gemini, configuration, and logging.

Alternative considered: retain `app/main.py` as the permanent orchestration layer. Rejected because concrete backend coupling and global application state already prevent meaningful static typing and focused tests.

### Use structural typing for replaceable runtime services

Classifier and text-generator contracts will use Python `Protocol` definitions and immutable result types. Runtime adapters will publish public metadata instead of evaluation code reading private members.

Alternative considered: a deep inheritance hierarchy. Rejected because adapters share behavior contracts but not implementation mechanics.

### Treat safety as deterministic policy around probabilistic components

Emergency escalation, abstention policy, request sufficiency, and forbidden generated advice remain outside the classifier and LLM. Emergency responses do not call Gemini. Safety fixtures are release gates independent of aggregate classifier accuracy.

Alternative considered: ask Gemini to perform all routing and safety checks. Rejected because availability, prompt sensitivity, and nondeterminism are unacceptable for emergency behavior.

### Treat data and models as immutable release inputs

Dataset builds, splits, training runs, exported backends, and model releases will emit manifests with hashes and provenance. Git tracks schemas/manifests and reports; an external immutable store holds large payloads.

Alternative considered: continue relying on ignored local folders. Rejected because the deployed model cannot currently be reconstructed or verified from the repository.

### Split CI by cost and evidence

Fast jobs run formatting, linting, typing, offline tests, coverage, contract checks, and SAM validation on every PR. Data, model, parity, and image jobs run when their immutable artifacts are available and SHALL fail on unexpected skips.

Alternative considered: install the full ML stack in every PR job. Rejected because it creates high latency and cost without improving feedback for unrelated changes.

### Keep one source of truth per configuration value

Runtime defaults live in a validated Settings model; project/tooling configuration lives in `pyproject.toml`; deployment overrides live in SAM and are checked against documentation. Handwritten documents do not independently redefine defaults.

## Risks / Trade-offs

- [Risk] Incremental boundaries temporarily create compatibility layers → Keep them small, mark removal tasks, and test both old and new imports.
- [Risk] Stricter request validation rejects existing payloads → Preserve aliases, snapshot real contracts, document bounds, and avoid breaking changes without a separate proposal.
- [Risk] Correct wellness sufficiency lowers or withholds existing scores → Version the scoring policy and require product/veterinary approval before changing the response contract.
- [Risk] Calibration increases abstention frequency → Measure selective accuracy and expose coverage as a product metric.
- [Risk] Model registry work delays releases → Begin with immutable object storage plus a small manifest rather than a platform migration.
- [Risk] Expanded CI becomes slow → Keep fast PR and artifact-backed model/release jobs separate.
- [Risk] Medical advice review blocks engineering → Record a named owner and make approval state visible in the model/advice artifact.

## Migration Plan

1. Capture current API, test, model, dataset, and evaluation baselines.
2. Apply isolated correctness fixes to schemas, errors, chat validation, telemetry, wellness validation, and documentation.
3. Add centralized tooling and CI gates without changing runtime behavior.
4. Introduce Settings and runtime protocols behind existing service construction.
5. Extract application use cases while retaining route and import compatibility.
6. Standardize dataset contracts and immutable splits before any model retraining.
7. Emit model bundles, calibration, full evaluation, and ONNX parity reports.
8. Publish immutable artifacts and harden deployment/release automation.
9. Remove documented compatibility layers after their deprecation tasks complete.

Rollback for each slice is the preceding code path or artifact version. Model releases remain independently rollbackable by immutable version. No task may require deleting user work or local model/data artifacts.

## Open Questions

- What minimum tracked dimensions permit a wellness band, and should insufficient data add a response state or retain the current response shape?
- Who owns veterinary approval for static and generated home advice?
- Which immutable artifact store and retention policy will be used for model bundles?
- Which full-class owner-language dataset can support release claims without synthetic-only evaluation?
- What calibrated safety and quality thresholds define the first trusted model baseline?
