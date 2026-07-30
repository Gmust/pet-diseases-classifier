## ADDED Requirements

### Requirement: Central Python project configuration
The repository SHALL define supported Python versions, dependency profiles, formatting, linting, typing, pytest, and coverage configuration from a documented central project configuration.

#### Scenario: Developer runs the repository quality command
- **WHEN** a developer runs the documented quality command in a supported environment
- **THEN** formatting checks, linting, type checking, and fast tests execute with the same configuration as CI

### Requirement: Reproducible dependency profiles
Runtime Torch, runtime ONNX, development, training, and export environments SHALL be independently reproducible and validated for dependency conflicts.

#### Scenario: Runtime ONNX environment is built
- **WHEN** the ONNX dependency profile is installed from its lock or hash-verified export
- **THEN** it excludes Torch and passes dependency validation

### Requirement: Explicit CI evidence
CI SHALL distinguish fast, data, model, and deployment checks, and any job claiming to validate a capability MUST fail if its required tests unexpectedly skip.

#### Scenario: Pull request changes ordinary Python code
- **WHEN** CI runs the fast pull-request workflow
- **THEN** formatting, linting, typing, offline tests, coverage, API contract checks, and deployment-template validation report explicit results

#### Scenario: Model release workflow runs
- **WHEN** immutable model artifacts are available to the model workflow
- **THEN** real-model regression, release evaluation, and backend parity tests run without self-skipping

### Requirement: Local pre-commit feedback
The repository SHALL provide editor defaults and pre-commit checks for formatting, linting, basic file validity, and accidental secret inclusion.

#### Scenario: Developer commits a malformed or secret-bearing change
- **WHEN** configured pre-commit hooks detect the violation
- **THEN** the commit is blocked with an actionable local error
