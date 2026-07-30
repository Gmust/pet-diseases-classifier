## ADDED Requirements

### Requirement: Canonical and traceable dataset builds
Every training or evaluation dataset build MUST use one canonical row contract and MUST emit source revisions, licenses, transformations, rejection counts, and a content fingerprint.

#### Scenario: Requested external source fails in strict mode
- **WHEN** a configured dataset source cannot be loaded during a strict build
- **THEN** the build fails and does not publish a complete dataset manifest

### Requirement: Immutable leakage-resistant splits
Training, validation, and test membership SHALL be generated before training, persisted by stable row id, and grouped to prevent known duplicate families crossing splits.

#### Scenario: Duplicate family crosses proposed splits
- **WHEN** exact or configured near-duplicate grouping places related rows in different splits
- **THEN** split publication fails or requires an explicit reviewed exception

### Requirement: Reproducible training bundle
Every training run SHALL emit validated configuration, source commit, environment versions, random seed, dataset/split/label fingerprints, machine-readable metrics, and model artifacts.

#### Scenario: Training completes successfully
- **WHEN** a model finishes training
- **THEN** a self-describing model bundle and model card are produced and pass bundle validation

### Requirement: Calibrated release evaluation
Model promotion MUST use explicit-label, full-claim evaluation that reports per-class metrics, calibration, abstention coverage, safety behavior, and relevant data segments.

#### Scenario: Required class has no evaluation support
- **WHEN** a release evaluation set has zero examples for a class claimed by the product
- **THEN** release evaluation fails rather than silently averaging the class away

### Requirement: Versioned backend parity
Torch and ONNX artifacts for the same release SHALL identify the same model version and SHALL satisfy defined prediction and metric-delta tolerances.

#### Scenario: ONNX export exceeds parity tolerance
- **WHEN** ONNX evaluation differs from the Torch baseline beyond a configured threshold
- **THEN** the ONNX artifact is not promoted

### Requirement: Verified artifact distribution
Runtime images SHALL retrieve or include model artifacts by immutable version and SHALL verify manifest compatibility and checksums before readiness succeeds.

#### Scenario: Model checksum mismatch
- **WHEN** a model file does not match its release manifest
- **THEN** service startup or readiness fails and the artifact is not served
