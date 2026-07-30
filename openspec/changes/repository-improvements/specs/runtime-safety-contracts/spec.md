## ADDED Requirements

### Requirement: Stable and validated API contracts
The service SHALL validate bounded request data and SHALL return stable public errors without internal exception details while preserving existing endpoint paths and camelCase fields.

#### Scenario: Unexpected inference failure
- **WHEN** a classifier adapter raises an unexpected internal exception
- **THEN** the API returns a stable error code and request identifier, logs the exception server-side, and does not expose the exception text

#### Scenario: Invalid chat turn ordering
- **WHEN** the final chat message is not a non-empty user message
- **THEN** the API rejects the request without invoking the classifier or text generator

### Requirement: Deterministic emergency policy
The service MUST evaluate emergency policy before generated content and MUST produce emergency responses without invoking an external text generator.

#### Scenario: Emergency appears in relevant chat context
- **WHEN** a reviewed emergency phrase is present in the latest user message or bounded relevant symptom context
- **THEN** the response is deterministic, has emergency urgency, and uses reviewed emergency guidance

#### Scenario: Negated emergency phrase
- **WHEN** a reviewed negation fixture states that an emergency symptom is absent
- **THEN** the fixture produces the policy's reviewed non-emergency result

### Requirement: Bounded generated-content behavior
External text generation SHALL use constrained output models, bounded input/output sizes, explicit timeouts, and deterministic fallback outcomes.

#### Scenario: Generator timeout or malformed response
- **WHEN** the configured generator times out or returns output that fails validation
- **THEN** the use case returns a valid bounded fallback and records the actual fallback reason internally

### Requirement: Defensible wellness sufficiency
The wellness service SHALL validate numeric ranges and SHALL distinguish missing evidence from positive health evidence.

#### Scenario: Species-only wellness request
- **WHEN** a wellness request supplies a species but no meaningful tracked dimensions
- **THEN** the response does not represent the pet as confidently healthy solely from the absence of data

### Requirement: Runtime readiness
The service SHALL expose liveness separately from readiness, and readiness SHALL identify a successfully validated model version and backend without exposing sensitive paths or secrets.

#### Scenario: Model bundle unavailable
- **WHEN** the service has not loaded a valid model bundle
- **THEN** liveness may succeed but readiness fails with a stable non-sensitive reason
