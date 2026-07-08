## ADDED Requirements

### Requirement: Accurate repository entry documentation
The README SHALL describe only existing endpoints, modules, setup profiles, active model metadata sources, and verified development commands.

#### Scenario: Removed endpoint or module
- **WHEN** an endpoint or module is removed from the codebase
- **THEN** README and linked API documentation contain no active usage instructions for it

### Requirement: Synchronized API and deployment documentation
Documented API schemas and deployment defaults SHALL be checked against generated OpenAPI, validated settings, and deployment templates.

#### Scenario: Deployment default changes
- **WHEN** a resource or timeout default changes in `template.yaml`
- **THEN** CI detects any conflicting documented default

### Requirement: Model and data transparency
Every promoted model release SHALL publish a model card and data card describing provenance, intended use, limitations, evaluation coverage, synthetic-data contribution, and safety evidence.

#### Scenario: Model release lacks governance artifacts
- **WHEN** a release candidate has no matching model card or data card
- **THEN** the release workflow rejects promotion

### Requirement: Maintained contributor and operational guidance
The repository SHALL document contribution quality gates, security reporting, sensitive-data rules, release, rollback, degraded-provider, and incident procedures.

#### Scenario: Operator rolls back a model
- **WHEN** a deployed model is identified as faulty
- **THEN** the runbook provides an executable immutable-version rollback and a verification procedure
