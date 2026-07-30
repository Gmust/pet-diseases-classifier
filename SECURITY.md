# Security Policy

Do not open a public issue for a vulnerability. Report it privately through the
repository host's security advisory feature and include affected revision,
impact, reproduction, and suggested mitigation. Do not include real pet-owner
data or live credentials.

Production requires `X-API-Key`; API and Gemini keys are stored in AWS Secrets
Manager. Rotate a suspected key immediately, inspect CloudWatch/X-Ray metadata
without copying request content, and follow `docs/runbooks/incident-response.md`.
Supported code is the deployed `main` revision and currently promoted immutable
model version.
