# Repository Status

This historical checklist was retired on 2026-07-07 after the
`repository-improvements` OpenSpec implementation replaced it with executable
quality, model-release, deployment, and documentation-drift gates.

Use these maintained sources instead:

- `openspec/changes/repository-improvements/tasks.md` for implementation status.
- `docs/architecture.md` and `docs/adr/` for current design decisions.
- `docs/data-card.md` and `docs/model-card.md` for release transparency.
- `docs/runbooks/` for release, rollback, provider degradation, and incidents.
- GitHub issues or a new OpenSpec change for unresolved product/model work.

The current local ONNX candidate is not promotable because the strict parity
gate fails; see `docs/baseline-2026-07-07.md` and run
`python -m ml_pipeline.release_gates parity` for fresh evidence.
