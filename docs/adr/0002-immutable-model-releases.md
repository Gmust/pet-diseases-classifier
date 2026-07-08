# ADR 0002: Publish immutable model releases

- Status: Accepted
- Date: 2026-07-07

## Decision

Each Torch/ONNX release uses one explicit version and a write-once bundle with a
`release-manifest.json` containing payload hashes, sizes, retention intent, and
the previous rollback version. Mutable channel metadata may point to releases;
release payloads are never overwritten.

## Consequences

Images and startup reject corrupted release bundles. Rollback selects an existing
version instead of rebuilding weights. Large payloads remain outside Git, while
manifests and release evidence stay traceable.
