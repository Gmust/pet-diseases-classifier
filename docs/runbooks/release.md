# Model Release Runbook

1. Publish `model-bundle-<version>` from a trusted workflow. It must contain
   `transformer_model/`, `transformer_model_onnx/`, both immutable manifests,
   `transformer_model/MODEL_CARD.md`, and `DATA_CARD.md`.
2. Record the producing workflow run id, then run:

   ```bash
   gh workflow run model-release.yml -f model_version=<version> -f artifact_run_id=<run-id>
   gh run list --workflow=model-release.yml --limit=1
   ```

3. Require green model regression, safety, parity, container scan/size, and
   release-evidence jobs. Download and retain the evidence artifact.
4. Deploy the exact image digest through SAM. Confirm `/health/ready` reports
   the expected backend/version and run `BASE=<url> API_KEY=<key> scripts/smoke_test.sh`.
5. Watch Lambda Errors, Throttles, duration, fallback, abstention, and red-flag
   events through the canary window. An Errors alarm triggers automatic rollback.
