# Model Release Runbook

1. Publish `model-bundle-<version>` from a trusted workflow. It must contain
   `transformer_model/`, `transformer_model_onnx/`, both immutable manifests,
   `transformer_model/MODEL_CARD.md`, `DATA_CARD.md`, and the parity holdout
   `owner_eval_real.parquet` at the bundle root (the images copy only the model
   directories, so it never ships).
2. Record the producing workflow run id, then run:

   ```bash
   gh workflow run model-release.yml -f model_version=<version> -f artifact_run_id=<run-id>
   gh run list --workflow=model-release.yml --limit=1
   ```

3. Require green model regression, safety, parity, container scan/size, and
   release-evidence jobs. Download and retain the evidence artifact.
4. Deploy the exact image digest through SAM. Confirm `/health/ready` returns
   `ready` with the expected backend/version (a model that misclassifies the
   built-in canary fails startup and logs `model_canary_failed` — roll back) and run
   `BASE=<url> API_KEY=<key> scripts/smoke_test.sh`. Parity on a CPU other than
   the runtime's is not evidence (ADR 0004): check x86 with AVX2, e.g. Docker
   `--platform linux/amd64` with Rosetta emulation off.
5. Watch Lambda Errors, Throttles, duration, fallback, abstention, and red-flag
   events through the post-deployment observation window: 15 minutes of
   production traffic after the smoke test passes. The deployment is healthy when
   the Errors and Throttles alarms stay `OK` for the whole window and fallback,
   abstention, and red-flag rates stay at their pre-deploy baseline. If an alarm
   fires, perform the operator-driven procedure in `docs/runbooks/rollback.md`;
   the current stack does not configure automatic rollback.
