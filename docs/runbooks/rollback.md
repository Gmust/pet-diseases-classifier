# Model Rollback Runbook

1. Read the current channel/release manifest and select its explicit
   `rollback_version`; never rebuild or select an unversioned `latest` image.
2. Retrieve and verify that version:

   ```bash
   python -m app.inference.model_registry /path/to/registry/releases/<rollback-version>
   ```

3. Redeploy the image digest previously recorded for that immutable version, or
   move the Lambda `live` alias to the previous healthy version.
4. Verify `/health/ready` reports the rollback version, execute the smoke test,
   and confirm Errors plus inference-failure metrics return to baseline.
5. Preserve the faulty artifact/evidence, block its channel promotion, and open
   an incident. Do not overwrite or delete the release.
