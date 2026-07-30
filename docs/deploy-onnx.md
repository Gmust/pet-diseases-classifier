# Deploy Runbook — ONNX on AWS Lambda

Ships the service as a quantized-ONNX Lambda image (no torch). Versus the old
torch image: smaller image, ~2-4 s cold start (was 15-30 s), and it runs at
**1 GB instead of 3 GB** — roughly **3× fewer GB-seconds** per request.

Prerequisites: Docker running, AWS CLI configured, AWS SAM CLI installed, and the
trained model present at `models/transformer_model/` (with `model.safetensors`).

---

## 1. Export the quantized ONNX model

Run where the trained weights live:

```bash
source .venv/bin/activate
# Build-time deps only. Pin onnxruntime 1.19.2 — newer versions reference
# torch.int4 and crash with torch<2.6 (which is what the .venv has).
pip install "optimum[exporters]" "onnxruntime==1.19.2"

python -m app.ml.export_onnx \
  --model-dir models/transformer_model \
  --output-dir models/transformer_model_onnx
```

This writes `models/transformer_model_onnx/` — `model.onnx`, `model_quantized.onnx`
(int8), plus `tokenizer.json` and `config.json`. int8 dynamic quantization
typically costs <1% macro-F1. (Export uses optimum's exporter + onnxruntime's
low-level `quantize_dynamic`; the *runtime* uses neither — see below.)

## 2. Parity check — BEFORE deploying

Confirm the ONNX model predicts the same as torch. Run the service locally on the
ONNX backend:

```bash
MODEL_BACKEND=onnx MODEL_PATH=models/transformer_model_onnx \
  USE_STATIC_EXPLANATIONS=true uvicorn app.main:app --port 8000
```

In another shell, run the smoke test and eyeball the predictions:

```bash
BASE=http://localhost:8000 bash scripts/smoke_test.sh
```

They should match the torch model (Digestive ~0.93, the sneezing case →
Respiratory, etc.). For a rigorous check, score the owner holdout on the ONNX
backend and compare to torch:

```bash
# torch baseline
python -m app.ml.evaluate --backend torch --model-dir models/transformer_model --data data/owner_eval.parquet
# onnx (the quantized model) — note --backend onnx
pip install onnxruntime==1.19.2 tokenizers==0.19.1   # if not already present
python -m app.ml.evaluate --backend onnx --model-dir models/transformer_model_onnx --data data/owner_eval.parquet
```

Top-1 accuracy should be within ~1% of torch. If it isn't, re-export with
`--no-quantize` (fp32 ONNX — larger but lossless) and re-check.

## 3. Build & deploy

```bash
sam build
sam deploy --guided     # first time: set stack name, region, and the params below
```

Parameters worth setting at deploy:
- `RuntimeSecretId` — a Secrets Manager secret containing non-empty `API_KEY`
  and `GEMINI_API_KEY` JSON fields. Production startup rejects missing auth.
- `FunctionMemory` — defaults to `1769` MB (approximately one full vCPU).
- `ReservedConcurrency` — defaults to `2` to bound memory/provider traffic.
- `ApiRateLimit` / `ApiBurstLimit` — gateway throttling (defaults 20 / 40).

The template already points at `Dockerfile.lambda.onnx` and sets
`MODEL_BACKEND=onnx`, `MODEL_PATH=models/transformer_model_onnx`,
`MemorySize=1769`, and `Timeout=60` by default.

Deployments publish a `live` alias using a 10%/5-minute canary. Lambda errors
automatically roll the deployment back. X-Ray is active, Lambda logs are retained
for 30 days, and CloudWatch alarms cover errors and sustained throttling.

After deploy, SAM prints the `ApiUrl` output.

## 4. Smoke test the live endpoint

```bash
BASE="https://<api-id>.execute-api.<region>.amazonaws.com/Prod" \
  API_KEY=<your key> bash scripts/smoke_test.sh
```

Check `GET /health` → 200, a `/predict` returns a sensible condition, and a
`/chat` turn returns a `mode`. The **first** call will be a cold start (a few
seconds); subsequent calls should be fast.

## 5. Rollback

If anything misbehaves, revert to the torch image without losing work:

```bash
# In template.yaml: set Dockerfile back to Dockerfile.lambda,
#   MemorySize/FunctionMemory to 3008, Timeout to 60,
#   remove MODEL_BACKEND, set MODEL_PATH=models/transformer_model
sam build && sam deploy
```

Or just `git revert` the deploy commit and redeploy. The torch `Dockerfile.lambda`
and `requirements.txt` are untouched, so the old path still works.

## Cold-start policy

The scheduled keep-warm invocation was removed. A five-run fresh-process
benchmark of the checked-in local ONNX artifact recorded a 178 ms median and
219 ms maximum model construction time; see
`docs/benchmarks/onnx-cold-start-2026-07-07.json`. This does not include Lambda
image-pull or platform initialization, so production p95 init duration must be
monitored in CloudWatch. Scheduled invocations do not guarantee reuse of the
container that receives user traffic. If the measured production cold-start SLO
is missed, use provisioned concurrency on the `live` alias rather than restoring
an unreliable keep-warm schedule.

## Notes

- **Image size:** dropping torch is the big win; expect the image to shrink from
  ~2.5 GB to well under 1 GB, which also speeds up `sam build`/push.
- **Gemini quota:** unrelated to this change, but remember `/chat` general-vs-health
  routing and tailored answers need `GEMINI_API_KEY` with available quota; without
  it the service still works via local fallbacks.
