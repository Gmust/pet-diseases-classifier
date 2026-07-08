# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FastAPI microservice with three AI endpoints for a pet-care app: `/predict` (symptom classification), `/chat` (unified stateless triage + general Q&A), `/wellness` (rule-based wellness scoring). Runs locally via uvicorn and on AWS Lambda via SAM (Mangum adapter).

## Commands

Setup uses `uv` (lockfile `uv.lock` pins all profiles). Optional-dependency extras:

```bash
uv sync --extra dev --extra torch-runtime   # local API dev + full local tests
uv sync --extra onnx-runtime                # ONNX inference only
uv sync --extra train                        # training pipeline
uv sync --extra export                       # ONNX export tooling
```

Quality gates (same entry points as CI — Makefile wraps these):

```bash
make quality      # lint + typecheck + test
make lint         # ruff check + black --check + isort --check-only
make typecheck    # mypy app
make test         # pytest
make format       # black + isort (write)
make pre-commit   # all pre-commit hooks on the repo
```

Run a single test: `python -m pytest tests/test_api.py::test_name -v`

Run the server: `uvicorn app.main:app --reload --port 8000` (Swagger at `/docs`).

Coverage gate is `fail_under = 85` (branch coverage over `app`, `app/ml/train.py` omitted).

## Test suite runs fully offline

`tests/conftest.py` makes the suite pass with no torch, no model weights, and no Gemini key:
- `Predictor.from_paths` is monkeypatched to a `FakePredictor` (torch is lazy-imported so it never loads).
- `GEMINI_API_KEY` and `API_KEY` are popped from the env before the app imports, so services use deterministic local fallbacks and auth is disabled.
- `app.state.services` is reset per test to rebuild with that test's fake predictor.

Do not add tests that require live model weights or network. `test_classifier_regression.py` self-skips when no weights are present (that's why CI installs only `requirements-dev.txt`).

## Architecture — key invariants

**The classifier is the sole decision-maker for conditions.** Gemini only generates human-facing prose (explanations, advice, chat answers, wellness narrative). Gemini can never override or set the predicted condition. Preserve this separation when editing services.

**Deterministic safety layer wins over Gemini.** `app/services/triage_safety.py` holds red-flag/emergency rules and abstention thresholds. In both `/predict` and `/chat`, a triggered red flag short-circuits: it overrides urgency, skips Gemini, and returns a fixed emergency message + `EMERGENCY_HOME_ADVICE`. Keep red-flag checks ahead of any Gemini call.

**`/chat` is stateless.** The service stores nothing. The caller (the .NET backend) persists message history and the rolling `symptomSummary` and replays them every turn. Each turn: red-flag check → local classifier (`predict_top_k`) → one Gemini call that both routes (`general` vs `health`) and writes the answer. Response branches on `mode` (`general` | `health` | `emergency`). See `docs/dotnet-chat-integration.md` and `docs/api-reference.md`.

**Two inference backends, selected by `MODEL_BACKEND` env.** `torch` (default, `app/ml/predictor.py`) or `onnx` (quantized, `app/ml/onnx_predictor.py`). Both expose `predict` / `predict_top_k` and are constructed in `build_services()` in `app/main.py`. The ONNX path is imported lazily.

**Service loading is Lambda-aware.** `build_services()` loads model + services; `ensure_services()` caches idempotently on `app.state.services`. On Lambda the model loads at INIT (via `app/lambda_handler.py`) so warm containers skip the load; locally it loads in the FastAPI `lifespan`. When adding a service, wire it through `AppServices` and `build_services()`, not into a route.

**Condition metadata is a static map.** `app/ml/condition_metadata.py` maps each of the 16 condition classes → urgency / specialist / disease category / home advice. The 16 classes come from consolidating 23 raw labels via `data/label_map.json`. Enum values and the class list live in `app/schemas.py` and the README table.

## Config (env vars)

`GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-2.5-flash-lite`), `API_KEY` (X-API-Key header auth; unset = auth disabled), `MODEL_PATH`, `MODEL_BACKEND` (`torch`|`onnx`), `LOW_CONFIDENCE_THRESHOLD` (default 0.65), `USE_STATIC_EXPLANATIONS` (skip Gemini on `/predict`), `ROOT_PATH` (set to `/Prod` on Lambda behind API Gateway).

Auth uses `hmac.compare_digest`. `GET /health` is always open.

## Training pipeline (`app/ml/`, needs `--extra train`)

`generate_synthetic.py` → `fetch_and_merge.py` → `prepare_dataset.py` → `train.py` → `evaluate.py`. Data lands in `data/*.parquet`. Owner-language holdout is built by `prepare_dataset.py`; evaluate model changes against it before promoting. See README "Training Pipeline".

## Deploy

AWS SAM: `sam build && sam deploy` (first time `--guided`). Template `template.yaml`, config `samconfig.toml`. Lambda Dockerfiles: `Dockerfile.lambda` (torch) and `Dockerfile.lambda.onnx`.

## Conventions

- Line length 100 (black, isort black profile, ruff). Ruff lint selects `B,E4,E7,E9,F,I,SIM,UP`.
- mypy runs on `app` with `check_untyped_defs`; `disallow_untyped_defs` is off (annotate new public functions anyway).
- Python 3.11–3.12.
