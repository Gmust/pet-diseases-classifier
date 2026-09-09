# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Agent-facing rules (scope, guardrails, what to ask before doing) live in `AGENTS.md`.

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
make quality      # lint + typecheck + contracts + test
make lint         # ruff check + black --check + isort --check-only
make typecheck    # mypy app
make test         # pytest
make format       # black + isort (write)
make pre-commit   # all pre-commit hooks on the repo
```

Run a single test: `python -m pytest tests/test_api.py::test_name -v`

Run the server: `uvicorn app.main:app --reload --port 8000` (Swagger at `/docs`).

Coverage gate is `fail_under = 85` (branch coverage over all of `app`, omitting only
`app/inference/onnx_predictor.py` and `app/lambda_handler.py`, which cannot run offline).
Scope is declared in `pyproject.toml`, so a new module is gated without touching CI.

## Test suite runs fully offline

`tests/conftest.py` makes the suite pass with no torch, no model weights, and no Gemini key:
- `Predictor.from_paths` is monkeypatched to a `FakePredictor` (torch is lazy-imported so it never loads).
- `GEMINI_API_KEY` and `API_KEY` are popped from the env before the app imports, so services use deterministic local fallbacks and auth is disabled.
- `app.state.services` is reset per test to rebuild with that test's fake predictor.

Do not add tests that require live model weights or network. `test_classifier_regression.py` self-skips when no weights are present (that's why CI installs only `requirements-dev.txt`).

## Architecture — key invariants

**The classifier is the sole decision-maker for conditions.** Gemini only generates human-facing prose (explanations, advice, chat answers, wellness narrative). Gemini can never override or set the predicted condition. Preserve this separation when editing services.

**Deterministic safety layer wins over Gemini.** `app/triage/safety.py` holds red-flag/emergency rules and abstention thresholds. In both `/predict` and `/chat`, a triggered red flag short-circuits: it overrides urgency, skips Gemini, and returns a fixed emergency message + `EMERGENCY_HOME_ADVICE`. Keep red-flag checks ahead of any Gemini call.

**`/chat` is stateless.** The service stores nothing. The caller (the .NET backend) persists message history and the rolling `symptomSummary` and replays them every turn. Each turn: red-flag check → local classifier (`predict_top_k`) → one Gemini call that both routes (`general` vs `health`) and writes the answer. Response branches on `mode` (`general` | `health` | `emergency`). See `docs/dotnet-chat-integration.md` and `docs/api-reference.md`.

**Two inference backends, selected by `MODEL_BACKEND` env.** `torch` (default, `app/inference/predictor.py`) or `onnx` (quantized, `app/inference/onnx_predictor.py`). Both expose `predict` / `predict_top_k` and are constructed in `build_services()` in `app/bootstrap.py`. The ONNX path is imported lazily.

**Service loading is Lambda-aware.** `build_services()` in `app/bootstrap.py` loads model + services; `ensure_services()` in `app/api/__init__.py` caches idempotently on `app.state.services`. On Lambda the model loads at INIT (via `app/lambda_handler.py`) so warm containers skip the load; locally it loads in the FastAPI `lifespan`. When adding a service, wire it through `AppServices` and `build_services()`, not into a route.

**Packages are organised by domain, and the layering is enforced by tests.** `app/api/` (transport) →
`app/wellness/`, `app/triage/`, `app/feeding/` (one per domain, each owning its schemas) →
`app/inference/`, `app/llm/`, `app/domain/` (shared). Domain packages never import each other, shared
packages never import a domain, and nothing under `app/` imports `ml_pipeline/` — both Lambda images
copy `app/` only. No module under `app/` may exceed 250 lines (`app/domain/conditions.py`, a data map,
is exempt). `tests/test_domain_boundaries.py` fails the build on any violation.

**Condition metadata is a static map.** `app/domain/conditions.py` maps each of the 16 condition classes → urgency / specialist / disease category / home advice. The 16 classes come from consolidating 23 raw labels via `data/label_map.json`. Enum values shared across domains live in `app/domain/enums.py`; the class list is in the README table.

## Config (env vars)

`GEMINI_API_KEY`, `GEMINI_MODEL` (default `gemini-2.5-flash-lite`), `API_KEY` (X-API-Key header auth; unset = auth disabled), `MODEL_PATH`, `MODEL_BACKEND` (`torch`|`onnx`), `LOW_CONFIDENCE_THRESHOLD` (default 0.65), `USE_STATIC_EXPLANATIONS` (skip Gemini on `/predict`), `ROOT_PATH` (set to `/Prod` on Lambda behind API Gateway).

Auth uses `hmac.compare_digest`. `GET /health` is always open.

## Training pipeline (`ml_pipeline/`, needs `--extra train`)

`generate_synthetic.py` → `fetch_and_merge.py` → `prepare_dataset.py` → `train.py` → `evaluate.py`
(all under `ml_pipeline/`, run as `python -m ml_pipeline.<module>`). Data lands in `data/*.parquet`. Owner-language holdout is built by `prepare_dataset.py`; evaluate model changes against it before promoting. See README "Training Pipeline".

## Deploy

AWS SAM: `sam build && sam deploy` (first time `--guided`). Template `template.yaml`, config `samconfig.toml`. Lambda Dockerfiles: `Dockerfile.lambda` (torch) and `Dockerfile.lambda.onnx`.

## Conventions

- Line length 100 (black, isort black profile, ruff). Ruff lint selects `B,E4,E7,E9,F,I,SIM,UP`.
- mypy runs on `app` with `check_untyped_defs`; `disallow_untyped_defs` is off (annotate new public functions anyway).
- Python 3.11–3.12.
