# Contributing

Use Python 3.11 and the locked environment:

```bash
uv sync --frozen --extra dev
uv run pre-commit install
uv run pytest -q
make lint typecheck contracts
```

Run data/model tests when their areas change. Model changes require immutable
manifests, cards, evaluation, safety, and parity evidence; use
`.github/workflows/model-release.yml`. Update OpenAPI/deployment docs and ADRs in
the same change as their source. Do not commit model binaries, datasets,
credentials, production payloads, prompts, symptom text, or generated caches.

Safety policy/advice changes require reviewed regression fixtures and the
veterinary approval metadata in `configs/advice-safety.json`.
