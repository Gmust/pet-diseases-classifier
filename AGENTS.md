# AGENTS.md

Rules for AI coding agents (Claude Code, Codex, Cursor, etc.) working in this repository.
Architecture, commands, and invariants live in [CLAUDE.md](CLAUDE.md) — read it first.
Human contributor policy is in [CONTRIBUTING.md](CONTRIBUTING.md) and applies to agents too.

## Before you finish

Every change must pass:

```bash
make quality      # lint + typecheck + contracts + test (coverage gate: 85%)
```

Formatting fixes: `make format`. Do not disable a lint rule, lower `--cov-fail-under`,
or add `# type: ignore` to make the gate pass — fix the code.

## Non-negotiable invariants

These are enforced by tests; breaking them fails the build.

1. **The classifier decides conditions, Gemini only writes prose.** Never let an LLM
   set or override a predicted condition, urgency, or specialist.
2. **The deterministic safety layer runs before any Gemini call.** A triggered red flag
   in `app/triage/safety.py` short-circuits: fixed emergency message, no LLM.
3. **`/chat` is stateless.** Do not add server-side session storage, caches keyed by
   user, or history persistence. The .NET caller owns state.
4. **Domain layering** (`tests/test_domain_boundaries.py`): `app/api/` → domain packages
   (`app/wellness/`, `app/triage/`, `app/feeding/`) → shared (`app/inference/`, `app/llm/`,
   `app/domain/`). Domains never import each other. Nothing under `app/` imports `ml_pipeline/`.
5. **250-line cap per module under `app/`** (`app/domain/conditions.py` exempt).
6. **New services go through `AppServices` and `build_services()`** in `app/bootstrap.py`,
   never constructed inside a route.

## Tests

- The suite runs fully offline. Never add a test that needs network, a Gemini key, or
  model weights. See `tests/conftest.py`.
- Add tests in the same change as the code. Behavioural change without a test is incomplete.
- Safety/advice changes need regression fixtures plus the veterinary approval metadata in
  `configs/advice-safety.json`.

## Never commit

Model binaries, datasets, `.env`, credentials, API keys, production payloads, real symptom
text, or generated caches (`.aws-sam/`, `graphify-out/`, `models/`).

## Ask before

- Deleting or reverting files, `git reset --hard`, `git checkout HEAD -- <file>`.
- Committing, pushing, opening PRs, or any AWS deploy (`sam deploy`).
- Adding a new runtime dependency — check whether stdlib or an installed package covers it.
- Changing the public API shape (`docs/api-reference.md`, `docs/openapi.sha256`) or the
  16-class condition map in `app/domain/conditions.py`.

## Scope discipline

- Smallest diff that solves the stated problem. No speculative abstractions, no
  interface with one implementation, no config for a value that never changes.
- Fix root causes, not the one call site named in the ticket — grep all callers first.
- Do not refactor, reformat, or "clean up" files outside the task.
- If a change touches API contracts or deployment, update `docs/` and the relevant ADR in
  `docs/adr/` in the same change (`make contracts` checks doc sync).

## Style

- Line length 100. black + isort (black profile) + ruff (`B,E4,E7,E9,F,I,SIM,UP`).
- Python 3.11–3.12. Annotate new public functions even though `disallow_untyped_defs` is off.
- Match the surrounding code: same naming, same comment density. Comments explain *why*.
