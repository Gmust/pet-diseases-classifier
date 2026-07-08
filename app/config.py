"""
Validated runtime configuration.

Single source of truth for env-derived defaults (design.md: "Keep one source
of truth per configuration value"). Replaces the scattered `os.getenv(...)`
calls that used to live in `app/main.py`.

`Settings()` is re-instantiated (not cached) each time `build_services()`
runs, so it re-reads the environment the same way the old `os.getenv` calls
did — this keeps the Lambda keep-warm path and the test suite (which mutates
env vars per test via monkeypatch) working unchanged.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    # No `env_file` here: `app.main` calls `load_dotenv()` once at import time,
    # which populates `os.environ` directly. Settings then reads only
    # `os.environ`, so tests that monkeypatch/delenv a var see that change
    # immediately (an `env_file` fallback would bypass monkeypatch.delenv and
    # keep reading the real `.env` value underneath it).
    model_config = SettingsConfigDict(extra="ignore")

    environment: Environment = Environment.DEVELOPMENT

    model_path: str = "models/transformer_model"
    model_backend: Literal["torch", "onnx"] = "torch"

    gemini_api_key: str | None = None
    # Comma-separated additional keys for quota rotation, e.g. "key2,key3,key4,key5".
    # `resolved_gemini_api_keys` prepends `gemini_api_key` so a single var also works.
    gemini_api_keys: str | None = None
    gemini_model: str = "gemini-2.5-flash-lite"

    api_key: str | None = None

    low_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    use_static_explanations: bool = False

    root_path: str = ""
    log_level: str = "INFO"

    @field_validator("gemini_api_key", "api_key", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def resolved_gemini_api_keys(self) -> list[str]:
        """Ordered, de-duplicated key list: primary key first, then rotation keys."""
        extra = [k.strip() for k in (self.gemini_api_keys or "").split(",") if k.strip()]
        keys = [self.gemini_api_key] if self.gemini_api_key else []
        keys.extend(extra)
        return list(dict.fromkeys(keys))

    @model_validator(mode="after")
    def _require_api_key_in_production(self) -> Settings:
        # Auth is opt-in (unset API_KEY disables it) everywhere except production,
        # where shipping with auth silently disabled is a safety hazard, not a
        # convenience — fail fast at startup instead of serving unauthenticated.
        if self.environment == Environment.PRODUCTION and not self.api_key:
            raise ValueError(
                "API_KEY must be set when ENVIRONMENT=production "
                "(unauthenticated production deployment is not allowed)."
            )
        return self


def get_settings() -> Settings:
    """Construct Settings from the current environment. Not cached — see module docstring."""
    return Settings()
