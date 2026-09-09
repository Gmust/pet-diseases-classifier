"""
Stable entry point: `uvicorn app.main:app`.

The application itself is assembled in `app.api`; this module only re-exports
it. The import path is referenced by the Lambda handler, the SAM template, the
README, and the test suite, so it stays put even as the internals move.
"""

from app.api import app, ensure_services
from app.bootstrap import build_services

__all__ = ["app", "build_services", "ensure_services"]
