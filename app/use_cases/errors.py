"""
Domain-level errors raised by application use cases.

Routes in `app.main` catch these and translate them to a stable HTTPException
— use cases never construct or raise `HTTPException` themselves, keeping
transport concerns (status codes, response shape) out of application logic.
"""

from __future__ import annotations


class UseCaseError(Exception):
    """Base class for errors raised by an application use case."""


class InvalidInputError(UseCaseError):
    """The request is well-formed per its schema but violates a business rule
    (e.g. empty classifier input, a non-user final chat message). Maps to 400."""


class InferenceUnavailableError(UseCaseError):
    """The classifier raised an unexpected error. Maps to 500; the original
    exception is not exposed to the caller."""
