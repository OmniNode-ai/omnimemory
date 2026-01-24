"""
OnexError compatibility stub.

This is a local implementation of OnexError until
omnibase_core.core.errors.core_errors is available.

NOTE ON Any TYPES:
This module intentionally uses 'Any' types for:
- correlation_id: Can be str, UUID, int, or any hashable identifier
- context/details: Error context can contain arbitrary serializable data
- **kwargs: Required for extensibility in compatibility stubs

These are documented exceptions to the zero-Any policy for compat modules.
"""

from __future__ import annotations

from typing import Any


class OnexError(Exception):
    """
    Base exception for all ONEX errors.

    Provides structured error context and chaining support.
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
        error_code: str | None = None,
        correlation_id: Any | None = None,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize OnexError.

        Args:
            message: Human-readable error message
            code: Error code identifier (legacy parameter)
            context: Additional context data
            cause: Original exception that caused this error
            error_code: Error code identifier (preferred over code)
            correlation_id: Request correlation ID for tracing
            details: Additional details (merged into context)
            **kwargs: Additional keyword arguments for extensibility
        """
        super().__init__(message)
        self.message = message
        # Support both 'code' and 'error_code' parameters
        self.code = error_code or code or "ONEX_ERROR"
        self.error_code = self.code  # Alias for compatibility
        # Merge context and details (copy to prevent caller mutation)
        self.context = dict(context) if context else {}
        if details:
            self.context.update(details)
        self.cause = cause
        self.correlation_id = correlation_id
        # Set __cause__ for proper exception chaining
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        """Return string representation."""
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message": self.message,
            "code": self.code,
            "error_code": self.error_code,
            "context": self.context,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "cause": str(self.cause) if self.cause else None,
        }


class BaseOnexError(OnexError):
    """Alias for OnexError used by legacy code."""
