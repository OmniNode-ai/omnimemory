"""
Node Result type for ONEX protocol compliance.

This module provides a simple NodeResult implementation to replace
the missing omnibase_core.core.monadic.model_node_result import.
"""

from typing import Generic, Optional, TypeVar

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


class NodeResult(Generic[T]):
    """
    Simple Result container for ONEX node operations.

    Provides a success/error result pattern similar to Rust's Result<T, E>
    or other monadic error handling patterns.
    """

    def __init__(self, value: Optional[T] = None, error: Optional[Exception] = None):
        self._value = value
        self._error = error
        self._is_success = error is None

    @classmethod
    def success(cls, value: T) -> "NodeResult[T]":
        """Create a successful result."""
        return cls(value=value)

    @classmethod
    def failure(cls, error: Exception) -> "NodeResult[T]":
        """Create a failed result."""
        return cls(error=error)

    @property
    def is_success(self) -> bool:
        """Check if the operation was successful."""
        return self._is_success

    @property
    def is_failure(self) -> bool:
        """Check if the operation failed."""
        return not self._is_success

    @property
    def value(self) -> T:
        """Get the success value (raises if failure)."""
        if self._is_success:
            return self._value
        raise ValueError("Cannot get value from failed result")

    @property
    def error(self) -> Optional[Exception]:
        """Get the error if operation failed."""
        return self._error

    def unwrap(self) -> T:
        """Unwrap the value or raise the error."""
        if self._is_success:
            return self._value
        raise self._error or RuntimeError("Operation failed without specific error")

    def unwrap_or(self, default: T) -> T:
        """Unwrap the value or return default."""
        return self._value if self._is_success else default
