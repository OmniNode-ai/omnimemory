"""
NodeResult monadic pattern - compatibility stub.

This is a local implementation of NodeResult until
omnibase_core.core.monadic.model_node_result is available.

NOTE ON Any TYPES:
This module intentionally uses 'Any' types for:
- metadata: dict[str, Any] - Result metadata can contain arbitrary serializable data
- **extra_metadata: Any - Allows flexible metadata addition from kwargs

These are documented exceptions to the zero-Any policy for compat modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class NodeResult(Generic[T]):
    """
    Monadic result pattern for ONEX error handling.

    Provides Railway-oriented programming patterns for
    clean error handling in async operations.
    """

    value: T | None = None
    error: Exception | None = None
    error_message: str | None = None
    is_success: bool = True
    provenance: list[str] = field(default_factory=list)
    trust_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        """Return True if this is a failure result."""
        return not self.is_success

    @classmethod
    def success(
        cls,
        value: T,
        provenance: list[str] | None = None,
        trust_score: float = 1.0,
        metadata: dict[str, Any] | None = None,
        **extra_metadata: Any,
    ) -> NodeResult[T]:
        """
        Create a successful result.

        Args:
            value: The success value
            provenance: List of provenance strings tracking operation path
            trust_score: Trust score for the result (0.0 to 1.0)
            metadata: Additional metadata dict
            **extra_metadata: Additional metadata as kwargs

        Returns:
            NodeResult with success value
        """
        combined_metadata = metadata or {}
        combined_metadata.update(extra_metadata)
        return cls(
            value=value,
            error=None,
            error_message=None,
            is_success=True,
            provenance=provenance or [],
            trust_score=trust_score,
            metadata=combined_metadata,
        )

    @classmethod
    def failure(
        cls,
        error: Exception | None = None,
        error_message: str | None = None,
        provenance: list[str] | None = None,
        trust_score: float = 0.0,
        metadata: dict[str, Any] | None = None,
        **extra_metadata: Any,
    ) -> NodeResult[T]:
        """
        Create a failure result.

        Args:
            error: The exception that caused the failure
            error_message: Human-readable error message
            provenance: List of provenance strings tracking operation path
            trust_score: Trust score for the result (0.0 to 1.0)
            metadata: Additional metadata dict
            **extra_metadata: Additional metadata as kwargs

        Returns:
            NodeResult with failure information
        """
        if error_message is None and error is not None:
            error_message = str(error)

        combined_metadata = metadata or {}
        combined_metadata.update(extra_metadata)
        return cls(
            value=None,
            error=error,
            error_message=error_message,
            is_success=False,
            provenance=provenance or [],
            trust_score=trust_score,
            metadata=combined_metadata,
        )

    def map(self, func: Callable[[T], U]) -> NodeResult[U]:
        """
        Transform the success value.

        Preserves provenance, trust_score, and metadata through the transformation.

        Args:
            func: Function to apply to success value

        Returns:
            New NodeResult with transformed value or original error
        """
        if not self.is_success:
            return NodeResult[U].failure(
                error=self.error,
                error_message=self.error_message,
                provenance=self.provenance,
                trust_score=self.trust_score,
                metadata=self.metadata,
            )

        try:
            new_value = func(self.value)
            return NodeResult[U].success(
                new_value,
                provenance=self.provenance,
                trust_score=self.trust_score,
                metadata=self.metadata,
            )
        except Exception as e:
            return NodeResult[U].failure(
                error=e,
                provenance=self.provenance,
                trust_score=self.trust_score,
                metadata=self.metadata,
            )

    def flat_map(self, func: Callable[[T], NodeResult[U]]) -> NodeResult[U]:
        """
        Chain operations that return NodeResult.

        Combines provenance chains and propagates trust scores through the chain.
        The inner result's trust_score is multiplied with the outer's to reflect
        cumulative trust degradation through the operation chain.

        Args:
            func: Function that returns a NodeResult

        Returns:
            The NodeResult from the chained operation with combined metadata
        """
        if not self.is_success:
            return NodeResult[U].failure(
                error=self.error,
                error_message=self.error_message,
                provenance=self.provenance,
                trust_score=self.trust_score,
                metadata=self.metadata,
            )

        try:
            inner_result = func(self.value)
            # Combine provenance chains (outer first, then inner)
            combined_provenance = self.provenance + inner_result.provenance
            # Multiply trust scores for cumulative trust degradation
            combined_trust = self.trust_score * inner_result.trust_score
            # Merge metadata (inner takes precedence)
            combined_metadata = {**self.metadata, **inner_result.metadata}

            if inner_result.is_success:
                return NodeResult[U].success(
                    inner_result.value,
                    provenance=combined_provenance,
                    trust_score=combined_trust,
                    metadata=combined_metadata,
                )
            else:
                return NodeResult[U].failure(
                    error=inner_result.error,
                    error_message=inner_result.error_message,
                    provenance=combined_provenance,
                    trust_score=combined_trust,
                    metadata=combined_metadata,
                )
        except Exception as e:
            return NodeResult[U].failure(
                error=e,
                provenance=self.provenance,
                trust_score=self.trust_score,
                metadata=self.metadata,
            )

    def unwrap(self) -> T:
        """
        Get the success value or raise the error.

        Returns:
            The success value

        Raises:
            Exception: The stored error if this is a failure
        """
        if not self.is_success:
            if self.error:
                raise self.error
            raise ValueError(self.error_message or "Unknown error")
        return self.value

    def unwrap_or(self, default: T) -> T:
        """
        Get the success value or return a default.

        Args:
            default: Value to return if this is a failure

        Returns:
            The success value or the default
        """
        if not self.is_success:
            return default
        return self.value

    def __bool__(self) -> bool:
        """Return True if this is a success result."""
        return self.is_success
