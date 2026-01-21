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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class NodeResult(Generic[T]):
    """
    Monadic result pattern for ONEX error handling.

    Provides Railway-oriented programming patterns for
    clean error handling in async operations.

    None Semantics:
        - Success results CAN have None as a valid value when T is Optional[X]
        - The `is_success` flag determines success/failure, NOT the value
        - Use `is_success` to check result state, not `value is not None`
        - Example: NodeResult[Optional[str]].success(None) is valid

    Type Safety:
        - Methods like map/flat_map/unwrap work with any T, including Optional types
        - The value field is Optional[T] to handle both failure (None) and
          success-with-None-value cases
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
            # Safe to cast: is_success=True implies value is not None
            new_value = func(cast(T, self.value))
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
            # Safe to cast: is_success=True implies value is not None
            inner_result = func(cast(T, self.value))
            # Combine provenance chains (outer first, then inner)
            combined_provenance = self.provenance + inner_result.provenance
            # Multiply trust scores for cumulative trust degradation
            combined_trust = self.trust_score * inner_result.trust_score
            # Merge metadata (inner takes precedence)
            combined_metadata = {**self.metadata, **inner_result.metadata}

            if inner_result.is_success:
                # Safe to cast: is_success=True implies value is not None
                return NodeResult[U].success(
                    cast(U, inner_result.value),
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

        Note: If T is Optional[X] and success(None) was called, this returns None.
        The return type is T, which may itself be Optional.

        Returns:
            The success value (may be None if T is Optional)

        Raises:
            Exception: The stored error if this is a failure
        """
        if not self.is_success:
            if self.error:
                raise self.error
            raise ValueError(self.error_message or "Unknown error")
        # Safe to cast: is_success=True implies value is not None
        return cast(T, self.value)

    def unwrap_or(self, default: T) -> T:
        """
        Get the success value or return a default.

        Note: The default is ONLY used when is_success=False (failure case).
        If this is a success result with value=None (when T is Optional[X]),
        None is returned, NOT the default.

        Args:
            default: Value to return if this is a failure

        Returns:
            The success value (may be None if T is Optional) or the default on failure
        """
        if not self.is_success:
            return default
        # Safe to cast: is_success=True implies value is not None
        return cast(T, self.value)

    def __bool__(self) -> bool:
        """Return True if this is a success result."""
        return self.is_success
