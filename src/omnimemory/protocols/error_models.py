# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Error Models and Exception Classes for OmniMemory ONEX Architecture

This module defines comprehensive error handling following ONEX standards,
including structured error codes, exception chaining, and monadic error patterns
that integrate with ModelBaseResult for consistent error handling across the system.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

# Type alias for field values in validation errors
# Supports common field types that can fail validation
FieldValueType = (
    str | int | float | bool | bytes | list[object] | dict[str, object] | None
)

# Import ModelOnexError from omnibase_core
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from pydantic import BaseModel, ConfigDict, Field

from ..models.foundation import ModelMetadata


def _normalize_context_to_dict(
    context: ModelMetadata | dict[str, object] | None,
) -> dict[str, object]:
    """
    Normalize context parameter to a mutable dict for error handling.

    Handles both ModelMetadata (uses to_dict()) and plain dicts.
    Returns an empty dict if context is None.
    """
    if context is None:
        return {}
    if isinstance(context, ModelMetadata):
        return dict(
            context.to_dict()
        )  # to_dict() returns dict[str, str], copy for safety
    # context is dict[str, object] at this point
    return dict(context)  # Shallow copy for mutability


# === ERROR CODES ===


class EnumOmniMemoryErrorCode(str, Enum):
    """Comprehensive error codes for OmniMemory operations."""

    # Validation Errors (ONEX_OMNIMEMORY_VAL_XXX)
    INVALID_INPUT = "ONEX_OMNIMEMORY_VAL_001_INVALID_INPUT"
    SCHEMA_VIOLATION = "ONEX_OMNIMEMORY_VAL_002_SCHEMA_VIOLATION"
    CONSTRAINT_VIOLATION = "ONEX_OMNIMEMORY_VAL_003_CONSTRAINT_VIOLATION"
    MISSING_REQUIRED_FIELD = "ONEX_OMNIMEMORY_VAL_004_MISSING_REQUIRED_FIELD"
    INVALID_FORMAT = "ONEX_OMNIMEMORY_VAL_005_INVALID_FORMAT"
    VALUE_OUT_OF_RANGE = "ONEX_OMNIMEMORY_VAL_006_VALUE_OUT_OF_RANGE"

    # Storage Errors (ONEX_OMNIMEMORY_STO_XXX)
    STORAGE_UNAVAILABLE = "ONEX_OMNIMEMORY_STO_001_STORAGE_UNAVAILABLE"
    QUOTA_EXCEEDED = "ONEX_OMNIMEMORY_STO_002_QUOTA_EXCEEDED"
    CORRUPTION_DETECTED = "ONEX_OMNIMEMORY_STO_003_CORRUPTION_DETECTED"
    STORAGE_FULL = "ONEX_OMNIMEMORY_STO_004_STORAGE_FULL"
    PERSISTENCE_FAILED = "ONEX_OMNIMEMORY_STO_005_PERSISTENCE_FAILED"
    BACKUP_FAILED = "ONEX_OMNIMEMORY_STO_006_BACKUP_FAILED"
    RESTORE_FAILED = "ONEX_OMNIMEMORY_STO_007_RESTORE_FAILED"
    STORAGE_TIMEOUT = "ONEX_OMNIMEMORY_STO_008_STORAGE_TIMEOUT"
    CONNECTION_FAILED = "ONEX_OMNIMEMORY_STO_009_CONNECTION_FAILED"
    TRANSACTION_FAILED = "ONEX_OMNIMEMORY_STO_010_TRANSACTION_FAILED"

    # Retrieval Errors (ONEX_OMNIMEMORY_RET_XXX)
    MEMORY_NOT_FOUND = "ONEX_OMNIMEMORY_RET_001_MEMORY_NOT_FOUND"
    INDEX_UNAVAILABLE = "ONEX_OMNIMEMORY_RET_002_INDEX_UNAVAILABLE"
    ACCESS_DENIED = "ONEX_OMNIMEMORY_RET_003_ACCESS_DENIED"
    SEARCH_FAILED = "ONEX_OMNIMEMORY_RET_004_SEARCH_FAILED"
    QUERY_INVALID = "ONEX_OMNIMEMORY_RET_005_QUERY_INVALID"
    SEARCH_TIMEOUT = "ONEX_OMNIMEMORY_RET_006_SEARCH_TIMEOUT"
    INDEX_CORRUPTION = "ONEX_OMNIMEMORY_RET_007_INDEX_CORRUPTION"
    EMBEDDING_UNAVAILABLE = "ONEX_OMNIMEMORY_RET_008_EMBEDDING_UNAVAILABLE"
    SIMILARITY_COMPUTATION_FAILED = (
        "ONEX_OMNIMEMORY_RET_009_SIMILARITY_COMPUTATION_FAILED"
    )
    FILTER_INVALID = "ONEX_OMNIMEMORY_RET_010_FILTER_INVALID"

    # Processing Errors (ONEX_OMNIMEMORY_PRO_XXX)
    PROCESSING_FAILED = "ONEX_OMNIMEMORY_PRO_001_PROCESSING_FAILED"
    MODEL_UNAVAILABLE = "ONEX_OMNIMEMORY_PRO_002_MODEL_UNAVAILABLE"
    RESOURCE_EXHAUSTED = "ONEX_OMNIMEMORY_PRO_003_RESOURCE_EXHAUSTED"
    ANALYSIS_FAILED = "ONEX_OMNIMEMORY_PRO_004_ANALYSIS_FAILED"
    EMBEDDING_GENERATION_FAILED = "ONEX_OMNIMEMORY_PRO_005_EMBEDDING_GENERATION_FAILED"
    PATTERN_RECOGNITION_FAILED = "ONEX_OMNIMEMORY_PRO_006_PATTERN_RECOGNITION_FAILED"
    SEMANTIC_ANALYSIS_FAILED = "ONEX_OMNIMEMORY_PRO_007_SEMANTIC_ANALYSIS_FAILED"
    INSIGHT_EXTRACTION_FAILED = "ONEX_OMNIMEMORY_PRO_008_INSIGHT_EXTRACTION_FAILED"
    MODEL_LOAD_FAILED = "ONEX_OMNIMEMORY_PRO_009_MODEL_LOAD_FAILED"
    COMPUTATION_TIMEOUT = "ONEX_OMNIMEMORY_PRO_010_COMPUTATION_TIMEOUT"

    # Coordination Errors (ONEX_OMNIMEMORY_COR_XXX)
    WORKFLOW_FAILED = "ONEX_OMNIMEMORY_COR_001_WORKFLOW_FAILED"
    DEADLOCK_DETECTED = "ONEX_OMNIMEMORY_COR_002_DEADLOCK_DETECTED"
    SYNC_FAILED = "ONEX_OMNIMEMORY_COR_003_SYNC_FAILED"
    AGENT_UNAVAILABLE = "ONEX_OMNIMEMORY_COR_004_AGENT_UNAVAILABLE"
    COORDINATION_TIMEOUT = "ONEX_OMNIMEMORY_COR_005_COORDINATION_TIMEOUT"
    PARALLEL_EXECUTION_FAILED = "ONEX_OMNIMEMORY_COR_006_PARALLEL_EXECUTION_FAILED"
    STATE_MANAGEMENT_FAILED = "ONEX_OMNIMEMORY_COR_007_STATE_MANAGEMENT_FAILED"
    BROADCAST_FAILED = "ONEX_OMNIMEMORY_COR_008_BROADCAST_FAILED"
    MIGRATION_FAILED = "ONEX_OMNIMEMORY_COR_009_MIGRATION_FAILED"
    ORCHESTRATION_FAILED = "ONEX_OMNIMEMORY_COR_010_ORCHESTRATION_FAILED"

    # System Errors (ONEX_OMNIMEMORY_SYS_XXX)
    INTERNAL_ERROR = "ONEX_OMNIMEMORY_SYS_001_INTERNAL_ERROR"
    CONFIG_ERROR = "ONEX_OMNIMEMORY_SYS_002_CONFIG_ERROR"
    DEPENDENCY_FAILED = "ONEX_OMNIMEMORY_SYS_003_DEPENDENCY_FAILED"
    SERVICE_UNAVAILABLE = "ONEX_OMNIMEMORY_SYS_004_SERVICE_UNAVAILABLE"
    INITIALIZATION_FAILED = "ONEX_OMNIMEMORY_SYS_005_INITIALIZATION_FAILED"
    SHUTDOWN_FAILED = "ONEX_OMNIMEMORY_SYS_006_SHUTDOWN_FAILED"
    HEALTH_CHECK_FAILED = "ONEX_OMNIMEMORY_SYS_007_HEALTH_CHECK_FAILED"
    METRICS_COLLECTION_FAILED = "ONEX_OMNIMEMORY_SYS_008_METRICS_COLLECTION_FAILED"
    SECURITY_VIOLATION = "ONEX_OMNIMEMORY_SYS_009_SECURITY_VIOLATION"
    RATE_LIMIT_EXCEEDED = "ONEX_OMNIMEMORY_SYS_010_RATE_LIMIT_EXCEEDED"


# === ERROR CATEGORY METADATA ===


class ModelErrorCategoryInfo(BaseModel):
    """Information about an error category."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
        frozen=True,  # Category info is immutable
        from_attributes=True,  # Enable ORM-style attribute access
    )

    prefix: str = Field(description="Error code prefix")
    description: str = Field(description="Category description")
    recoverable: bool = Field(description="Whether errors are generally recoverable")
    default_retry_count: int = Field(3, description="Default retry count")
    default_backoff_factor: float = Field(2.0, description="Default backoff multiplier")


ERROR_CATEGORIES: dict[str, ModelErrorCategoryInfo] = {
    "VALIDATION": ModelErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_VAL",
        description="Input validation errors",
        recoverable=False,
        default_retry_count=0,
        default_backoff_factor=1.0,
    ),
    "STORAGE": ModelErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_STO",
        description="Storage system errors",
        recoverable=True,
        default_retry_count=3,
        default_backoff_factor=2.0,
    ),
    "RETRIEVAL": ModelErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_RET",
        description="Memory retrieval errors",
        recoverable=True,
        default_retry_count=2,
        default_backoff_factor=1.5,
    ),
    "PROCESSING": ModelErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_PRO",
        description="Intelligence processing errors",
        recoverable=True,
        default_retry_count=2,
        default_backoff_factor=2.0,
    ),
    "COORDINATION": ModelErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_COR",
        description="Coordination and workflow errors",
        recoverable=True,
        default_retry_count=3,
        default_backoff_factor=1.5,
    ),
    "SYSTEM": ModelErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_SYS",
        description="System-level errors",
        recoverable=False,
        default_retry_count=1,
        default_backoff_factor=1.0,
    ),
}


def get_error_category(
    error_code: EnumOmniMemoryErrorCode,
) -> ModelErrorCategoryInfo | None:
    """Get error category information for an error code."""
    for _category_name, category_info in ERROR_CATEGORIES.items():
        if error_code.value.startswith(category_info.prefix):
            return category_info
    return None


# === BASE EXCEPTION CLASSES ===


class ProtocolOmniMemoryError(ModelOnexError):
    """
    Base exception class for all OmniMemory errors.

    Extends ONEX ModelOnexError with OmniMemory-specific functionality
    including error categorization, recovery hints, and monadic integration.
    """

    def __init__(
        self,
        error_code: EnumOmniMemoryErrorCode,
        message: str,
        context: ModelMetadata | dict[str, object] | None = None,
        correlation_id: UUID | None = None,
        cause: Exception | None = None,
        recovery_hint: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        """
        Initialize OmniMemory error.

        Args:
            error_code: Specific error code from EnumOmniMemoryErrorCode
            message: Human-readable error message
            context: Additional error context information (ModelMetadata or dict)
            correlation_id: Request correlation ID for tracing
            cause: Underlying exception that caused this error
            recovery_hint: Suggestion for error recovery
            retry_after: Suggested retry delay in seconds
        """
        # Get error category information
        category_info = get_error_category(error_code)

        # Enhance context with category information
        # Normalize ModelMetadata to dict using to_dict() for proper key-value extraction
        enhanced_context: dict[str, object] = _normalize_context_to_dict(context)
        if category_info:
            enhanced_context.update(
                {
                    "error_category": category_info.prefix.split("_")[-1].lower(),
                    "recoverable": category_info.recoverable,
                    "default_retry_count": category_info.default_retry_count,
                    "default_backoff_factor": category_info.default_backoff_factor,
                }
            )

        # Add recovery information
        if recovery_hint:
            enhanced_context["recovery_hint"] = recovery_hint
        if retry_after:
            enhanced_context["retry_after_seconds"] = retry_after

        # Initialize base ModelOnexError
        super().__init__(
            message=message,
            error_code=error_code.value,
            correlation_id=correlation_id,
            context=enhanced_context,
        )

        # Store additional OmniMemory-specific information
        self.omnimemory_error_code = error_code
        self.category_info = category_info
        self.recovery_hint = recovery_hint
        self.retry_after = retry_after
        self.cause = cause

        # Chain the underlying cause if provided
        if cause:
            self.__cause__ = cause

    def is_recoverable(self) -> bool:
        """Check if this error is generally recoverable."""
        return self.category_info.recoverable if self.category_info else False

    def get_retry_count(self) -> int:
        """Get suggested retry count for this error."""
        return self.category_info.default_retry_count if self.category_info else 0

    def get_backoff_factor(self) -> float:
        """Get suggested backoff factor for retries."""
        return self.category_info.default_backoff_factor if self.category_info else 1.0

    def to_dict(self) -> dict[str, object]:
        """Convert error to dictionary for serialization."""
        base_dict: dict[str, object] = {
            "error_code": self.omnimemory_error_code.value,
            "message": self.message,
            "context": self.context,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "recoverable": self.is_recoverable(),
            "retry_count": self.get_retry_count(),
            "backoff_factor": self.get_backoff_factor(),
        }

        if self.recovery_hint:
            base_dict["recovery_hint"] = self.recovery_hint
        if self.retry_after:
            base_dict["retry_after_seconds"] = self.retry_after
        if self.cause:
            base_dict["cause"] = str(self.cause)

        return base_dict


# === CATEGORY-SPECIFIC EXCEPTION CLASSES ===


class ProtocolValidationError(ProtocolOmniMemoryError):
    """Exception for input validation errors."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        field_value: FieldValueType | None = None,
        validation_rule: str | None = None,
        context: ModelMetadata | dict[str, object] | None = None,
        correlation_id: UUID | None = None,
        cause: Exception | None = None,
    ) -> None:
        # Determine specific validation error code
        error_code = EnumOmniMemoryErrorCode.INVALID_INPUT
        if "schema" in message.lower():
            error_code = EnumOmniMemoryErrorCode.SCHEMA_VIOLATION
        elif "constraint" in message.lower():
            error_code = EnumOmniMemoryErrorCode.CONSTRAINT_VIOLATION
        elif "required" in message.lower():
            error_code = EnumOmniMemoryErrorCode.MISSING_REQUIRED_FIELD
        elif "format" in message.lower():
            error_code = EnumOmniMemoryErrorCode.INVALID_FORMAT
        elif "range" in message.lower():
            error_code = EnumOmniMemoryErrorCode.VALUE_OUT_OF_RANGE

        # Build context with validation details - normalize to mutable dict
        context_dict: dict[str, object] = _normalize_context_to_dict(context)
        if field_name:
            context_dict["field_name"] = field_name
        if field_value is not None:
            context_dict["field_value"] = str(field_value)
        if validation_rule:
            context_dict["validation_rule"] = validation_rule

        super().__init__(
            error_code=error_code,
            message=message,
            context=context_dict,
            correlation_id=correlation_id,
            cause=cause,
            recovery_hint="Review and correct input data per schema requirements",
        )


class ProtocolStorageError(ProtocolOmniMemoryError):
    """Exception for storage system errors."""

    def __init__(
        self,
        message: str,
        storage_system: str | None = None,
        operation: str | None = None,
        context: ModelMetadata | dict[str, object] | None = None,
        correlation_id: UUID | None = None,
        cause: Exception | None = None,
    ) -> None:
        # Determine specific storage error code
        error_code = EnumOmniMemoryErrorCode.STORAGE_UNAVAILABLE
        if "quota" in message.lower() or "full" in message.lower():
            error_code = EnumOmniMemoryErrorCode.QUOTA_EXCEEDED
        elif "corrupt" in message.lower():
            error_code = EnumOmniMemoryErrorCode.CORRUPTION_DETECTED
        elif "persist" in message.lower():
            error_code = EnumOmniMemoryErrorCode.PERSISTENCE_FAILED
        elif "backup" in message.lower():
            error_code = EnumOmniMemoryErrorCode.BACKUP_FAILED
        elif "restore" in message.lower():
            error_code = EnumOmniMemoryErrorCode.RESTORE_FAILED
        elif "timeout" in message.lower():
            error_code = EnumOmniMemoryErrorCode.STORAGE_TIMEOUT
        elif "connection" in message.lower():
            error_code = EnumOmniMemoryErrorCode.CONNECTION_FAILED
        elif "transaction" in message.lower():
            error_code = EnumOmniMemoryErrorCode.TRANSACTION_FAILED

        # Build context with storage details - normalize to mutable dict
        context_dict: dict[str, object] = _normalize_context_to_dict(context)
        if storage_system:
            context_dict["storage_system"] = storage_system
        if operation:
            context_dict["operation"] = operation

        super().__init__(
            error_code=error_code,
            message=message,
            context=context_dict,
            correlation_id=correlation_id,
            cause=cause,
            recovery_hint="Check storage health and retry with backoff",
            retry_after=5,  # Suggest 5 second retry delay
        )


class ProtocolRetrievalError(ProtocolOmniMemoryError):
    """Exception for memory retrieval errors."""

    def __init__(
        self,
        message: str,
        memory_id: UUID | None = None,
        query: str | None = None,
        context: ModelMetadata | dict[str, object] | None = None,
        correlation_id: UUID | None = None,
        cause: Exception | None = None,
    ) -> None:
        # Determine specific retrieval error code
        error_code = EnumOmniMemoryErrorCode.SEARCH_FAILED
        if "not found" in message.lower():
            error_code = EnumOmniMemoryErrorCode.MEMORY_NOT_FOUND
        elif "index" in message.lower() and "unavailable" in message.lower():
            error_code = EnumOmniMemoryErrorCode.INDEX_UNAVAILABLE
        elif "access denied" in message.lower():
            error_code = EnumOmniMemoryErrorCode.ACCESS_DENIED
        elif "invalid" in message.lower() and "query" in message.lower():
            error_code = EnumOmniMemoryErrorCode.QUERY_INVALID
        elif "timeout" in message.lower():
            error_code = EnumOmniMemoryErrorCode.SEARCH_TIMEOUT
        elif "corrupt" in message.lower():
            error_code = EnumOmniMemoryErrorCode.INDEX_CORRUPTION
        elif "embedding" in message.lower():
            error_code = EnumOmniMemoryErrorCode.EMBEDDING_UNAVAILABLE
        elif "similarity" in message.lower():
            error_code = EnumOmniMemoryErrorCode.SIMILARITY_COMPUTATION_FAILED
        elif "filter" in message.lower():
            error_code = EnumOmniMemoryErrorCode.FILTER_INVALID

        # Build context with retrieval details - normalize to mutable dict
        context_dict: dict[str, object] = _normalize_context_to_dict(context)
        if memory_id:
            context_dict["memory_id"] = str(memory_id)
        if query:
            context_dict["query"] = query

        super().__init__(
            error_code=error_code,
            message=message,
            context=context_dict,
            correlation_id=correlation_id,
            cause=cause,
            recovery_hint="Verify search parameters and check index health",
        )


class ProtocolProcessingError(ProtocolOmniMemoryError):
    """Exception for intelligence processing errors."""

    def __init__(
        self,
        message: str,
        processing_stage: str | None = None,
        model_name: str | None = None,
        context: ModelMetadata | dict[str, object] | None = None,
        correlation_id: UUID | None = None,
        cause: Exception | None = None,
    ) -> None:
        # Determine specific processing error code
        error_code = EnumOmniMemoryErrorCode.PROCESSING_FAILED
        if "model unavailable" in message.lower():
            error_code = EnumOmniMemoryErrorCode.MODEL_UNAVAILABLE
        elif "resource" in message.lower() and "exhaust" in message.lower():
            error_code = EnumOmniMemoryErrorCode.RESOURCE_EXHAUSTED
        elif "analysis failed" in message.lower():
            error_code = EnumOmniMemoryErrorCode.ANALYSIS_FAILED
        elif "embedding" in message.lower():
            error_code = EnumOmniMemoryErrorCode.EMBEDDING_GENERATION_FAILED
        elif "pattern" in message.lower():
            error_code = EnumOmniMemoryErrorCode.PATTERN_RECOGNITION_FAILED
        elif "semantic" in message.lower():
            error_code = EnumOmniMemoryErrorCode.SEMANTIC_ANALYSIS_FAILED
        elif "insight" in message.lower():
            error_code = EnumOmniMemoryErrorCode.INSIGHT_EXTRACTION_FAILED
        elif "model load" in message.lower():
            error_code = EnumOmniMemoryErrorCode.MODEL_LOAD_FAILED
        elif "timeout" in message.lower():
            error_code = EnumOmniMemoryErrorCode.COMPUTATION_TIMEOUT

        # Build context with processing details - normalize to mutable dict
        context_dict: dict[str, object] = _normalize_context_to_dict(context)
        if processing_stage:
            context_dict["processing_stage"] = processing_stage
        if model_name:
            context_dict["model_name"] = model_name

        super().__init__(
            error_code=error_code,
            message=message,
            context=context_dict,
            correlation_id=correlation_id,
            cause=cause,
            recovery_hint="Check model availability and processing resources",
        )


class ProtocolCoordinationError(ProtocolOmniMemoryError):
    """Exception for coordination and workflow errors."""

    def __init__(
        self,
        message: str,
        workflow_id: UUID | None = None,
        agent_ids: list[UUID] | None = None,
        context: ModelMetadata | dict[str, object] | None = None,
        correlation_id: UUID | None = None,
        cause: Exception | None = None,
    ) -> None:
        # Determine specific coordination error code
        error_code = EnumOmniMemoryErrorCode.WORKFLOW_FAILED
        if "deadlock" in message.lower():
            error_code = EnumOmniMemoryErrorCode.DEADLOCK_DETECTED
        elif "sync" in message.lower() and "failed" in message.lower():
            error_code = EnumOmniMemoryErrorCode.SYNC_FAILED
        elif "agent unavailable" in message.lower():
            error_code = EnumOmniMemoryErrorCode.AGENT_UNAVAILABLE
        elif "timeout" in message.lower():
            error_code = EnumOmniMemoryErrorCode.COORDINATION_TIMEOUT
        elif "parallel" in message.lower():
            error_code = EnumOmniMemoryErrorCode.PARALLEL_EXECUTION_FAILED
        elif "state" in message.lower():
            error_code = EnumOmniMemoryErrorCode.STATE_MANAGEMENT_FAILED
        elif "broadcast" in message.lower():
            error_code = EnumOmniMemoryErrorCode.BROADCAST_FAILED
        elif "migration" in message.lower():
            error_code = EnumOmniMemoryErrorCode.MIGRATION_FAILED
        elif "orchestration" in message.lower():
            error_code = EnumOmniMemoryErrorCode.ORCHESTRATION_FAILED

        # Build context with coordination details - normalize to mutable dict
        context_dict: dict[str, object] = _normalize_context_to_dict(context)
        if workflow_id:
            context_dict["workflow_id"] = str(workflow_id)
        if agent_ids:
            context_dict["agent_ids"] = [str(aid) for aid in agent_ids]

        super().__init__(
            error_code=error_code,
            message=message,
            context=context_dict,
            correlation_id=correlation_id,
            cause=cause,
            recovery_hint="Check agent availability and retry coordination",
        )


class ProtocolSystemError(ProtocolOmniMemoryError):
    """Exception for system-level errors."""

    def __init__(
        self,
        message: str,
        system_component: str | None = None,
        context: ModelMetadata | dict[str, object] | None = None,
        correlation_id: UUID | None = None,
        cause: Exception | None = None,
    ) -> None:
        # Determine specific system error code
        error_code = EnumOmniMemoryErrorCode.INTERNAL_ERROR
        if "config" in message.lower():
            error_code = EnumOmniMemoryErrorCode.CONFIG_ERROR
        elif "dependency" in message.lower():
            error_code = EnumOmniMemoryErrorCode.DEPENDENCY_FAILED
        elif "service unavailable" in message.lower():
            error_code = EnumOmniMemoryErrorCode.SERVICE_UNAVAILABLE
        elif "initialization" in message.lower():
            error_code = EnumOmniMemoryErrorCode.INITIALIZATION_FAILED
        elif "shutdown" in message.lower():
            error_code = EnumOmniMemoryErrorCode.SHUTDOWN_FAILED
        elif "health check" in message.lower():
            error_code = EnumOmniMemoryErrorCode.HEALTH_CHECK_FAILED
        elif "metrics" in message.lower():
            error_code = EnumOmniMemoryErrorCode.METRICS_COLLECTION_FAILED
        elif "security" in message.lower():
            error_code = EnumOmniMemoryErrorCode.SECURITY_VIOLATION
        elif "rate limit" in message.lower():
            error_code = EnumOmniMemoryErrorCode.RATE_LIMIT_EXCEEDED

        # Build context with system details - normalize to mutable dict
        context_dict: dict[str, object] = _normalize_context_to_dict(context)
        if system_component:
            context_dict["system_component"] = system_component

        super().__init__(
            error_code=error_code,
            message=message,
            context=context_dict,
            correlation_id=correlation_id,
            cause=cause,
            recovery_hint="Contact system administrator for system-level issues",
        )


# === ERROR UTILITIES ===


def wrap_exception(
    exception: Exception,
    error_code: EnumOmniMemoryErrorCode,
    message: str | None = None,
    context: ModelMetadata | dict[str, object] | None = None,
    correlation_id: UUID | None = None,
    recovery_hint: str | None = None,
    retry_after: int | None = None,
) -> ProtocolOmniMemoryError:
    """
    Wrap a generic exception in an ProtocolOmniMemoryError.

    Args:
        exception: The original exception to wrap
        error_code: The OmniMemory error code to use
        message: Optional custom message (uses exception message if not provided)
        context: Optional error context information
        correlation_id: Optional request correlation ID
        recovery_hint: Optional suggestion for error recovery
        retry_after: Optional suggested retry delay in seconds

    Returns:
        ProtocolOmniMemoryError wrapping the original exception
    """
    error_message = message or str(exception)
    return ProtocolOmniMemoryError(
        error_code=error_code,
        message=error_message,
        context=context,
        correlation_id=correlation_id,
        cause=exception,
        recovery_hint=recovery_hint,
        retry_after=retry_after,
    )


def chain_errors(
    primary_error: ProtocolOmniMemoryError,
    secondary_error: Exception,
) -> ProtocolOmniMemoryError:
    """
    Chain a secondary error to a primary ProtocolOmniMemoryError.

    Args:
        primary_error: The primary ProtocolOmniMemoryError
        secondary_error: The secondary exception to chain

    Returns:
        Updated primary error with chained secondary error
    """
    if primary_error.cause is None:
        primary_error.cause = secondary_error
        primary_error.__cause__ = secondary_error
    else:
        # If there's already a cause, chain it
        current: Exception = primary_error.cause
        while hasattr(current, "__cause__") and current.__cause__ is not None:
            current = current.__cause__  # type: ignore[assignment]
        current.__cause__ = secondary_error

    return primary_error


def create_error_summary(errors: list[ProtocolOmniMemoryError]) -> dict[str, object]:
    """
    Create a summary of multiple errors for reporting.

    Args:
        errors: List of ProtocolOmniMemoryError instances

    Returns:
        ErrorSummary containing error summary statistics with all fields populated
    """
    if not errors:
        return {
            "total_errors": 0,
            "recoverable_errors": 0,
            "non_recoverable_errors": 0,
            "error_counts_by_code": {},
            "error_counts_by_category": {},
            "recovery_rate": 0.0,
        }

    error_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    recoverable_count = 0

    for error in errors:
        # Count by error code
        error_code = error.omnimemory_error_code.value
        error_counts[error_code] = error_counts.get(error_code, 0) + 1

        # Count by category
        if error.category_info:
            category = error.category_info.prefix.split("_")[-1].lower()
            category_counts[category] = category_counts.get(category, 0) + 1

        # Count recoverable errors
        if error.is_recoverable():
            recoverable_count += 1

    return {
        "total_errors": len(errors),
        "recoverable_errors": recoverable_count,
        "non_recoverable_errors": len(errors) - recoverable_count,
        "error_counts_by_code": error_counts,
        "error_counts_by_category": category_counts,
        "recovery_rate": recoverable_count / len(errors) if errors else 0.0,
    }
