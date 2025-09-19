"""
Error Models and Exception Classes for OmniMemory ONEX Architecture

This module defines comprehensive error handling following ONEX standards,
including structured error codes, exception chaining, and monadic error patterns
that integrate with NodeResult for consistent error handling across the system.
"""

from enum import Enum
from typing import Optional
from uuid import UUID


# Use local base exception instead of omnibase_core
class BaseOnexError(Exception):
    """Base exception for ONEX errors."""

    pass


from pydantic import BaseModel, Field

from ..models.foundation import ModelMetadata

# === ERROR CODES ===


class OmniMemoryErrorCode(str, Enum):
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


class ErrorCategoryInfo(BaseModel):
    """Information about an error category."""

    prefix: str = Field(description="Error code prefix")
    description: str = Field(description="Category description")
    recoverable: bool = Field(description="Whether errors are generally recoverable")
    default_retry_count: int = Field(3, description="Default retry count")
    default_backoff_factor: float = Field(2.0, description="Default backoff multiplier")


ERROR_CATEGORIES: Dict[str, ErrorCategoryInfo] = {
    "VALIDATION": ErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_VAL",
        description="Input validation errors",
        recoverable=False,
        default_retry_count=0,
        default_backoff_factor=1.0,
    ),
    "STORAGE": ErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_STO",
        description="Storage system errors",
        recoverable=True,
        default_retry_count=3,
        default_backoff_factor=2.0,
    ),
    "RETRIEVAL": ErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_RET",
        description="Memory retrieval errors",
        recoverable=True,
        default_retry_count=2,
        default_backoff_factor=1.5,
    ),
    "PROCESSING": ErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_PRO",
        description="Intelligence processing errors",
        recoverable=True,
        default_retry_count=2,
        default_backoff_factor=2.0,
    ),
    "COORDINATION": ErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_COR",
        description="Coordination and workflow errors",
        recoverable=True,
        default_retry_count=3,
        default_backoff_factor=1.5,
    ),
    "SYSTEM": ErrorCategoryInfo(
        prefix="ONEX_OMNIMEMORY_SYS",
        description="System-level errors",
        recoverable=False,
        default_retry_count=1,
        default_backoff_factor=1.0,
    ),
}


def get_error_category(error_code: OmniMemoryErrorCode) -> Optional[ErrorCategoryInfo]:
    """Get error category information for an error code."""
    for category_name, category_info in ERROR_CATEGORIES.items():
        if error_code.value.startswith(category_info.prefix):
            return category_info
    return None


# === BASE EXCEPTION CLASSES ===


class OmniMemoryError(BaseOnexError):
    """
    Base exception class for all OmniMemory errors.

    Extends ONEX BaseOnexError with OmniMemory-specific functionality
    including error categorization, recovery hints, and monadic integration.
    """

    def __init__(
        self,
        error_code: OmniMemoryErrorCode,
        message: str,
        context: Optional[ModelMetadata] = None,
        correlation_id: Optional[UUID] = None,
        cause: Optional[Exception] = None,
        recovery_hint: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize OmniMemory error.

        Args:
            error_code: Specific error code from OmniMemoryErrorCode
            message: Human-readable error message
            context: Additional error context information
            correlation_id: Request correlation ID for tracing
            cause: Underlying exception that caused this error
            recovery_hint: Suggestion for error recovery
            retry_after: Suggested retry delay in seconds
            **kwargs: Additional keyword arguments passed to BaseOnexError
        """
        # Get error category information
        category_info = get_error_category(error_code)

        # Enhance context with category information
        enhanced_context = context or {}
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

        # Initialize base OnexError
        super().__init__(
            error_code=error_code.value,
            message=message,
            context=enhanced_context,
            correlation_id=correlation_id,
            **kwargs,
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

    def to_dict(self) -> dict[str, str]:
        """Convert error to dictionary for serialization."""
        base_dict = {
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


class ValidationError(OmniMemoryError):
    """Exception for input validation errors."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        validation_rule: Optional[str] = None,
        **kwargs,
    ):
        # Determine specific validation error code
        error_code = OmniMemoryErrorCode.INVALID_INPUT
        if "schema" in message.lower():
            error_code = OmniMemoryErrorCode.SCHEMA_VIOLATION
        elif "constraint" in message.lower():
            error_code = OmniMemoryErrorCode.CONSTRAINT_VIOLATION
        elif "required" in message.lower():
            error_code = OmniMemoryErrorCode.MISSING_REQUIRED_FIELD
        elif "format" in message.lower():
            error_code = OmniMemoryErrorCode.INVALID_FORMAT
        elif "range" in message.lower():
            error_code = OmniMemoryErrorCode.VALUE_OUT_OF_RANGE

        # Build context with validation details
        context = kwargs.get("context", {})
        if field_name:
            context["field_name"] = field_name
        if field_value is not None:
            context["field_value"] = str(field_value)
        if validation_rule:
            context["validation_rule"] = validation_rule

        kwargs["context"] = context
        kwargs[
            "recovery_hint"
        ] = "Review and correct the input data according to the schema requirements"

        super().__init__(error_code=error_code, message=message, **kwargs)


class StorageError(OmniMemoryError):
    """Exception for storage system errors."""

    def __init__(
        self,
        message: str,
        storage_system: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        # Determine specific storage error code
        error_code = OmniMemoryErrorCode.STORAGE_UNAVAILABLE
        if "quota" in message.lower() or "full" in message.lower():
            error_code = OmniMemoryErrorCode.QUOTA_EXCEEDED
        elif "corrupt" in message.lower():
            error_code = OmniMemoryErrorCode.CORRUPTION_DETECTED
        elif "persist" in message.lower():
            error_code = OmniMemoryErrorCode.PERSISTENCE_FAILED
        elif "backup" in message.lower():
            error_code = OmniMemoryErrorCode.BACKUP_FAILED
        elif "restore" in message.lower():
            error_code = OmniMemoryErrorCode.RESTORE_FAILED
        elif "timeout" in message.lower():
            error_code = OmniMemoryErrorCode.STORAGE_TIMEOUT
        elif "connection" in message.lower():
            error_code = OmniMemoryErrorCode.CONNECTION_FAILED
        elif "transaction" in message.lower():
            error_code = OmniMemoryErrorCode.TRANSACTION_FAILED

        # Build context with storage details
        context = kwargs.get("context", {})
        if storage_system:
            context["storage_system"] = storage_system
        if operation:
            context["operation"] = operation

        kwargs["context"] = context
        kwargs[
            "recovery_hint"
        ] = "Check storage system health and retry with exponential backoff"
        kwargs["retry_after"] = 5  # Suggest 5 second retry delay

        super().__init__(error_code=error_code, message=message, **kwargs)


class RetrievalError(OmniMemoryError):
    """Exception for memory retrieval errors."""

    def __init__(
        self,
        message: str,
        memory_id: Optional[UUID] = None,
        query: Optional[str] = None,
        **kwargs,
    ):
        # Determine specific retrieval error code
        error_code = OmniMemoryErrorCode.SEARCH_FAILED
        if "not found" in message.lower():
            error_code = OmniMemoryErrorCode.MEMORY_NOT_FOUND
        elif "index" in message.lower() and "unavailable" in message.lower():
            error_code = OmniMemoryErrorCode.INDEX_UNAVAILABLE
        elif "access denied" in message.lower():
            error_code = OmniMemoryErrorCode.ACCESS_DENIED
        elif "invalid" in message.lower() and "query" in message.lower():
            error_code = OmniMemoryErrorCode.QUERY_INVALID
        elif "timeout" in message.lower():
            error_code = OmniMemoryErrorCode.SEARCH_TIMEOUT
        elif "corrupt" in message.lower():
            error_code = OmniMemoryErrorCode.INDEX_CORRUPTION
        elif "embedding" in message.lower():
            error_code = OmniMemoryErrorCode.EMBEDDING_UNAVAILABLE
        elif "similarity" in message.lower():
            error_code = OmniMemoryErrorCode.SIMILARITY_COMPUTATION_FAILED
        elif "filter" in message.lower():
            error_code = OmniMemoryErrorCode.FILTER_INVALID

        # Build context with retrieval details
        context = kwargs.get("context", {})
        if memory_id:
            context["memory_id"] = str(memory_id)
        if query:
            context["query"] = query

        kwargs["context"] = context
        kwargs["recovery_hint"] = "Verify search parameters and check index health"

        super().__init__(error_code=error_code, message=message, **kwargs)


class ProcessingError(OmniMemoryError):
    """Exception for intelligence processing errors."""

    def __init__(
        self,
        message: str,
        processing_stage: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        # Determine specific processing error code
        error_code = OmniMemoryErrorCode.PROCESSING_FAILED
        if "model unavailable" in message.lower():
            error_code = OmniMemoryErrorCode.MODEL_UNAVAILABLE
        elif "resource" in message.lower() and "exhaust" in message.lower():
            error_code = OmniMemoryErrorCode.RESOURCE_EXHAUSTED
        elif "analysis failed" in message.lower():
            error_code = OmniMemoryErrorCode.ANALYSIS_FAILED
        elif "embedding" in message.lower():
            error_code = OmniMemoryErrorCode.EMBEDDING_GENERATION_FAILED
        elif "pattern" in message.lower():
            error_code = OmniMemoryErrorCode.PATTERN_RECOGNITION_FAILED
        elif "semantic" in message.lower():
            error_code = OmniMemoryErrorCode.SEMANTIC_ANALYSIS_FAILED
        elif "insight" in message.lower():
            error_code = OmniMemoryErrorCode.INSIGHT_EXTRACTION_FAILED
        elif "model load" in message.lower():
            error_code = OmniMemoryErrorCode.MODEL_LOAD_FAILED
        elif "timeout" in message.lower():
            error_code = OmniMemoryErrorCode.COMPUTATION_TIMEOUT

        # Build context with processing details
        context = kwargs.get("context", {})
        if processing_stage:
            context["processing_stage"] = processing_stage
        if model_name:
            context["model_name"] = model_name

        kwargs["context"] = context
        kwargs["recovery_hint"] = "Check model availability and processing resources"

        super().__init__(error_code=error_code, message=message, **kwargs)


class CoordinationError(OmniMemoryError):
    """Exception for coordination and workflow errors."""

    def __init__(
        self,
        message: str,
        workflow_id: Optional[UUID] = None,
        agent_ids: Optional[List[UUID]] = None,
        **kwargs,
    ):
        # Determine specific coordination error code
        error_code = OmniMemoryErrorCode.WORKFLOW_FAILED
        if "deadlock" in message.lower():
            error_code = OmniMemoryErrorCode.DEADLOCK_DETECTED
        elif "sync" in message.lower() and "failed" in message.lower():
            error_code = OmniMemoryErrorCode.SYNC_FAILED
        elif "agent unavailable" in message.lower():
            error_code = OmniMemoryErrorCode.AGENT_UNAVAILABLE
        elif "timeout" in message.lower():
            error_code = OmniMemoryErrorCode.COORDINATION_TIMEOUT
        elif "parallel" in message.lower():
            error_code = OmniMemoryErrorCode.PARALLEL_EXECUTION_FAILED
        elif "state" in message.lower():
            error_code = OmniMemoryErrorCode.STATE_MANAGEMENT_FAILED
        elif "broadcast" in message.lower():
            error_code = OmniMemoryErrorCode.BROADCAST_FAILED
        elif "migration" in message.lower():
            error_code = OmniMemoryErrorCode.MIGRATION_FAILED
        elif "orchestration" in message.lower():
            error_code = OmniMemoryErrorCode.ORCHESTRATION_FAILED

        # Build context with coordination details
        context = kwargs.get("context", {})
        if workflow_id:
            context["workflow_id"] = str(workflow_id)
        if agent_ids:
            context["agent_ids"] = [str(aid) for aid in agent_ids]

        kwargs["context"] = context
        kwargs["recovery_hint"] = "Check agent availability and retry coordination"

        super().__init__(error_code=error_code, message=message, **kwargs)


class SystemError(OmniMemoryError):
    """Exception for system-level errors."""

    def __init__(
        self,
        message: str,
        system_component: Optional[str] = None,
        **kwargs,
    ):
        # Determine specific system error code
        error_code = OmniMemoryErrorCode.INTERNAL_ERROR
        if "config" in message.lower():
            error_code = OmniMemoryErrorCode.CONFIG_ERROR
        elif "dependency" in message.lower():
            error_code = OmniMemoryErrorCode.DEPENDENCY_FAILED
        elif "service unavailable" in message.lower():
            error_code = OmniMemoryErrorCode.SERVICE_UNAVAILABLE
        elif "initialization" in message.lower():
            error_code = OmniMemoryErrorCode.INITIALIZATION_FAILED
        elif "shutdown" in message.lower():
            error_code = OmniMemoryErrorCode.SHUTDOWN_FAILED
        elif "health check" in message.lower():
            error_code = OmniMemoryErrorCode.HEALTH_CHECK_FAILED
        elif "metrics" in message.lower():
            error_code = OmniMemoryErrorCode.METRICS_COLLECTION_FAILED
        elif "security" in message.lower():
            error_code = OmniMemoryErrorCode.SECURITY_VIOLATION
        elif "rate limit" in message.lower():
            error_code = OmniMemoryErrorCode.RATE_LIMIT_EXCEEDED

        # Build context with system details
        context = kwargs.get("context", {})
        if system_component:
            context["system_component"] = system_component

        kwargs["context"] = context
        kwargs["recovery_hint"] = "Contact system administrator for system-level issues"

        super().__init__(error_code=error_code, message=message, **kwargs)


# === ERROR UTILITIES ===


def wrap_exception(
    exception: Exception,
    error_code: OmniMemoryErrorCode,
    message: Optional[str] = None,
    **kwargs,
) -> OmniMemoryError:
    """
    Wrap a generic exception in an OmniMemoryError.

    Args:
        exception: The original exception to wrap
        error_code: The OmniMemory error code to use
        message: Optional custom message (uses exception message if not provided)
        **kwargs: Additional arguments for OmniMemoryError constructor

    Returns:
        OmniMemoryError wrapping the original exception
    """
    error_message = message or str(exception)
    return OmniMemoryError(
        error_code=error_code,
        message=error_message,
        cause=exception,
        **kwargs,
    )


def chain_errors(
    primary_error: OmniMemoryError,
    secondary_error: Exception,
) -> OmniMemoryError:
    """
    Chain a secondary error to a primary OmniMemoryError.

    Args:
        primary_error: The primary OmniMemoryError
        secondary_error: The secondary exception to chain

    Returns:
        Updated primary error with chained secondary error
    """
    if primary_error.cause is None:
        primary_error.cause = secondary_error
        primary_error.__cause__ = secondary_error
    else:
        # If there's already a cause, chain it
        current = primary_error.cause
        while hasattr(current, "__cause__") and current.__cause__ is not None:
            current = current.__cause__
        current.__cause__ = secondary_error

    return primary_error


def create_error_summary(errors: list[OmniMemoryError]) -> dict[str, str]:
    """
    Create a summary of multiple errors for reporting.

    Args:
        errors: List of OmniMemoryError instances

    Returns:
        Dictionary containing error summary statistics
    """
    if not errors:
        return {"total_errors": 0}

    error_counts = {}
    category_counts = {}
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
