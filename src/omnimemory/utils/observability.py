# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Observability utilities for OmniMemory ONEX architecture.

- ContextVar integration for correlation ID tracking
- Distributed tracing support
- Enhanced logging with correlation context
- Performance monitoring and metrics collection
- In-process metrics: Counter, Histogram, Gauge
- Handler observability wrapper for "one wrapper, one log line, no payload" pattern

Structured Log Schema (for downstream ingestion - ELK, Datadog, etc.):
------------------------------------------------------------------------
All handler operation log events follow this consistent schema:

Required fields:
    - correlation_id (str): Unique request correlation identifier
    - operation (str): Operation name (e.g., "store", "retrieve", "delete")
    - handler (str): Handler name (e.g., "filesystem", "postgresql")
    - status (str): Operation status, one of "success" or "failure"
    - latency_ms (float): Operation latency in milliseconds (2 decimal places)
    - timestamp (str): ISO8601 timestamp in UTC (format: YYYY-MM-DDTHH:MM:SS.sssZ)

Optional fields (only present on failure):
    - error_type (str): Exception class name (e.g., "ValueError", "IOError")
    - error_message (str): Sanitized error message (PII-safe)

Example log event (success):
    {
        "correlation_id": "abc123-def456",
        "operation": "store",
        "handler": "filesystem",
        "status": "success",
        "latency_ms": 45.23,
        "timestamp": "2025-01-19T12:34:56.789Z"
    }

Example log event (failure):
    {
        "correlation_id": "abc123-def456",
        "operation": "store",
        "handler": "filesystem",
        "status": "failure",
        "latency_ms": 102.5,
        "timestamp": "2025-01-19T12:34:56.789Z",
        "error_type": "IOError",
        "error_message": "Permission denied"
    }

Metric Label Validation (strict_labels option):
-----------------------------------------------
All metric classes (Counter, Histogram, Gauge) support a `strict_labels` parameter:

    - strict_labels=False (default): Labels are matched by name, missing labels
      default to empty string, extra labels are ignored. This mode exists for
      backwards compatibility only.

    - strict_labels=True (RECOMMENDED FOR PRODUCTION): Validates that EXACTLY the
      expected labels are provided. Raises ValueError if labels are missing or
      extra labels are passed. This catches bugs early and prevents data quality issues.

**WARNING: Silent Mis-Tagging Without strict_labels**

When strict_labels=False (the default), missing labels silently default to empty
string (""). This can cause serious issues:

    1. **Metric Pollution**: Operations get tagged with "" instead of the correct
       value, making metrics unreliable for alerting and dashboards.

    2. **Hidden Bugs**: Typos in label names (e.g., "opration" vs "operation")
       silently create new label dimensions with empty required labels.

    3. **Cardinality Issues**: Empty string labels aggregate unrelated operations,
       skewing percentiles and averages.

    4. **Debugging Nightmares**: When alerts fire, you can't trace back to the
       specific operation because labels are missing or incorrect.

**Best Practices**:
    - ALWAYS use strict_labels=True in production code
    - Use strict_labels=False only for temporary migrations or testing
    - Consider enabling strict_labels globally via a configuration flag

Example - Lenient mode (DEFAULT - NOT RECOMMENDED):
    counter = Counter("ops", ["operation", "status"])
    counter.inc(operation="store")  # "status" silently defaults to ""
    # Result: metric recorded as ("store", "") - SILENT MIS-TAG!
    # This won't raise an error but WILL corrupt your metrics!

Example - Strict mode (RECOMMENDED FOR PRODUCTION):
    counter = Counter("ops", ["operation", "status"], strict_labels=True)
    counter.inc(operation="store")
    # Raises: ValueError: Label validation failed for metric 'ops':
    #         missing labels: ['status'].
    #         Expected: ['operation', 'status'], got: ['operation']

Example - Extra labels also caught in strict mode:
    counter = Counter("ops", ["operation", "status"], strict_labels=True)
    counter.inc(operation="store", status="ok", region="us-east")
    # Raises: ValueError: Label validation failed for metric 'ops':
    #         extra labels: ['region'].
    #         Expected: ['operation', 'status'],
    #         got: ['operation', 'region', 'status']
"""

from __future__ import annotations

import functools
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Literal,
    TypeVar,
    cast,
)

import structlog

from ..models.foundation.model_typed_collections import (
    ModelKeyValuePair,
    ModelMetadata,
)
from ..models.utils.model_correlation_context import ModelCorrelationContext
from ..models.utils.model_structured_log_entry import (
    ModelStructuredLogEntry,
    TraceLevel,
)

# Re-export TraceLevel for public API
__all__ = ["TraceLevel"]

# Aliases for internal use
CorrelationContext = ModelCorrelationContext
StructuredLogEntry = ModelStructuredLogEntry

# Type variable for generic function types
F = TypeVar("F", bound=Callable[..., object])

# Type alias for metadata values - supports common serializable types
# This replaces Any with explicit types for type safety
MetadataValue = str | int | float | bool | None

# === LABEL VALIDATION UTILITIES ===


class LabelValidationError(Exception):
    """Exception raised when label validation fails.

    Attributes:
        metric_name: Name of the metric where validation failed
        missing_labels: Set of required labels that were not provided
        extra_labels: Set of unexpected labels that were provided
        expected_labels: Set of labels that were expected
        provided_labels: Set of labels that were actually provided
    """

    def __init__(
        self,
        metric_name: str,
        missing_labels: set[str],
        extra_labels: set[str],
        expected_labels: set[str],
        provided_labels: set[str],
    ) -> None:
        self.metric_name = metric_name
        self.missing_labels = missing_labels
        self.extra_labels = extra_labels
        self.expected_labels = expected_labels
        self.provided_labels = provided_labels

        # Build error message
        errors = []
        if missing_labels:
            errors.append(f"missing required labels: {sorted(missing_labels)}")
        if extra_labels:
            errors.append(f"unexpected extra labels: {sorted(extra_labels)}")

        message = (
            f"Label validation failed for metric '{metric_name}': "
            f"{'; '.join(errors)}. "
            f"Expected labels: {sorted(expected_labels)}, "
            f"got: {sorted(provided_labels)}"
        )
        super().__init__(message)


def validate_metric_labels(
    labels: dict[str, str],
    required_labels: set[str],
    allowed_labels: set[str] | None = None,
    metric_name: str = "unknown",
    strict: bool = True,
) -> None:
    """Validate labels against required and allowed sets.

    This is a standalone utility function for label validation that can be used
    outside of metric operations. It enforces that:
    1. All required labels are present
    2. No unexpected labels are provided (if strict=True)

    Args:
        labels: Dictionary of label key-value pairs to validate
        required_labels: Set of label names that MUST be present
        allowed_labels: Set of all allowed label names. If None, defaults to
                       required_labels (no extra labels allowed). Pass a superset
                       of required_labels to allow optional labels.
        metric_name: Name of metric for error messages (default: "unknown")
        strict: If True, raise exception on validation failure.
               If False, only log warnings for extra labels. (default: True)

    Raises:
        LabelValidationError: If strict=True and validation fails
        ValueError: If required_labels is empty

    Example:
        # Strict validation - must have exactly these labels
        validate_metric_labels(
            labels={"operation": "store", "status": "success"},
            required_labels={"operation", "status"},
            metric_name="my_counter"
        )

        # Allow optional labels
        validate_metric_labels(
            labels={"operation": "store", "region": "us-east"},
            required_labels={"operation"},
            allowed_labels={"operation", "region", "zone"},
            metric_name="my_counter"
        )

        # Warn-only mode (log warnings but don't raise)
        validate_metric_labels(
            labels={"operation": "store", "unknown": "value"},
            required_labels={"operation"},
            metric_name="my_counter",
            strict=False  # Will log warning for "unknown" but not raise
        )
    """
    if not required_labels:
        raise ValueError("required_labels must not be empty")

    # Default allowed_labels to required_labels if not specified
    if allowed_labels is None:
        allowed_labels = required_labels

    provided = set(labels.keys())
    missing = required_labels - provided
    extra = provided - allowed_labels

    if missing or extra:
        if strict:
            raise LabelValidationError(
                metric_name=metric_name,
                missing_labels=missing,
                extra_labels=extra,
                expected_labels=allowed_labels,
                provided_labels=provided,
            )
        # Non-strict mode: log warnings but don't raise
        _label_logger = structlog.get_logger("omnimemory.label_validation")
        if missing:
            _label_logger.error(
                "label_validation_missing_required",
                metric_name=metric_name,
                missing_labels=sorted(missing),
                provided_labels=sorted(provided),
                required_labels=sorted(required_labels),
            )
        if extra:
            _label_logger.warning(
                "label_validation_unexpected_extra",
                metric_name=metric_name,
                extra_labels=sorted(extra),
                provided_labels=sorted(provided),
                allowed_labels=sorted(allowed_labels),
            )


# === STRUCTURED LOG SCHEMA VALIDATION ===
# StructuredLogEntry model is imported from ..models.utils.model_observability


def validate_log_entry(
    log_data: dict[str, object],
    raise_on_error: bool = True,
) -> StructuredLogEntry | None:
    """Validate a log entry against the structured log schema.

    This function validates that a log entry dictionary conforms to the
    expected schema for OmniMemory handler operations. It can be used to:
    1. Validate log entries before emission
    2. Validate log entries during testing
    3. Validate log entries during log aggregation/parsing

    Args:
        log_data: Dictionary containing log entry fields
        raise_on_error: If True, raise ValidationError on schema violation.
                       If False, return None on validation failure. (default: True)

    Returns:
        StructuredLogEntry if validation succeeds, None if raise_on_error=False
        and validation fails

    Raises:
        pydantic.ValidationError: If raise_on_error=True and validation fails

    Example:
        # Validate and get typed object
        entry = validate_log_entry({
            "correlation_id": "abc123",
            "operation": "store",
            "handler": "filesystem",
            "status": "success",
            "latency_ms": 45.23,
            "timestamp": "2025-01-19T12:34:56.789Z"
        })

        # Check without raising
        entry = validate_log_entry(log_data, raise_on_error=False)
        if entry is None:
            print("Log entry is invalid")
    """
    from pydantic import ValidationError

    try:
        return StructuredLogEntry.model_validate(log_data)
    except ValidationError:
        if raise_on_error:
            raise
        return None


def create_validated_log_entry(
    correlation_id: str,
    operation: str,
    handler: str,
    status: Literal["success", "failure"],
    latency_ms: float,
    error_type: str | None = None,
    error_message: str | None = None,
) -> StructuredLogEntry:
    """Create a validated log entry with automatic timestamp generation.

    This is a convenience function for creating log entries that:
    1. Automatically generates an ISO8601 UTC timestamp
    2. Validates all fields against the schema
    3. Returns a typed StructuredLogEntry object

    Args:
        correlation_id: Unique request correlation identifier
        operation: Operation name (e.g., "store", "retrieve", "delete")
        handler: Handler name (e.g., "filesystem", "postgresql")
        status: Operation status ("success" or "failure")
        latency_ms: Operation latency in milliseconds
        error_type: Exception class name (only for failures)
        error_message: Sanitized error message (only for failures)

    Returns:
        Validated StructuredLogEntry object

    Raises:
        pydantic.ValidationError: If any field fails validation

    Example:
        entry = create_validated_log_entry(
            correlation_id="abc123",
            operation="store",
            handler="filesystem",
            status="success",
            latency_ms=45.23
        )
    """
    # Generate ISO8601 timestamp in UTC with millisecond precision
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    return StructuredLogEntry(
        correlation_id=correlation_id,
        operation=operation,
        handler=handler,
        status=status,
        latency_ms=round(latency_ms, 2),
        timestamp=timestamp,
        error_type=error_type,
        error_message=error_message,
    )


# Optional psutil import for memory tracking - gracefully degrade if unavailable
_psutil_available = False
try:
    import psutil  # type: ignore[import-untyped]

    _psutil_available = True
except ImportError:
    psutil = None
from .error_sanitizer import SanitizationLevel
from .error_sanitizer import sanitize_error as _base_sanitize_error

# === SECURITY VALIDATION FUNCTIONS ===


def validate_correlation_id(correlation_id: str) -> bool:
    """
    Validate correlation ID format to prevent injection attacks.

    Args:
        correlation_id: Correlation ID to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not correlation_id:
        return False

    # Allow UUIDs (with or without hyphens) and alphanumeric strings up to 64 chars
    # This prevents injection while allowing reasonable correlation ID formats
    pattern = r"^[a-zA-Z0-9\-_]{1,64}$"
    return re.match(pattern, correlation_id) is not None


def sanitize_metadata_value(value: object) -> MetadataValue:
    """
    Sanitize metadata values to prevent injection attacks.

    Converts arbitrary objects to safe serializable types (str, int, float, bool, None).

    Args:
        value: Value to sanitize (accepts any object for flexibility)

    Returns:
        Sanitized value as one of the safe MetadataValue types
    """
    if isinstance(value, str):
        # Remove potential injection patterns and limit length
        sanitized = re.sub(r'[<>"\'\\\n\r\t]', "", value)
        return sanitized[:1000]  # Limit string length
    elif isinstance(value, bool):
        # Check bool before int since bool is a subclass of int
        return value
    elif isinstance(value, int | float):
        return value
    elif value is None:
        return None
    else:
        # Convert to string and sanitize
        return sanitize_metadata_value(str(value))


def _sanitize_error(error: Exception) -> str:
    """
    Sanitize error messages to prevent information disclosure in logs.

    Uses the centralized error sanitizer for consistent security handling.

    Args:
        error: Exception to sanitize

    Returns:
        Safe error message without sensitive information
    """
    return _base_sanitize_error(
        error, context="observability", level=SanitizationLevel.STANDARD
    )


# Context variables for correlation tracking
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
operation_var: ContextVar[str | None] = ContextVar("operation", default=None)

logger = structlog.get_logger(__name__)


# === IN-PROCESS METRICS ===
# Minimal implementation for P1C observability - no external dependencies


# Default histogram buckets for latency measurements (in milliseconds)
DEFAULT_LATENCY_BUCKETS: tuple[float, ...] = (
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
)

# Default maximum number of unique label combinations per metric
# This prevents unbounded memory growth from high-cardinality labels
DEFAULT_MAX_METRIC_ENTRIES: int = 10000

# Default maximum number of active traces in ObservabilityManager
# This prevents unbounded memory growth from long-running or abandoned traces
DEFAULT_MAX_ACTIVE_TRACES: int = 1000


@dataclass
class CounterValue:
    """Thread-safe counter value with labels."""

    value: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, amount: int = 1) -> None:
        """Increment the counter."""
        with self._lock:
            self.value += amount

    def get(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self.value


class Counter:
    """In-process counter metric for tracking totals.

    Thread-safe counter that tracks total operations by labels.
    Uses bounded storage with LRU eviction to prevent unbounded memory growth.

    Label Validation:
        By default (strict_labels=False), missing labels silently default to empty
        string (""), which can cause metric pollution and debugging issues. For
        production use, ALWAYS set strict_labels=True to catch label mismatches
        immediately.

    Example - Basic usage (lenient mode, NOT RECOMMENDED for production):
        counter = Counter("memory_operation_total", ["operation", "status", "handler"])
        counter.inc(operation="store", status="success", handler="filesystem")
        counter.inc(operation="store")  # DANGER: "status" and "handler" become ""

    Example - Production usage (strict mode, RECOMMENDED):
        counter = Counter("ops", ["operation", "status"], strict_labels=True)
        counter.inc(operation="store", status="ok")  # OK - all labels provided
        counter.inc(operation="store")  # Raises ValueError: missing labels: ['status']
        counter.inc(operation="store", status="ok", extra="bad")
        # Raises ValueError: extra labels: ['extra']

    Warning:
        Without strict_labels=True, typos in label names will NOT raise errors.
        Instead, the typo creates a new label with empty string for required labels:

        counter = Counter("ops", ["operation", "status"])
        counter.inc(opration="store", status="ok")  # Note typo: "opration"
        # Result: labels are ("", "ok") - "opration" is ignored, "operation" is ""!
    """

    def __init__(
        self,
        name: str,
        label_names: list[str],
        max_entries: int = DEFAULT_MAX_METRIC_ENTRIES,
        strict_labels: bool = False,
    ) -> None:
        """Initialize counter with name and label names.

        Args:
            name: Metric name (e.g., "memory_operation_total")
            label_names: List of label names (e.g., ["operation", "status", "handler"])
            max_entries: Maximum number of unique label combinations (default 10000).
                         Oldest entries are evicted when limit is exceeded.
            strict_labels: If True, validate that all required labels are provided
                          and no extra labels are passed. Raises ValueError on mismatch.
                          Default is False for backwards compatibility.
        """
        self.name = name
        self.label_names = label_names
        self.max_entries = max_entries
        self.strict_labels = strict_labels
        self._label_names_set = frozenset(label_names)
        # Use OrderedDict for LRU eviction (oldest entries evicted first)
        self._values: OrderedDict[tuple[str, ...], CounterValue] = OrderedDict()
        self._lock = threading.Lock()

    def _validate_labels(self, labels: dict[str, str]) -> None:
        """Validate labels against expected label names.

        Args:
            labels: Label key-value pairs to validate

        Raises:
            ValueError: If strict_labels is True and labels don't match expected names
        """
        if not self.strict_labels:
            return

        provided = set(labels.keys())
        expected = self._label_names_set

        missing = expected - provided
        extra = provided - expected

        errors = []
        if missing:
            errors.append(f"missing labels: {sorted(missing)}")
        if extra:
            errors.append(f"extra labels: {sorted(extra)}")

        if errors:
            raise ValueError(
                f"Label validation failed for metric '{self.name}': "
                f"{'; '.join(errors)}. "
                f"Expected: {sorted(expected)}, got: {sorted(provided)}"
            )

    def inc(self, amount: int = 1, **labels: str) -> None:
        """Increment the counter with given labels.

        Args:
            amount: Amount to increment by (default 1)
            **labels: Label key-value pairs

        Raises:
            ValueError: If strict_labels is True and labels don't match expected names

        Thread-safety: The value holder reference is captured under the lock
        to prevent race conditions with reset() or eviction.
        """
        self._validate_labels(labels)
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                # Evict oldest entries if at capacity
                while len(self._values) >= self.max_entries:
                    self._values.popitem(last=False)  # Remove oldest (FIFO)
                self._values[key] = CounterValue()
            else:
                # Move to end to mark as recently used
                self._values.move_to_end(key)
            # Capture reference under lock to prevent TOCTOU race with reset/eviction
            value_holder = self._values[key]
        # Safe to call outside lock - value_holder is our reference
        value_holder.inc(amount)

    def get(self, **labels: str) -> int:
        """Get counter value for given labels."""
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                return 0
            # Move to end to mark as recently used
            self._values.move_to_end(key)
            return self._values[key].get()

    def get_all(self) -> dict[tuple[str, ...], int]:
        """Get all counter values with their labels."""
        with self._lock:
            return {k: v.get() for k, v in self._values.items()}

    def _labels_to_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        """Convert labels dict to hashable key tuple."""
        return tuple(labels.get(name, "") for name in self.label_names)

    def labels_from_key(self, key: tuple[str, ...]) -> dict[str, str]:
        """Convert key tuple back to labels dict."""
        return dict(zip(self.label_names, key, strict=True))


@dataclass
class HistogramValue:
    """Thread-safe histogram value with buckets."""

    buckets: tuple[float, ...]
    bucket_counts: list[int] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        """Initialize bucket counts."""
        if not self.bucket_counts:
            self.bucket_counts = [0] * (len(self.buckets) + 1)  # +1 for +Inf

    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            self.sum_value += value
            self.count += 1
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    self.bucket_counts[i] += 1
            # Always increment +Inf bucket
            self.bucket_counts[-1] += 1

    def get_snapshot(self) -> dict[str, float | int | list[int] | list[float]]:
        """Get a snapshot of histogram values."""
        with self._lock:
            return {
                "sum": self.sum_value,
                "count": self.count,
                "buckets": list(self.bucket_counts),
                "bucket_bounds": list(self.buckets) + [float("inf")],
            }


class Histogram:
    """In-process histogram metric for tracking distributions.

    Thread-safe histogram that tracks value distributions with configurable buckets.
    Uses bounded storage with LRU eviction to prevent unbounded memory growth.

    Label Validation:
        By default (strict_labels=False), missing labels silently default to empty
        string (""), which corrupts histogram percentiles by mixing unrelated data.
        For production use, ALWAYS set strict_labels=True.

    Example - Basic usage (lenient mode, NOT RECOMMENDED for production):
        hist = Histogram("memory_storage_latency_ms", ["operation", "handler"])
        hist.observe(45.2, operation="store", handler="filesystem")
        hist.observe(100.0, operation="store")  # DANGER: "handler" becomes ""
        # Now p99 latency for ("store", "") mixes data that should be separate!

    Example - Production usage (strict mode, RECOMMENDED):
        hist = Histogram("latency", ["operation", "handler"], strict_labels=True)
        hist.observe(45.2, operation="store", handler="fs")  # OK - all labels
        hist.observe(45.2, operation="store")
        # Raises ValueError: missing labels: ['handler']
        hist.observe(45.2, operation="store", handler="fs", region="us")
        # Raises ValueError: extra labels: ['region']

    Warning:
        Missing labels in histograms are especially dangerous because they corrupt
        percentile calculations. If some observations go to ("store", "fs") and
        others to ("store", ""), your p50/p99 metrics become meaningless.
    """

    def __init__(
        self,
        name: str,
        label_names: list[str],
        buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS,
        max_entries: int = DEFAULT_MAX_METRIC_ENTRIES,
        strict_labels: bool = False,
    ) -> None:
        """Initialize histogram with name, labels, and buckets.

        Args:
            name: Metric name (e.g., "memory_storage_latency_ms")
            label_names: List of label names
            buckets: Tuple of bucket boundaries (default: latency buckets)
            max_entries: Maximum number of unique label combinations (default 10000).
                         Oldest entries are evicted when limit is exceeded.
            strict_labels: If True, validate that all required labels are provided
                          and no extra labels are passed. Raises ValueError on mismatch.
                          Default is False for backwards compatibility.

        Raises:
            ValueError: If buckets tuple is empty, contains non-positive values,
                       or is not in strictly ascending order.
        """
        # Validate buckets
        if not buckets:
            raise ValueError("Histogram buckets tuple must not be empty")

        for i, bucket in enumerate(buckets):
            if bucket <= 0:
                raise ValueError(
                    f"Histogram bucket values must be positive (> 0), "
                    f"got {bucket} at index {i}"
                )

        for i in range(1, len(buckets)):
            if buckets[i] <= buckets[i - 1]:
                raise ValueError(
                    f"Histogram buckets must be in strictly ascending order, "
                    f"but bucket[{i}]={buckets[i]} <= bucket[{i - 1}]={buckets[i - 1]}"
                )

        self.name = name
        self.label_names = label_names
        self.buckets = buckets
        self.max_entries = max_entries
        self.strict_labels = strict_labels
        self._label_names_set = frozenset(label_names)
        # Use OrderedDict for LRU eviction (oldest entries evicted first)
        self._values: OrderedDict[tuple[str, ...], HistogramValue] = OrderedDict()
        self._lock = threading.Lock()

    def _validate_labels(self, labels: dict[str, str]) -> None:
        """Validate labels against expected label names.

        Args:
            labels: Label key-value pairs to validate

        Raises:
            ValueError: If strict_labels is True and labels don't match expected names
        """
        if not self.strict_labels:
            return

        provided = set(labels.keys())
        expected = self._label_names_set

        missing = expected - provided
        extra = provided - expected

        errors = []
        if missing:
            errors.append(f"missing labels: {sorted(missing)}")
        if extra:
            errors.append(f"extra labels: {sorted(extra)}")

        if errors:
            raise ValueError(
                f"Label validation failed for metric '{self.name}': "
                f"{'; '.join(errors)}. "
                f"Expected: {sorted(expected)}, got: {sorted(provided)}"
            )

    def observe(self, value: float, **labels: str) -> None:
        """Record an observation with given labels.

        Args:
            value: Value to observe
            **labels: Label key-value pairs

        Raises:
            ValueError: If strict_labels is True and labels don't match expected names

        Thread-safety: The value holder reference is captured under the lock
        to prevent race conditions with reset() or eviction.
        """
        self._validate_labels(labels)
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                # Evict oldest entries if at capacity
                while len(self._values) >= self.max_entries:
                    self._values.popitem(last=False)  # Remove oldest (FIFO)
                self._values[key] = HistogramValue(buckets=self.buckets)
            else:
                # Move to end to mark as recently used
                self._values.move_to_end(key)
            # Capture reference under lock to prevent TOCTOU race with reset/eviction
            value_holder = self._values[key]
        # Safe to call outside lock - value_holder is our reference
        value_holder.observe(value)

    def get(self, **labels: str) -> dict[str, float | int | list[int] | list[float]]:
        """Get histogram snapshot for given labels."""
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                return {"sum": 0.0, "count": 0, "buckets": [], "bucket_bounds": []}
            # Move to end to mark as recently used
            self._values.move_to_end(key)
            return self._values[key].get_snapshot()

    def get_all(
        self,
    ) -> dict[tuple[str, ...], dict[str, float | int | list[int] | list[float]]]:
        """Get all histogram values with their labels."""
        with self._lock:
            return {k: v.get_snapshot() for k, v in self._values.items()}

    def _labels_to_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        """Convert labels dict to hashable key tuple."""
        return tuple(labels.get(name, "") for name in self.label_names)

    def labels_from_key(self, key: tuple[str, ...]) -> dict[str, str]:
        """Convert key tuple back to labels dict."""
        return dict(zip(self.label_names, key, strict=True))


@dataclass
class GaugeValue:
    """Thread-safe gauge value."""

    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, value: float) -> None:
        """Set the gauge value."""
        with self._lock:
            self.value = value

    def get(self) -> float:
        """Get current gauge value."""
        with self._lock:
            return self.value


class Gauge:
    """In-process gauge metric for tracking current values.

    Thread-safe gauge that tracks current state values.
    Uses bounded storage with LRU eviction to prevent unbounded memory growth.

    Label Validation:
        By default (strict_labels=False), missing labels silently default to empty
        string (""), which can cause incorrect state reporting (e.g., health checks
        reporting wrong handler status). For production use, ALWAYS set
        strict_labels=True.

    Example - Basic usage (lenient mode, NOT RECOMMENDED for production):
        gauge = Gauge("handler_health_status", ["handler"])
        gauge.set(1.0, handler="filesystem")  # healthy
        gauge.set(0.0, handler="filesystem")  # unhealthy
        gauge.set(1.0)  # DANGER: sets health for handler="" (unknown handler!)

    Example - Production usage (strict mode, RECOMMENDED):
        gauge = Gauge("health", ["handler", "region"], strict_labels=True)
        gauge.set(1.0, handler="fs", region="us-east")  # OK - all labels
        gauge.set(1.0, handler="fs")
        # Raises ValueError: missing labels: ['region']
        gauge.set(1.0, handler="fs", region="us", zone="a")
        # Raises ValueError: extra labels: ['zone']

    Warning:
        Gauges with missing labels can cause incorrect health reporting. If your
        alerting checks gauge.get(handler="postgres") but the gauge was set with
        no handler label, you'll get 0.0 (default) instead of the actual health
        status, potentially causing false alerts or missed outages.
    """

    def __init__(
        self,
        name: str,
        label_names: list[str],
        max_entries: int = DEFAULT_MAX_METRIC_ENTRIES,
        strict_labels: bool = False,
    ) -> None:
        """Initialize gauge with name and label names.

        Args:
            name: Metric name (e.g., "handler_health_status")
            label_names: List of label names
            max_entries: Maximum number of unique label combinations (default 10000).
                         Oldest entries are evicted when limit is exceeded.
            strict_labels: If True, validate that all required labels are provided
                          and no extra labels are passed. Raises ValueError on mismatch.
                          Default is False for backwards compatibility.
        """
        self.name = name
        self.label_names = label_names
        self.max_entries = max_entries
        self.strict_labels = strict_labels
        self._label_names_set = frozenset(label_names)
        # Use OrderedDict for LRU eviction (oldest entries evicted first)
        self._values: OrderedDict[tuple[str, ...], GaugeValue] = OrderedDict()
        self._lock = threading.Lock()

    def _validate_labels(self, labels: dict[str, str]) -> None:
        """Validate labels against expected label names.

        Args:
            labels: Label key-value pairs to validate

        Raises:
            ValueError: If strict_labels is True and labels don't match expected names
        """
        if not self.strict_labels:
            return

        provided = set(labels.keys())
        expected = self._label_names_set

        missing = expected - provided
        extra = provided - expected

        errors = []
        if missing:
            errors.append(f"missing labels: {sorted(missing)}")
        if extra:
            errors.append(f"extra labels: {sorted(extra)}")

        if errors:
            raise ValueError(
                f"Label validation failed for metric '{self.name}': "
                f"{'; '.join(errors)}. "
                f"Expected: {sorted(expected)}, got: {sorted(provided)}"
            )

    def set(self, value: float, **labels: str) -> None:
        """Set the gauge value with given labels.

        Args:
            value: Value to set
            **labels: Label key-value pairs

        Raises:
            ValueError: If strict_labels is True and labels don't match expected names

        Thread-safety: The value holder reference is captured under the lock
        to prevent race conditions with reset() or eviction.
        """
        self._validate_labels(labels)
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                # Evict oldest entries if at capacity
                while len(self._values) >= self.max_entries:
                    self._values.popitem(last=False)  # Remove oldest (FIFO)
                self._values[key] = GaugeValue()
            else:
                # Move to end to mark as recently used
                self._values.move_to_end(key)
            # Capture reference under lock to prevent TOCTOU race with reset/eviction
            value_holder = self._values[key]
        # Safe to call outside lock - value_holder is our reference
        value_holder.set(value)

    def get(self, **labels: str) -> float:
        """Get gauge value for given labels."""
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                return 0.0
            # Move to end to mark as recently used
            self._values.move_to_end(key)
            return self._values[key].get()

    def get_all(self) -> dict[tuple[str, ...], float]:
        """Get all gauge values with their labels."""
        with self._lock:
            return {k: v.get() for k, v in self._values.items()}

    def _labels_to_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        """Convert labels dict to hashable key tuple."""
        return tuple(labels.get(name, "") for name in self.label_names)

    def labels_from_key(self, key: tuple[str, ...]) -> dict[str, str]:
        """Convert key tuple back to labels dict."""
        return dict(zip(self.label_names, key, strict=True))


class MetricsRegistry:
    """Registry for in-process metrics.

    Provides a central place to access all metrics for the OmniMemory system.
    This is a singleton-like registry that holds all metrics instances.

    Thread-safety:
    - Uses double-checked locking pattern for safe singleton initialization
    - Uses instance-level RLock (_instance_lock) for operations across multiple metrics
    - The _initialized flag is checked inside the lock to prevent race conditions
    - get_all_metrics() and reset() are protected by locks for consistency
    - RLock is used to allow reentrant access (e.g., nested metric operations)

    Lock hierarchy (to prevent deadlocks):
    1. _class_lock: Acquired first for singleton creation/destruction
    2. _instance_lock: Acquired second for multi-metric operations
    3. Individual metric locks: Acquired last for single-metric operations

    Example:
        registry = MetricsRegistry()
        registry.memory_operation_total.inc(operation="store", status="success")
        registry.memory_storage_latency_ms.observe(45.2, operation="store")
    """

    _instance: MetricsRegistry | None = None
    _class_lock = (
        threading.Lock()
    )  # Class-level lock for singleton creation/destruction
    _instance_lock: threading.RLock  # Instance-level lock for multi-metric operations
    _initialized: bool  # Flag to track initialization state

    def __new__(cls) -> MetricsRegistry:
        """Singleton pattern for metrics registry with double-checked locking.

        Thread-safety is achieved by:
        1. Quick check outside lock (fast path for already-initialized case)
        2. Lock acquisition for creation
        3. Re-check inside lock (handles race between check and lock acquisition)
        4. Full initialization inside lock (prevents concurrent initialization)

        Note: We cache _instance in a local variable to prevent TOCTOU race
        conditions where _instance could change between the None check and
        the _initialized check.
        """
        # Fast path: instance already exists and is initialized
        # Cache in local variable to prevent TOCTOU race condition
        instance = cls._instance
        if instance is not None and getattr(instance, "_initialized", False):
            return instance

        # Slow path: need to create or initialize
        with cls._class_lock:
            # Double-check after acquiring lock
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False  # noqa: SLF001  # Intentional private access in singleton pattern
                # Create instance-level RLock for multi-metric operations
                # RLock allows reentrant access from the same thread
                cls._instance._instance_lock = threading.RLock()  # noqa: SLF001  # Intentional private access in singleton pattern

            # Initialize inside the lock to prevent concurrent initialization
            if not cls._instance._initialized:  # noqa: SLF001  # Intentional private access in singleton pattern
                cls._instance._do_initialize()  # noqa: SLF001  # Intentional private access in singleton pattern

            return cls._instance

    def __init__(self) -> None:
        """No-op init - all initialization done in __new__ under lock."""
        # Initialization is done in __new__ to ensure thread safety

    def _do_initialize(self) -> None:
        """Perform actual initialization (called under class lock from __new__).

        This method is only called once, under the class lock, ensuring
        thread-safe initialization of all metrics.
        """
        # Counter: Total operations by type and status
        self.memory_operation_total = Counter(
            name="memory_operation_total",
            label_names=["operation", "status", "handler"],
        )

        # Histogram: Storage operation latency
        self.memory_storage_latency_ms = Histogram(
            name="memory_storage_latency_ms",
            label_names=["operation", "handler"],
        )

        # Histogram: Retrieval operation latency
        self.memory_retrieval_latency_ms = Histogram(
            name="memory_retrieval_latency_ms",
            label_names=["operation", "handler"],
        )

        # Gauge: Handler health status (1=healthy, 0=unhealthy)
        self.handler_health_status = Gauge(
            name="handler_health_status",
            label_names=["handler"],
        )

        # Mark as initialized AFTER all metrics are created
        self._initialized = True

    def get_all_metrics(self) -> dict[str, dict[str, object]]:
        """Get snapshot of all metrics for reporting.

        Thread-safety: This method acquires the instance lock to ensure a
        consistent snapshot across all metrics. Without this lock, concurrent
        reset() calls could result in partially stale data.
        """
        with self._instance_lock:
            return {
                "memory_operation_total": {
                    "type": "counter",
                    "values": {
                        str(k): v
                        for k, v in self.memory_operation_total.get_all().items()
                    },
                },
                "memory_storage_latency_ms": {
                    "type": "histogram",
                    "values": {
                        str(k): v
                        for k, v in self.memory_storage_latency_ms.get_all().items()
                    },
                },
                "memory_retrieval_latency_ms": {
                    "type": "histogram",
                    "values": {
                        str(k): v
                        for k, v in self.memory_retrieval_latency_ms.get_all().items()
                    },
                },
                "handler_health_status": {
                    "type": "gauge",
                    "values": {
                        str(k): v
                        for k, v in self.handler_health_status.get_all().items()
                    },
                },
            }

    @classmethod
    def reset(cls) -> None:
        """Reset the registry by clearing all metrics data (primarily for testing).

        WARNING: This method should only be used in tests. Using it in production
        code may lead to inconsistent state if other code holds references to
        metric objects.

        Thread-safety: This method acquires both the class lock and instance lock:
        1. Class lock prevents concurrent singleton modifications
        2. Instance lock prevents concurrent get_all_metrics() from seeing partial state
        3. Individual metric locks ensure atomic clearing of each metric

        The instance itself is preserved, only the data is cleared.
        """
        # Warn if called outside tests (check for common test indicators)
        import sys
        import warnings

        in_test = any(
            "pytest" in module or "unittest" in module or "test" in module.lower()
            for module in sys.modules
        )
        if not in_test:
            warnings.warn(
                "MetricsRegistry.reset() called outside of tests. "
                "This may lead to inconsistent metrics state.",
                UserWarning,
                stacklevel=2,
            )

        with cls._class_lock:
            if cls._instance is not None and cls._instance._initialized:  # noqa: SLF001  # Intentional private access in singleton pattern
                # Acquire instance lock to prevent concurrent get_all_metrics()
                with cls._instance._instance_lock:  # noqa: SLF001  # Intentional private access in singleton pattern
                    # Clear all metrics data from the existing instance
                    # This preserves references while resetting state
                    # Note: We do NOT set _instance = None to avoid orphaning references
                    cls._instance._clear_all_metrics()  # noqa: SLF001  # Intentional private access in singleton pattern

    @classmethod
    def _reset_instance_for_testing(cls) -> None:
        """Fully reset the singleton instance (TESTING ONLY).

        WARNING: This method WILL orphan any existing references to the registry
        or its metrics. Only use this in tests that need a completely fresh instance
        and do not have any code holding references to the old instance.

        For most tests, use reset() instead which preserves references.

        Thread-safety: This method acquires both class lock and instance lock:
        1. Class lock prevents concurrent singleton modifications
        2. Instance lock ensures atomic clearing before instance destruction
        It will invalidate any references obtained before the reset completes.
        """
        with cls._class_lock:
            if cls._instance is not None and cls._instance._initialized:  # noqa: SLF001  # Intentional private access in singleton pattern
                # Clear metrics first to help any stale references
                with cls._instance._instance_lock:  # noqa: SLF001  # Intentional private access in singleton pattern
                    cls._instance._clear_all_metrics()  # noqa: SLF001  # Intentional private access in singleton pattern
            # Reset the instance - next access will create a new one
            cls._instance = None

    def _clear_all_metrics(self) -> None:
        """Clear all data from metrics atomically.

        Thread-safety: This method must be called while holding both:
        1. cls._class_lock (for singleton protection)
        2. self._instance_lock (for multi-metric operation atomicity)

        It acquires each metric's individual lock to safely clear the data,
        ensuring no metric operation is in progress during the clear.
        """
        # Clear counter values - intentional private access for thread-safe metric clearing
        with self.memory_operation_total._lock:  # noqa: SLF001
            self.memory_operation_total._values.clear()  # noqa: SLF001

        # Clear histogram values
        with self.memory_storage_latency_ms._lock:  # noqa: SLF001
            self.memory_storage_latency_ms._values.clear()  # noqa: SLF001
        with self.memory_retrieval_latency_ms._lock:  # noqa: SLF001
            self.memory_retrieval_latency_ms._values.clear()  # noqa: SLF001

        # Clear gauge values
        with self.handler_health_status._lock:  # noqa: SLF001
            self.handler_health_status._values.clear()  # noqa: SLF001


# Global metrics registry instance
metrics_registry = MetricsRegistry()

# TraceLevel is imported from ..models.utils.model_observability


class OperationType(Enum):
    """Operation type enumeration for categorizing operations."""

    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_SEARCH = "memory_search"
    INTELLIGENCE_PROCESS = "intelligence_process"
    HEALTH_CHECK = "health_check"
    MIGRATION = "migration"
    CLEANUP = "cleanup"
    EXTERNAL_API = "external_api"


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""

    start_time: float
    end_time: float | None = None
    duration: float | None = None
    memory_usage_start: float | None = None
    memory_usage_end: float | None = None
    memory_delta: float | None = None
    success: bool | None = None
    error_type: str | None = None


# CorrelationContext is imported from ..models.utils.model_observability


class ObservabilityManager:
    """
    Comprehensive observability manager for OmniMemory.

    Provides:
    - Correlation ID management and propagation
    - Distributed tracing support
    - Performance monitoring
    - Enhanced logging with context

    Thread-safety:
        All trace storage operations are protected by a lock to ensure
        thread-safe access to the active traces dictionary.

    Memory bounds:
        Active traces are stored in a bounded OrderedDict with LRU eviction.
        When max_active_traces is reached, the oldest trace is evicted to
        make room for new traces. This prevents unbounded memory growth from
        long-running or abandoned traces.
    """

    def __init__(
        self,
        max_active_traces: int = DEFAULT_MAX_ACTIVE_TRACES,
    ) -> None:
        """Initialize the observability manager.

        Args:
            max_active_traces: Maximum number of active traces to store.
                              When this limit is reached, the oldest trace
                              is evicted. Default is 1000.
        """
        self.max_active_traces = max_active_traces
        # Use OrderedDict for bounded storage with LRU eviction
        self._active_traces: OrderedDict[str, PerformanceMetrics] = OrderedDict()
        self._traces_lock = threading.Lock()
        self._logger = structlog.get_logger(__name__)

    @asynccontextmanager
    async def correlation_context(
        self,
        correlation_id: str | None = None,
        request_id: str | None = None,
        user_id: str | None = None,
        operation: str | None = None,
        trace_level: TraceLevel = TraceLevel.INFO,
        **metadata: MetadataValue,
    ) -> AsyncGenerator[CorrelationContext, None]:
        """
        Async context manager for correlation tracking.

        Args:
            correlation_id: Unique correlation identifier
            request_id: Request identifier
            user_id: User identifier
            operation: Operation name
            trace_level: Tracing level
            **metadata: Additional metadata
        """
        # Validate correlation ID if provided
        if correlation_id and not validate_correlation_id(correlation_id):
            raise ValueError(f"Invalid correlation ID format: {correlation_id}")

        # Sanitize metadata values and convert to ModelMetadata
        metadata_pairs = [
            ModelKeyValuePair(key=key, value=str(sanitize_metadata_value(value)))
            for key, value in metadata.items()
            if sanitize_metadata_value(value) is not None
        ]
        sanitized_metadata = ModelMetadata(pairs=metadata_pairs)

        # Create context
        context = CorrelationContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            request_id=request_id,
            user_id=user_id,
            operation=operation,
            parent_correlation_id=correlation_id_var.get(),
            trace_level=trace_level,
            metadata=sanitized_metadata,
        )

        # Set context variables
        correlation_token = correlation_id_var.set(context.correlation_id)
        request_token = request_id_var.set(context.request_id)
        user_token = user_id_var.set(context.user_id)
        operation_token = operation_var.set(context.operation)

        try:
            self._logger.info(
                "correlation_context_started",
                correlation_id=context.correlation_id,
                request_id=context.request_id,
                user_id=context.user_id,
                operation=context.operation,
                trace_level=context.trace_level.value,
                metadata=context.metadata,
            )

            yield context

        except Exception as e:
            self._logger.error(
                "correlation_context_error",
                correlation_id=context.correlation_id,
                error=_sanitize_error(e),
                error_type=type(e).__name__,
            )
            raise
        finally:
            # Reset context variables
            correlation_id_var.reset(correlation_token)
            request_id_var.reset(request_token)
            user_id_var.reset(user_token)
            operation_var.reset(operation_token)

            self._logger.info(
                "correlation_context_ended",
                correlation_id=context.correlation_id,
                operation=context.operation,
            )

    @asynccontextmanager
    async def trace_operation(
        self,
        operation_name: str,
        operation_type: OperationType,
        trace_performance: bool = True,
        **additional_context: MetadataValue,
    ) -> AsyncGenerator[str, None]:
        """
        Async context manager for operation tracing.

        Args:
            operation_name: Name of the operation being traced
            operation_type: Type of operation
            trace_performance: Whether to track performance metrics
            **additional_context: Additional context for tracing
        """
        trace_id = str(uuid.uuid4())
        correlation_id = correlation_id_var.get()

        # Initialize performance metrics if requested
        start_memory: float | None = None
        if trace_performance:
            # Only track memory if psutil is available
            if _psutil_available and psutil is not None:
                try:
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024  # MB
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                    psutil.Error,
                    OSError,
                    AttributeError,
                ) as e:
                    # Gracefully handle all psutil errors:
                    # - NoSuchProcess: process terminated
                    # - AccessDenied: insufficient permissions
                    # - ZombieProcess: process is zombie state
                    # - Error: base psutil exception
                    # - OSError: OS-level errors
                    # - AttributeError: corrupted psutil module
                    self._logger.debug(
                        "psutil_memory_tracking_unavailable",
                        reason=type(e).__name__,
                        phase="start",
                    )
                    start_memory = None

            metrics = PerformanceMetrics(
                start_time=time.time(), memory_usage_start=start_memory
            )
            # Add trace with bounded storage (evict oldest if at capacity)
            with self._traces_lock:
                # Evict oldest traces if at capacity
                while len(self._active_traces) >= self.max_active_traces:
                    evicted_id, evicted_metrics = self._active_traces.popitem(
                        last=False
                    )
                    self._logger.warning(
                        "trace_evicted_due_to_capacity",
                        evicted_trace_id=evicted_id,
                        evicted_duration=evicted_metrics.duration,
                        max_active_traces=self.max_active_traces,
                    )
                self._active_traces[trace_id] = metrics

        try:
            self._logger.info(
                "operation_started",
                trace_id=trace_id,
                correlation_id=correlation_id,
                operation_name=operation_name,
                operation_type=operation_type.value,
                **additional_context,
            )

            yield trace_id

            # Mark as successful (thread-safe)
            if trace_performance:
                with self._traces_lock:
                    if trace_id in self._active_traces:
                        self._active_traces[trace_id].success = True

        except Exception as e:
            # Mark as failed and log error (thread-safe)
            if trace_performance:
                with self._traces_lock:
                    if trace_id in self._active_traces:
                        self._active_traces[trace_id].success = False
                        self._active_traces[trace_id].error_type = type(e).__name__

            self._logger.error(
                "operation_failed",
                trace_id=trace_id,
                correlation_id=correlation_id,
                operation_name=operation_name,
                operation_type=operation_type.value,
                error=_sanitize_error(e),
                error_type=type(e).__name__,
                **additional_context,
            )
            raise
        finally:
            # Complete performance metrics if requested (thread-safe)
            metrics_final: PerformanceMetrics | None = None
            if trace_performance:
                # Get trace reference under lock
                with self._traces_lock:
                    if trace_id in self._active_traces:
                        metrics_final = self._active_traces[trace_id]

                # Process metrics outside lock (I/O operations)
                if metrics_final is not None:
                    metrics_final.end_time = time.time()
                    metrics_final.duration = (
                        metrics_final.end_time - metrics_final.start_time
                    )

                    if metrics_final.memory_usage_start is not None:
                        # Only track memory delta if psutil is available
                        if _psutil_available and psutil is not None:
                            try:
                                process = psutil.Process()
                                end_memory = (
                                    process.memory_info().rss / 1024 / 1024
                                )  # MB
                                metrics_final.memory_usage_end = end_memory
                                metrics_final.memory_delta = (
                                    end_memory - metrics_final.memory_usage_start
                                )
                            except (
                                psutil.NoSuchProcess,
                                psutil.AccessDenied,
                                psutil.ZombieProcess,
                                psutil.Error,
                                OSError,
                                AttributeError,
                            ) as e:
                                # Gracefully handle all psutil errors:
                                # - NoSuchProcess: process terminated
                                # - AccessDenied: insufficient permissions
                                # - ZombieProcess: process is zombie state
                                # - Error: base psutil exception
                                # - OSError: OS-level errors
                                # - AttributeError: corrupted psutil module
                                self._logger.debug(
                                    "psutil_memory_tracking_unavailable",
                                    reason=type(e).__name__,
                                    phase="end",
                                )

                    self._logger.info(
                        "operation_completed",
                        trace_id=trace_id,
                        correlation_id=correlation_id,
                        operation_name=operation_name,
                        operation_type=operation_type.value,
                        duration=metrics_final.duration,
                        memory_delta=metrics_final.memory_delta,
                        success=metrics_final.success,
                        error_type=metrics_final.error_type,
                        **additional_context,
                    )

                    # Clean up completed trace under lock
                    with self._traces_lock:
                        if trace_id in self._active_traces:
                            del self._active_traces[trace_id]

    def get_current_context(self) -> dict[str, str | None]:
        """Get current correlation context."""
        return {
            "correlation_id": correlation_id_var.get(),
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            "operation": operation_var.get(),
        }

    def get_performance_metrics(self) -> dict[str, PerformanceMetrics]:
        """Get current performance metrics for active traces (thread-safe)."""
        with self._traces_lock:
            return dict(self._active_traces)

    def log_with_context(
        self, level: str, message: str, **additional_fields: MetadataValue
    ) -> None:
        """Log a message with current correlation context."""
        context = self.get_current_context()

        log_method = getattr(self._logger, level.lower(), self._logger.info)
        log_method(message, **context, **additional_fields)


# Global observability manager instance
observability_manager = ObservabilityManager()


# Convenience functions for common patterns
@asynccontextmanager
async def correlation_context(
    correlation_id: str | None = None,
    request_id: str | None = None,
    user_id: str | None = None,
    operation: str | None = None,
    trace_level: TraceLevel = TraceLevel.INFO,
    **metadata: MetadataValue,
) -> AsyncGenerator[CorrelationContext, None]:
    """Convenience function for correlation context management."""
    async with observability_manager.correlation_context(
        correlation_id=correlation_id,
        request_id=request_id,
        user_id=user_id,
        operation=operation,
        trace_level=trace_level,
        **metadata,
    ) as ctx:
        yield ctx


@asynccontextmanager
async def trace_operation(
    operation_name: str,
    operation_type: OperationType | str,
    trace_performance: bool = True,
    **context: MetadataValue,
) -> AsyncGenerator[str, None]:
    """Convenience function for operation tracing."""
    if isinstance(operation_type, str):
        # Try to convert string to OperationType
        try:
            operation_type = OperationType(operation_type)
        except ValueError:
            # Default to external API if unknown
            operation_type = OperationType.EXTERNAL_API

    async with observability_manager.trace_operation(
        operation_name=operation_name,
        operation_type=operation_type,
        trace_performance=trace_performance,
        **context,
    ) as trace_id:
        yield trace_id


def get_correlation_id() -> str | None:
    """Get current correlation ID from context."""
    return correlation_id_var.get()


def get_request_id() -> str | None:
    """Get current request ID from context."""
    return request_id_var.get()


def log_with_correlation(level: str, message: str, **fields: MetadataValue) -> None:
    """Log a message with correlation context."""
    observability_manager.log_with_context(level, message, **fields)


def inject_correlation_context(func: F) -> F:
    """Decorator to inject correlation context into function logs."""

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        context = observability_manager.get_current_context()
        logger.info(
            f"function_called_{func.__name__}",
            **context,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys()),
        )
        try:
            result = func(*args, **kwargs)
            logger.info(f"function_completed_{func.__name__}", **context, success=True)
            return result
        except Exception as e:
            logger.error(
                f"function_failed_{func.__name__}",
                **context,
                error=_sanitize_error(e),
                error_type=type(e).__name__,
            )
            raise

    return wrapper  # type: ignore[return-value]


def inject_correlation_context_async(func: F) -> F:
    """Async decorator to inject correlation context into function logs."""

    @functools.wraps(func)
    async def wrapper(*args: object, **kwargs: object) -> object:
        context = observability_manager.get_current_context()
        logger.info(
            f"async_function_called_{func.__name__}",
            **context,
            args_count=len(args),
            kwargs_keys=list(kwargs.keys()),
        )
        try:
            result = await cast("Awaitable[object]", func(*args, **kwargs))
            logger.info(
                f"async_function_completed_{func.__name__}", **context, success=True
            )
            return result
        except Exception as e:
            logger.error(
                f"async_function_failed_{func.__name__}",
                **context,
                error=_sanitize_error(e),
                error_type=type(e).__name__,
            )
            raise

    return wrapper  # type: ignore[return-value]


# === HANDLER OBSERVABILITY WRAPPER ===
# "One wrapper, one log line, no payload" pattern for P1C observability


@dataclass
class HandlerMetrics:
    """Metrics captured during handler execution.

    This dataclass holds all metrics collected during a single handler operation,
    ready to be logged and recorded.
    """

    correlation_id: str
    operation: str
    handler: str
    status: Literal["success", "failure"]
    latency_ms: float
    error_type: str | None = None
    error_message: str | None = None


def _get_safe_content_metadata(
    content: str | None,
    field_name: str = "content",
) -> dict[str, str | int | bool]:
    """Extract safe metadata from content without logging PII.

    Instead of logging raw content, we log:
    - Length of content
    - Hash prefix (first 8 chars of SHA-256)
    - Whether content exists

    Args:
        content: Raw content string (may contain PII)
        field_name: Name prefix for the metadata fields

    Returns:
        Dict with safe metadata about the content
    """
    if content is None:
        return {
            f"{field_name}_exists": False,
            f"{field_name}_len": 0,
        }

    import hashlib

    content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
    return {
        f"{field_name}_exists": True,
        f"{field_name}_len": len(content),
        f"{field_name}_hash": content_hash,
    }


class HandlerObservabilityWrapper:
    """Wrapper for handler operations providing observability.

    Implements the "one wrapper, one log line, no payload" pattern:
    - Wraps handler execution with timing
    - Records metrics (latency histogram, operation counter, health gauge)
    - Emits single structured log event per operation
    - Ensures no PII in log output

    Configuration Options:
        validate_log_schema: If True, validates all log entries against the
                            StructuredLogEntry schema before emission. This catches
                            schema violations early but has a small performance cost.
                            Default is False for backwards compatibility.
                            RECOMMENDED for development and testing.

    Example usage:
        ```python
        wrapper = HandlerObservabilityWrapper(handler_name="filesystem")

        async def store_memory(request: MemoryStoreRequest) -> MemoryStoreResponse:
            async with wrapper.observe_operation(
                operation="store",
                correlation_id=str(request.correlation_id),
            ) as ctx:
                # ... perform actual storage operation ...
                result = await do_storage(request)
                return result
        ```

    Example with schema validation:
        ```python
        # Enable schema validation (recommended for development)
        wrapper = HandlerObservabilityWrapper(
            handler_name="filesystem",
            validate_log_schema=True
        )
        ```
    """

    # Pattern for valid handler names: alphanumeric, underscore, hyphen, max 64 chars
    _HANDLER_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

    def __init__(
        self,
        handler_name: str,
        registry: MetricsRegistry | None = None,
        validate_log_schema: bool = False,
    ) -> None:
        """Initialize the wrapper.

        Args:
            handler_name: Name of the handler (e.g., "filesystem", "postgresql").
                         Must be a non-empty string containing only alphanumeric
                         characters, underscores, and hyphens (max 64 characters).
            registry: Optional metrics registry (defaults to current singleton)
            validate_log_schema: If True, validate log entries against the
                                StructuredLogEntry schema before emission.
                                Default is False for backwards compatibility.

        Raises:
            ValueError: If handler_name is empty or doesn't match
                       the required pattern.
        """
        if not handler_name:
            raise ValueError("handler_name must be a non-empty string")

        # Validate handler_name matches safe pattern
        if not self._HANDLER_NAME_PATTERN.match(handler_name):
            raise ValueError(
                f"handler_name must contain only alphanumeric characters, "
                f"underscores, and hyphens (max 64 characters), got: {handler_name!r}"
            )

        self.handler_name = handler_name
        self._custom_registry: MetricsRegistry | None = registry
        self._validate_log_schema = validate_log_schema
        self._logger = structlog.get_logger(f"omnimemory.handler.{handler_name}")

        # Set initial health status to healthy
        self.registry.handler_health_status.set(1.0, handler=handler_name)

    @property
    def registry(self) -> MetricsRegistry:
        """Get the metrics registry (always returns current singleton if not custom)."""
        if self._custom_registry is not None:
            return self._custom_registry
        return MetricsRegistry()

    @asynccontextmanager
    async def observe_operation(
        self,
        operation: str,
        correlation_id: str | None = None,
    ) -> AsyncGenerator[dict[str, str], None]:
        """Context manager for observing handler operations.

        Implements the core observability pattern:
        1. Start timer
        2. Execute operation in try/except
        3. Record histogram for latency
        4. Increment counter for operation/status
        5. Update health gauge
        6. Emit single structured log event

        Args:
            operation: Operation name (e.g., "store", "retrieve", "delete")
            correlation_id: Request correlation ID (generated if not provided)

        Yields:
            Dict with context info (correlation_id, operation, handler)
        """
        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        # Validate correlation ID format
        if not validate_correlation_id(correlation_id):
            correlation_id = str(uuid.uuid4())

        # Set correlation context
        token = correlation_id_var.set(correlation_id)

        # Context to yield
        ctx = {
            "correlation_id": correlation_id,
            "operation": operation,
            "handler": self.handler_name,
        }

        # Start timing
        start_time = time.perf_counter()
        status: Literal["success", "failure"] = "success"
        error_type: str | None = None
        error_message: str | None = None

        try:
            yield ctx

        except Exception as e:
            status = "failure"
            error_type = type(e).__name__
            error_message = _sanitize_error(e)
            raise

        finally:
            # Calculate latency
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Record metrics
            self._record_metrics(
                operation=operation,
                status=status,
                latency_ms=latency_ms,
            )

            # Emit single structured log event
            self._emit_log_event(
                HandlerMetrics(
                    correlation_id=correlation_id,
                    operation=operation,
                    handler=self.handler_name,
                    status=status,
                    latency_ms=latency_ms,
                    error_type=error_type,
                    error_message=error_message,
                )
            )

            # Reset correlation context
            correlation_id_var.reset(token)

    def _record_metrics(
        self,
        operation: str,
        status: Literal["success", "failure"],
        latency_ms: float,
    ) -> None:
        """Record all metrics for the operation.

        Args:
            operation: Operation name
            status: Operation status (success/failure)
            latency_ms: Operation latency in milliseconds
        """
        # Increment operation counter
        self.registry.memory_operation_total.inc(
            operation=operation,
            status=status,
            handler=self.handler_name,
        )

        # Record latency histogram
        # Use appropriate histogram based on operation type
        if operation in ("store", "update", "delete"):
            self.registry.memory_storage_latency_ms.observe(
                latency_ms,
                operation=operation,
                handler=self.handler_name,
            )
        elif operation in ("retrieve", "list", "search"):
            self.registry.memory_retrieval_latency_ms.observe(
                latency_ms,
                operation=operation,
                handler=self.handler_name,
            )
        else:
            # Default to storage histogram for unknown operations
            self.registry.memory_storage_latency_ms.observe(
                latency_ms,
                operation=operation,
                handler=self.handler_name,
            )

        # Update health gauge based on status
        # Simple policy: success = healthy (1.0), failure = unhealthy (0.0)
        # In production, you might use a sliding window or circuit breaker
        if status == "success":
            self.registry.handler_health_status.set(1.0, handler=self.handler_name)
        else:
            self.registry.handler_health_status.set(0.0, handler=self.handler_name)

    def _emit_log_event(self, metrics: HandlerMetrics) -> None:
        """Emit a single structured log event for the operation.

        This implements the "one log line" part of the pattern.
        All relevant metrics and context are included in a single event.

        Log Schema (consistent with module-level documentation):
            Required fields (ALWAYS present):
                - correlation_id (str): Unique request correlation identifier
                - operation (str): Operation name (e.g., "store", "retrieve")
                - handler (str): Handler name (e.g., "filesystem", "postgresql")
                - status (str): "success" or "failure"
                - latency_ms (float): Latency in milliseconds (2 decimal places)
                - timestamp (str): ISO8601 UTC timestamp (YYYY-MM-DDTHH:MM:SS.sssZ)

            Optional fields (only on failure):
                - error_type (str): Exception class name
                - error_message (str): Sanitized error message (PII-safe)

        Schema Compliance:
            This method strictly adheres to the documented schema at module level.
            All required fields are ALWAYS present (never None). The timestamp
            format is ISO8601 compliant with millisecond precision and 'Z' suffix
            indicating UTC timezone.

            If validate_log_schema=True was set in __init__, each log entry is
            validated against the StructuredLogEntry Pydantic model before emission.
            This catches schema violations early during development.

        Args:
            metrics: Captured metrics for the operation
        """
        # Generate ISO8601 timestamp in UTC with millisecond precision
        # Format: YYYY-MM-DDTHH:MM:SS.sssZ (e.g., "2025-01-19T12:34:56.789Z")
        # Using timezone-aware datetime.now(timezone.utc) instead of deprecated utcnow()
        timestamp = (
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )

        # Build log data with ALL required fields (schema compliance)
        # Note: All required fields must be non-None strings/floats
        log_data: dict[str, str | float] = {
            "correlation_id": metrics.correlation_id,  # Required: str
            "operation": metrics.operation,  # Required: str
            "handler": metrics.handler,  # Required: str
            "status": metrics.status,  # Required: "success" | "failure"
            "latency_ms": round(metrics.latency_ms, 2),  # Required: float (2 decimals)
            "timestamp": timestamp,  # Required: ISO8601 UTC string
        }

        # Add optional fields ONLY on failure (schema compliance)
        # These fields are omitted entirely on success, not set to None
        if metrics.status == "failure":
            if metrics.error_type is not None:
                log_data["error_type"] = metrics.error_type
            if metrics.error_message is not None:
                log_data["error_message"] = metrics.error_message

        # Validate against StructuredLogEntry schema if enabled
        # This catches schema violations early during development/testing
        if self._validate_log_schema:
            try:
                StructuredLogEntry.model_validate(log_data)
            except Exception as validation_error:
                # Log validation failure but still emit the log event
                # This prevents observability failures from breaking the application
                self._logger.warning(
                    "omnimemory.handler.log_schema_validation_failed",
                    error=str(validation_error),
                    handler=self.handler_name,
                    operation=metrics.operation,
                )

        # Log at appropriate level based on status
        if metrics.status == "success":
            self._logger.info("omnimemory.handler.operation", **log_data)
        else:
            self._logger.error("omnimemory.handler.operation", **log_data)

    def mark_healthy(self) -> None:
        """Explicitly mark handler as healthy."""
        self.registry.handler_health_status.set(1.0, handler=self.handler_name)

    def mark_unhealthy(self) -> None:
        """Explicitly mark handler as unhealthy."""
        self.registry.handler_health_status.set(0.0, handler=self.handler_name)

    def get_handler_stats(self) -> dict[str, object]:
        """Get statistics for this handler.

        Returns:
            Dict containing counter totals, histogram stats, and health status

        Note:
            This method filters metrics by converting keys back to labeled dicts
            using each metric's `labels_from_key()` method, then checking the
            'handler' label explicitly. This is safer than position-based matching
            because it doesn't depend on the order of labels in label_names.
        """
        # Get all counter values for this handler
        # Use labels_from_key() for explicit label matching (not position-based)
        counter_metric = self.registry.memory_operation_total
        all_counters = counter_metric.get_all()
        handler_counters = {
            k: v
            for k, v in all_counters.items()
            if counter_metric.labels_from_key(k).get("handler") == self.handler_name
        }

        # Get histogram stats for this handler
        # Use labels_from_key() for explicit label matching
        storage_metric = self.registry.memory_storage_latency_ms
        retrieval_metric = self.registry.memory_retrieval_latency_ms

        storage_histograms = storage_metric.get_all()
        retrieval_histograms = retrieval_metric.get_all()

        handler_storage = {
            k: v
            for k, v in storage_histograms.items()
            if storage_metric.labels_from_key(k).get("handler") == self.handler_name
        }
        handler_retrieval = {
            k: v
            for k, v in retrieval_histograms.items()
            if retrieval_metric.labels_from_key(k).get("handler") == self.handler_name
        }

        # Get health status
        health = self.registry.handler_health_status.get(handler=self.handler_name)

        return {
            "handler": self.handler_name,
            "health_status": health,
            "operation_counts": handler_counters,
            "storage_latency": handler_storage,
            "retrieval_latency": handler_retrieval,
        }
