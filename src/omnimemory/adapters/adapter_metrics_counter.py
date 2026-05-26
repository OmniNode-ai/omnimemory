# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""In-process metrics counters, histograms, gauges, and label validation for OmniMemory.

Extracted from observability.py (OMN-11580).

- LabelValidationError and validate_metric_labels for label enforcement
- Counter, Histogram, Gauge with bounded LRU storage
- MetricsRegistry singleton for system-wide metric access
- Log schema validation utilities
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import structlog

from ..models.utils.model_structured_log_entry import (
    ModelStructuredLogEntry,
    TraceLevel,
)
from .adapter_error_sanitizer import SanitizationLevel
from .adapter_error_sanitizer import sanitize_error as _base_sanitize_error

# Re-export TraceLevel for public API
__all__ = ["TraceLevel"]

# Alias for internal use
StructuredLogEntry = ModelStructuredLogEntry

# Type alias for metadata values - supports common serializable types
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
    """Validate labels against required and allowed sets."""
    if not required_labels:
        raise ValueError("required_labels must not be empty")

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


def validate_log_entry(
    log_data: dict[str, object],
    raise_on_error: bool = True,
) -> StructuredLogEntry | None:
    """Validate a log entry against the structured log schema."""
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
    """Create a validated log entry with automatic timestamp generation."""
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


# === SECURITY VALIDATION FUNCTIONS ===


def validate_correlation_id(correlation_id: str) -> bool:
    """Validate correlation ID format to prevent injection attacks."""
    if not correlation_id:
        return False
    pattern = r"^[a-zA-Z0-9\-_]{1,64}$"
    return re.match(pattern, correlation_id) is not None


def sanitize_metadata_value(value: object) -> MetadataValue:
    """Sanitize metadata values to prevent injection attacks."""
    if isinstance(value, str):
        sanitized = re.sub(r'[<>"\'\\\n\r\t]', "", value)
        return sanitized[:1000]
    elif isinstance(value, bool) or isinstance(value, int | float):
        return value
    elif value is None:
        return None
    else:
        return sanitize_metadata_value(str(value))


def _sanitize_error(error: Exception) -> str:
    """Sanitize error messages to prevent information disclosure in logs."""
    return _base_sanitize_error(
        error, context="observability", level=SanitizationLevel.STANDARD
    )


# === IN-PROCESS METRICS ===

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

DEFAULT_MAX_METRIC_ENTRIES: int = 10000
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
    """

    def __init__(
        self,
        name: str,
        label_names: list[str],
        max_entries: int = DEFAULT_MAX_METRIC_ENTRIES,
        strict_labels: bool = False,
    ) -> None:
        self.name = name
        self.label_names = label_names
        self.max_entries = max_entries
        self.strict_labels = strict_labels
        self._label_names_set = frozenset(label_names)
        self._values: OrderedDict[tuple[str, ...], CounterValue] = OrderedDict()
        self._lock = threading.Lock()

    def _validate_labels(self, labels: dict[str, str]) -> None:
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
        """Increment the counter with given labels."""
        self._validate_labels(labels)
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                while len(self._values) >= self.max_entries:
                    self._values.popitem(last=False)
                self._values[key] = CounterValue()
            else:
                self._values.move_to_end(key)
            value_holder = self._values[key]
        value_holder.inc(amount)

    def get(self, **labels: str) -> int:
        """Get counter value for given labels."""
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                return 0
            self._values.move_to_end(key)
            return self._values[key].get()

    def get_all(self) -> dict[tuple[str, ...], int]:
        """Get all counter values with their labels."""
        with self._lock:
            return {k: v.get() for k, v in self._values.items()}

    def _labels_to_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        return tuple(labels.get(name, "") for name in self.label_names)

    def labels_from_key(self, key: tuple[str, ...]) -> dict[str, str]:
        return dict(zip(self.label_names, key, strict=True))

    def reset(self) -> None:
        """Clear all counter data (for testing)."""
        with self._lock:
            self._values.clear()


@dataclass
class HistogramValue:
    """Thread-safe histogram value with buckets."""

    buckets: tuple[float, ...]
    bucket_counts: list[int] = field(default_factory=list)
    sum_value: float = 0.0
    count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.bucket_counts:
            self.bucket_counts = [0] * (len(self.buckets) + 1)

    def observe(self, value: float) -> None:
        with self._lock:
            self.sum_value += value
            self.count += 1
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    self.bucket_counts[i] += 1
            self.bucket_counts[-1] += 1

    def get_snapshot(self) -> dict[str, float | int | list[int] | list[float]]:
        with self._lock:
            return {
                "sum": self.sum_value,
                "count": self.count,
                "buckets": list(self.bucket_counts),
                "bucket_bounds": list(self.buckets) + [float("inf")],
            }


class Histogram:
    """In-process histogram metric for tracking distributions."""

    def __init__(
        self,
        name: str,
        label_names: list[str],
        buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS,
        max_entries: int = DEFAULT_MAX_METRIC_ENTRIES,
        strict_labels: bool = False,
    ) -> None:
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
        self._values: OrderedDict[tuple[str, ...], HistogramValue] = OrderedDict()
        self._lock = threading.Lock()

    def _validate_labels(self, labels: dict[str, str]) -> None:
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
        """Record an observation with given labels."""
        self._validate_labels(labels)
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                while len(self._values) >= self.max_entries:
                    self._values.popitem(last=False)
                self._values[key] = HistogramValue(buckets=self.buckets)
            else:
                self._values.move_to_end(key)
            value_holder = self._values[key]
        value_holder.observe(value)

    def get(self, **labels: str) -> dict[str, float | int | list[int] | list[float]]:
        """Get histogram snapshot for given labels."""
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                return {"sum": 0.0, "count": 0, "buckets": [], "bucket_bounds": []}
            self._values.move_to_end(key)
            return self._values[key].get_snapshot()

    def get_all(
        self,
    ) -> dict[tuple[str, ...], dict[str, float | int | list[int] | list[float]]]:
        """Get all histogram values with their labels."""
        with self._lock:
            return {k: v.get_snapshot() for k, v in self._values.items()}

    def _labels_to_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        return tuple(labels.get(name, "") for name in self.label_names)

    def labels_from_key(self, key: tuple[str, ...]) -> dict[str, str]:
        return dict(zip(self.label_names, key, strict=True))

    def reset(self) -> None:
        """Clear all histogram data (for testing)."""
        with self._lock:
            self._values.clear()


@dataclass
class GaugeValue:
    """Thread-safe gauge value."""

    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, value: float) -> None:
        with self._lock:
            self.value = value

    def get(self) -> float:
        with self._lock:
            return self.value


class Gauge:
    """In-process gauge metric for tracking current values."""

    def __init__(
        self,
        name: str,
        label_names: list[str],
        max_entries: int = DEFAULT_MAX_METRIC_ENTRIES,
        strict_labels: bool = False,
    ) -> None:
        self.name = name
        self.label_names = label_names
        self.max_entries = max_entries
        self.strict_labels = strict_labels
        self._label_names_set = frozenset(label_names)
        self._values: OrderedDict[tuple[str, ...], GaugeValue] = OrderedDict()
        self._lock = threading.Lock()

    def _validate_labels(self, labels: dict[str, str]) -> None:
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
        """Set the gauge value with given labels."""
        self._validate_labels(labels)
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                while len(self._values) >= self.max_entries:
                    self._values.popitem(last=False)
                self._values[key] = GaugeValue()
            else:
                self._values.move_to_end(key)
            value_holder = self._values[key]
        value_holder.set(value)

    def get(self, **labels: str) -> float:
        """Get gauge value for given labels."""
        key = self._labels_to_key(labels)
        with self._lock:
            if key not in self._values:
                return 0.0
            self._values.move_to_end(key)
            return self._values[key].get()

    def get_all(self) -> dict[tuple[str, ...], float]:
        """Get all gauge values with their labels."""
        with self._lock:
            return {k: v.get() for k, v in self._values.items()}

    def _labels_to_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        return tuple(labels.get(name, "") for name in self.label_names)

    def labels_from_key(self, key: tuple[str, ...]) -> dict[str, str]:
        return dict(zip(self.label_names, key, strict=True))

    def reset(self) -> None:
        """Clear all gauge data (for testing)."""
        with self._lock:
            self._values.clear()


class MetricsRegistry:
    """Registry for in-process metrics.

    Provides a central place to access all metrics for the OmniMemory system.
    Singleton with thread-safe double-checked locking initialization.
    """

    _instance: MetricsRegistry | None = None
    _class_lock = threading.Lock()
    _instance_lock: threading.RLock
    _initialized: bool

    def __new__(cls) -> MetricsRegistry:
        instance = cls._instance
        if instance is not None and getattr(instance, "_initialized", False):
            return instance
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False  # noqa: SLF001
                cls._instance._instance_lock = threading.RLock()  # noqa: SLF001
            if not cls._instance._initialized:  # noqa: SLF001
                cls._instance._do_initialize()  # noqa: SLF001
            return cls._instance

    def __init__(self) -> None:  # stub-ok: singleton-noop-init-by-design
        pass

    def _do_initialize(self) -> None:
        self.memory_operation_total = Counter(
            name="memory_operation_total",
            label_names=["operation", "status", "handler"],
        )
        self.memory_storage_latency_ms = Histogram(
            name="memory_storage_latency_ms",
            label_names=["operation", "handler"],
        )
        self.memory_retrieval_latency_ms = Histogram(
            name="memory_retrieval_latency_ms",
            label_names=["operation", "handler"],
        )
        self.handler_health_status = Gauge(
            name="handler_health_status",
            label_names=["handler"],
        )
        self._initialized = True

    def get_all_metrics(self) -> dict[str, dict[str, object]]:
        """Get snapshot of all metrics for reporting."""
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
        """Reset the registry by clearing all metrics data (primarily for testing)."""
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
            if cls._instance is not None and cls._instance._initialized:  # noqa: SLF001
                with cls._instance._instance_lock:  # noqa: SLF001
                    cls._instance._clear_all_metrics()  # noqa: SLF001

    @classmethod
    def _reset_instance_for_testing(cls) -> None:
        """Fully reset the singleton instance (TESTING ONLY)."""
        with cls._class_lock:
            if cls._instance is not None and cls._instance._initialized:  # noqa: SLF001
                with cls._instance._instance_lock:  # noqa: SLF001
                    cls._instance._clear_all_metrics()  # noqa: SLF001
            cls._instance = None

    def _clear_all_metrics(self) -> None:
        """Clear all data from metrics atomically."""
        with self.memory_operation_total._lock:  # noqa: SLF001
            self.memory_operation_total._values.clear()  # noqa: SLF001
        with self.memory_storage_latency_ms._lock:  # noqa: SLF001
            self.memory_storage_latency_ms._values.clear()  # noqa: SLF001
        with self.memory_retrieval_latency_ms._lock:  # noqa: SLF001
            self.memory_retrieval_latency_ms._values.clear()  # noqa: SLF001
        with self.handler_health_status._lock:  # noqa: SLF001
            self.handler_health_status._values.clear()  # noqa: SLF001


# Global metrics registry instance
metrics_registry = MetricsRegistry()

# Unused import kept for type checking; not used at runtime
_ = time
