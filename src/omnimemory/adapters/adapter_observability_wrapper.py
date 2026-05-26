# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Handler observability wrapper for OmniMemory.

Extracted from observability.py (OMN-11580).

Implements the "one wrapper, one log line, no payload" pattern for P1C observability:
- HandlerMetrics dataclass for captured metrics
- HandlerObservabilityWrapper context manager for handler instrumentation
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import structlog

from .adapter_correlation_context import correlation_id_var
from .adapter_metrics_counter import (
    MetricsRegistry,
    StructuredLogEntry,
    _sanitize_error,
    validate_correlation_id,
)


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

    Instead of logging raw content, logs:
    - Length of content
    - Hash prefix (first 8 chars of SHA-256)
    - Whether content exists
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
                            StructuredLogEntry schema before emission.
                            Default is False. RECOMMENDED for development and testing.

    Example usage:
        ```python
        wrapper = HandlerObservabilityWrapper(handler_name="filesystem")

        async def store_memory(request: MemoryStoreRequest) -> MemoryStoreResponse:
            async with wrapper.observe_operation(
                operation="store",
                correlation_id=str(request.correlation_id),
            ) as ctx:
                result = await do_storage(request)
                return result
        ```
    """

    _HANDLER_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

    def __init__(
        self,
        handler_name: str,
        registry: MetricsRegistry | None = None,
        validate_log_schema: bool = False,
    ) -> None:
        """Initialize the wrapper.

        Args:
            handler_name: Name of the handler. Must contain only alphanumeric
                         characters, underscores, and hyphens (max 64 characters).
            registry: Optional metrics registry (defaults to current singleton).
            validate_log_schema: If True, validate log entries before emission.

        Raises:
            ValueError: If handler_name is empty or doesn't match the required pattern.
        """
        if not handler_name:
            raise ValueError("handler_name must be a non-empty string")

        if not self._HANDLER_NAME_PATTERN.match(handler_name):
            raise ValueError(
                f"handler_name must contain only alphanumeric characters, "
                f"underscores, and hyphens (max 64 characters), got: {handler_name!r}"
            )

        self.handler_name = handler_name
        self._custom_registry: MetricsRegistry | None = registry
        self._validate_log_schema = validate_log_schema
        self._logger = structlog.get_logger(f"omnimemory.handler.{handler_name}")

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
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        if not validate_correlation_id(correlation_id):
            correlation_id = str(uuid.uuid4())

        token = correlation_id_var.set(correlation_id)

        ctx = {
            "correlation_id": correlation_id,
            "operation": operation,
            "handler": self.handler_name,
        }

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
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            self._record_metrics(
                operation=operation,
                status=status,
                latency_ms=latency_ms,
            )

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

            correlation_id_var.reset(token)

    def _record_metrics(
        self,
        operation: str,
        status: Literal["success", "failure"],
        latency_ms: float,
    ) -> None:
        """Record all metrics for the operation."""
        self.registry.memory_operation_total.inc(
            operation=operation,
            status=status,
            handler=self.handler_name,
        )

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
            self.registry.memory_storage_latency_ms.observe(
                latency_ms,
                operation=operation,
                handler=self.handler_name,
            )

        if status == "success":
            self.registry.handler_health_status.set(1.0, handler=self.handler_name)
        else:
            self.registry.handler_health_status.set(0.0, handler=self.handler_name)

    def _emit_log_event(self, metrics: HandlerMetrics) -> None:
        """Emit a single structured log event for the operation."""
        timestamp = (
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )

        log_data: dict[str, str | float] = {
            "correlation_id": metrics.correlation_id,
            "operation": metrics.operation,
            "handler": metrics.handler,
            "status": metrics.status,
            "latency_ms": round(metrics.latency_ms, 2),
            "timestamp": timestamp,
        }

        if metrics.status == "failure":
            if metrics.error_type is not None:
                log_data["error_type"] = metrics.error_type
            if metrics.error_message is not None:
                log_data["error_message"] = metrics.error_message

        if self._validate_log_schema:
            try:
                StructuredLogEntry.model_validate(log_data)
            except Exception as validation_error:
                self._logger.warning(
                    "omnimemory.handler.log_schema_validation_failed",
                    error=str(validation_error),
                    handler=self.handler_name,
                    operation=metrics.operation,
                )

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
        """Get statistics for this handler."""
        counter_metric = self.registry.memory_operation_total
        all_counters = counter_metric.get_all()
        handler_counters = {
            k: v
            for k, v in all_counters.items()
            if counter_metric.labels_from_key(k).get("handler") == self.handler_name
        }

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

        health = self.registry.handler_health_status.get(handler=self.handler_name)

        return {
            "handler": self.handler_name,
            "health_status": health,
            "operation_counts": handler_counters,
            "storage_latency": handler_storage,
            "retrieval_latency": handler_retrieval,
        }
