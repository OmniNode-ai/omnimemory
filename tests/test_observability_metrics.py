# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for P1C observability hooks.

Tests for in-process metrics (Counter, Histogram, Gauge), MetricsRegistry,
and HandlerObservabilityWrapper.
"""

import asyncio
import threading
import uuid

import pytest
from pydantic import ValidationError

from omnimemory.utils.observability import (
    Counter,
    Gauge,
    HandlerMetrics,
    HandlerObservabilityWrapper,
    Histogram,
    LabelValidationError,
    MetricsRegistry,
    StructuredLogEntry,
    _get_safe_content_metadata,
    create_validated_log_entry,
    validate_log_entry,
    validate_metric_labels,
)


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_initialization(self) -> None:
        """Test counter initializes with correct name and labels."""
        counter = Counter("test_counter", ["label1", "label2"])
        assert counter.name == "test_counter"
        assert counter.label_names == ["label1", "label2"]

    def test_counter_increment_default(self) -> None:
        """Test counter increments by 1 by default."""
        counter = Counter("test_counter", ["operation"])
        counter.inc(operation="store")
        assert counter.get(operation="store") == 1

    def test_counter_increment_custom_amount(self) -> None:
        """Test counter increments by custom amount."""
        counter = Counter("test_counter", ["operation"])
        counter.inc(5, operation="store")
        assert counter.get(operation="store") == 5

    def test_counter_multiple_labels(self) -> None:
        """Test counter tracks different label combinations separately."""
        counter = Counter("test_counter", ["operation", "status"])
        counter.inc(operation="store", status="success")
        counter.inc(operation="store", status="failure")
        counter.inc(operation="retrieve", status="success")

        assert counter.get(operation="store", status="success") == 1
        assert counter.get(operation="store", status="failure") == 1
        assert counter.get(operation="retrieve", status="success") == 1
        assert counter.get(operation="retrieve", status="failure") == 0

    def test_counter_get_all(self) -> None:
        """Test get_all returns all counter values."""
        counter = Counter("test_counter", ["operation"])
        counter.inc(operation="store")
        counter.inc(operation="retrieve")
        counter.inc(operation="store")

        all_values = counter.get_all()
        assert len(all_values) == 2
        assert all_values[("store",)] == 2
        assert all_values[("retrieve",)] == 1

    def test_counter_labels_from_key(self) -> None:
        """Test converting key tuple back to labels dict."""
        counter = Counter("test_counter", ["operation", "status"])
        key = ("store", "success")
        labels = counter.labels_from_key(key)
        assert labels == {"operation": "store", "status": "success"}

    def test_counter_thread_safety(self) -> None:
        """Test counter is thread-safe."""
        counter = Counter("test_counter", ["operation"])
        num_threads = 10
        increments_per_thread = 100

        def increment_counter() -> None:
            for _ in range(increments_per_thread):
                counter.inc(operation="concurrent")

        threads = [
            threading.Thread(target=increment_counter) for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert (
            counter.get(operation="concurrent") == num_threads * increments_per_thread
        )


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_initialization(self) -> None:
        """Test histogram initializes with correct name and labels."""
        hist = Histogram("test_hist", ["operation"])
        assert hist.name == "test_hist"
        assert hist.label_names == ["operation"]
        assert len(hist.buckets) > 0  # Default buckets

    def test_histogram_custom_buckets(self) -> None:
        """Test histogram with custom buckets."""
        custom_buckets = (1.0, 5.0, 10.0)
        hist = Histogram("test_hist", ["operation"], buckets=custom_buckets)
        assert hist.buckets == custom_buckets

    def test_histogram_observe(self) -> None:
        """Test histogram observation."""
        hist = Histogram("test_hist", ["operation"], buckets=(10.0, 50.0, 100.0))
        hist.observe(25.0, operation="store")
        hist.observe(75.0, operation="store")
        hist.observe(5.0, operation="store")

        snapshot = hist.get(operation="store")
        assert snapshot["count"] == 3
        assert snapshot["sum"] == 105.0
        # Bucket counts: [1 for <=10, 2 for <=50, 3 for <=100, 3 for +Inf]
        assert snapshot["buckets"] == [1, 2, 3, 3]

    def test_histogram_multiple_labels(self) -> None:
        """Test histogram tracks different label combinations separately."""
        hist = Histogram("test_hist", ["operation", "handler"], buckets=(10.0, 50.0))
        hist.observe(25.0, operation="store", handler="fs")
        hist.observe(5.0, operation="retrieve", handler="fs")

        store_snapshot = hist.get(operation="store", handler="fs")
        retrieve_snapshot = hist.get(operation="retrieve", handler="fs")

        assert store_snapshot["count"] == 1
        assert retrieve_snapshot["count"] == 1

    def test_histogram_get_empty(self) -> None:
        """Test get returns empty snapshot for non-existent labels."""
        hist = Histogram("test_hist", ["operation"])
        snapshot = hist.get(operation="nonexistent")
        assert snapshot["count"] == 0
        assert snapshot["sum"] == 0.0

    def test_histogram_get_all(self) -> None:
        """Test get_all returns all histogram values."""
        hist = Histogram("test_hist", ["operation"], buckets=(10.0,))
        hist.observe(5.0, operation="store")
        hist.observe(15.0, operation="retrieve")

        all_values = hist.get_all()
        assert len(all_values) == 2


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_initialization(self) -> None:
        """Test gauge initializes with correct name and labels."""
        gauge = Gauge("test_gauge", ["handler"])
        assert gauge.name == "test_gauge"
        assert gauge.label_names == ["handler"]

    def test_gauge_set_and_get(self) -> None:
        """Test gauge set and get operations."""
        gauge = Gauge("test_gauge", ["handler"])
        gauge.set(1.0, handler="filesystem")
        assert gauge.get(handler="filesystem") == 1.0

        gauge.set(0.0, handler="filesystem")
        assert gauge.get(handler="filesystem") == 0.0

    def test_gauge_multiple_labels(self) -> None:
        """Test gauge tracks different handlers separately."""
        gauge = Gauge("test_gauge", ["handler"])
        gauge.set(1.0, handler="filesystem")
        gauge.set(0.0, handler="postgresql")

        assert gauge.get(handler="filesystem") == 1.0
        assert gauge.get(handler="postgresql") == 0.0

    def test_gauge_get_default(self) -> None:
        """Test get returns 0.0 for non-existent labels."""
        gauge = Gauge("test_gauge", ["handler"])
        assert gauge.get(handler="nonexistent") == 0.0

    def test_gauge_get_all(self) -> None:
        """Test get_all returns all gauge values."""
        gauge = Gauge("test_gauge", ["handler"])
        gauge.set(1.0, handler="fs")
        gauge.set(0.5, handler="redis")

        all_values = gauge.get_all()
        assert len(all_values) == 2
        assert all_values[("fs",)] == 1.0
        assert all_values[("redis",)] == 0.5


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        MetricsRegistry.reset()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        MetricsRegistry.reset()

    def test_registry_singleton(self) -> None:
        """Test registry is a singleton."""
        registry1 = MetricsRegistry()
        registry2 = MetricsRegistry()
        assert registry1 is registry2

    def test_registry_has_required_metrics(self) -> None:
        """Test registry has all required P1C metrics."""
        registry = MetricsRegistry()

        assert hasattr(registry, "memory_operation_total")
        assert hasattr(registry, "memory_storage_latency_ms")
        assert hasattr(registry, "memory_retrieval_latency_ms")
        assert hasattr(registry, "handler_health_status")

    def test_registry_metrics_work(self) -> None:
        """Test registry metrics can be used."""
        registry = MetricsRegistry()

        registry.memory_operation_total.inc(
            operation="store", status="success", handler="fs"
        )
        assert (
            registry.memory_operation_total.get(
                operation="store", status="success", handler="fs"
            )
            == 1
        )

        registry.memory_storage_latency_ms.observe(
            45.0, operation="store", handler="fs"
        )
        snapshot = registry.memory_storage_latency_ms.get(
            operation="store", handler="fs"
        )
        assert snapshot["count"] == 1

        registry.handler_health_status.set(1.0, handler="fs")
        assert registry.handler_health_status.get(handler="fs") == 1.0

    def test_registry_get_all_metrics(self) -> None:
        """Test get_all_metrics returns all metrics."""
        registry = MetricsRegistry()
        registry.memory_operation_total.inc(
            operation="store", status="success", handler="fs"
        )

        all_metrics = registry.get_all_metrics()

        assert "memory_operation_total" in all_metrics
        assert "memory_storage_latency_ms" in all_metrics
        assert "memory_retrieval_latency_ms" in all_metrics
        assert "handler_health_status" in all_metrics


class TestHandlerObservabilityWrapper:
    """Tests for HandlerObservabilityWrapper."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        MetricsRegistry.reset()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        MetricsRegistry.reset()

    @pytest.mark.asyncio
    async def test_wrapper_successful_operation(self) -> None:
        """Test wrapper records metrics for successful operation."""
        wrapper = HandlerObservabilityWrapper(handler_name="test_handler")

        async with wrapper.observe_operation(
            operation="store",
            correlation_id="test-correlation-id",
        ) as ctx:
            assert ctx["operation"] == "store"
            assert ctx["handler"] == "test_handler"
            assert ctx["correlation_id"] == "test-correlation-id"
            await asyncio.sleep(0.01)  # Simulate some work

        # Check counter was incremented
        registry = MetricsRegistry()
        count = registry.memory_operation_total.get(
            operation="store", status="success", handler="test_handler"
        )
        assert count == 1

        # Check histogram was recorded
        snapshot = registry.memory_storage_latency_ms.get(
            operation="store", handler="test_handler"
        )
        assert snapshot["count"] == 1
        assert snapshot["sum"] >= 10.0  # At least 10ms

        # Check health gauge is set to healthy
        health = registry.handler_health_status.get(handler="test_handler")
        assert health == 1.0

    @pytest.mark.asyncio
    async def test_wrapper_failed_operation(self) -> None:
        """Test wrapper records metrics for failed operation."""
        wrapper = HandlerObservabilityWrapper(handler_name="test_handler")

        with pytest.raises(ValueError):
            async with wrapper.observe_operation(
                operation="store",
                correlation_id="test-correlation-id",
            ):
                raise ValueError("Test error")

        # Check counter was incremented with failure status
        registry = MetricsRegistry()
        count = registry.memory_operation_total.get(
            operation="store", status="failure", handler="test_handler"
        )
        assert count == 1

        # Check health gauge is set to unhealthy
        health = registry.handler_health_status.get(handler="test_handler")
        assert health == 0.0

    @pytest.mark.asyncio
    async def test_wrapper_generates_correlation_id(self) -> None:
        """Test wrapper generates correlation ID if not provided."""
        wrapper = HandlerObservabilityWrapper(handler_name="test_handler")

        async with wrapper.observe_operation(operation="store") as ctx:
            assert ctx["correlation_id"] is not None
            # Should be a valid UUID
            uuid.UUID(ctx["correlation_id"])

    @pytest.mark.asyncio
    async def test_wrapper_retrieval_uses_correct_histogram(self) -> None:
        """Test retrieval operations use retrieval histogram."""
        wrapper = HandlerObservabilityWrapper(handler_name="test_handler")

        async with wrapper.observe_operation(operation="retrieve"):
            await asyncio.sleep(0.01)

        registry = MetricsRegistry()

        # Should NOT be in storage histogram
        storage_snapshot = registry.memory_storage_latency_ms.get(
            operation="retrieve", handler="test_handler"
        )
        assert storage_snapshot["count"] == 0

        # Should be in retrieval histogram
        retrieval_snapshot = registry.memory_retrieval_latency_ms.get(
            operation="retrieve", handler="test_handler"
        )
        assert retrieval_snapshot["count"] == 1

    @pytest.mark.asyncio
    async def test_wrapper_mark_healthy_unhealthy(self) -> None:
        """Test manual health status changes."""
        wrapper = HandlerObservabilityWrapper(handler_name="test_handler")

        # Initial state is healthy
        registry = MetricsRegistry()
        assert registry.handler_health_status.get(handler="test_handler") == 1.0

        wrapper.mark_unhealthy()
        assert registry.handler_health_status.get(handler="test_handler") == 0.0

        wrapper.mark_healthy()
        assert registry.handler_health_status.get(handler="test_handler") == 1.0

    @pytest.mark.asyncio
    async def test_wrapper_get_handler_stats(self) -> None:
        """Test get_handler_stats returns correct statistics."""
        wrapper = HandlerObservabilityWrapper(handler_name="test_handler")

        async with wrapper.observe_operation(operation="store"):
            pass

        async with wrapper.observe_operation(operation="retrieve"):
            pass

        stats = wrapper.get_handler_stats()

        assert stats["handler"] == "test_handler"
        assert stats["health_status"] == 1.0
        assert "operation_counts" in stats
        assert "storage_latency" in stats
        assert "retrieval_latency" in stats

    @pytest.mark.asyncio
    async def test_wrapper_invalid_correlation_id_replaced(self) -> None:
        """Test invalid correlation ID is replaced with valid UUID."""
        wrapper = HandlerObservabilityWrapper(handler_name="test_handler")

        # Invalid correlation ID with special characters
        async with wrapper.observe_operation(
            operation="store",
            correlation_id="invalid<script>id",
        ) as ctx:
            # Should be replaced with a valid UUID
            assert ctx["correlation_id"] != "invalid<script>id"
            uuid.UUID(ctx["correlation_id"])  # Should not raise


class TestSafeContentMetadata:
    """Tests for _get_safe_content_metadata helper."""

    def test_none_content(self) -> None:
        """Test handling of None content."""
        result = _get_safe_content_metadata(None)
        assert result["content_exists"] is False
        assert result["content_len"] == 0

    def test_valid_content(self) -> None:
        """Test extraction of safe metadata from content."""
        content = "This is test content with potentially sensitive data"
        result = _get_safe_content_metadata(content)

        assert result["content_exists"] is True
        assert result["content_len"] == len(content)
        assert "content_hash" in result
        assert len(result["content_hash"]) == 8  # First 8 chars of SHA-256

    def test_custom_field_name(self) -> None:
        """Test custom field name prefix."""
        result = _get_safe_content_metadata("test", field_name="body")
        assert "body_exists" in result
        assert "body_len" in result
        assert "body_hash" in result

    def test_deterministic_hash(self) -> None:
        """Test hash is deterministic for same content."""
        content = "same content"
        result1 = _get_safe_content_metadata(content)
        result2 = _get_safe_content_metadata(content)
        assert result1["content_hash"] == result2["content_hash"]

    def test_different_hash_for_different_content(self) -> None:
        """Test different content produces different hash."""
        result1 = _get_safe_content_metadata("content A")
        result2 = _get_safe_content_metadata("content B")
        assert result1["content_hash"] != result2["content_hash"]


class TestHandlerMetricsDataclass:
    """Tests for HandlerMetrics dataclass."""

    def test_handler_metrics_creation(self) -> None:
        """Test HandlerMetrics can be created with required fields."""
        metrics = HandlerMetrics(
            correlation_id="test-id",
            operation="store",
            handler="filesystem",
            status="success",
            latency_ms=45.5,
        )

        assert metrics.correlation_id == "test-id"
        assert metrics.operation == "store"
        assert metrics.handler == "filesystem"
        assert metrics.status == "success"
        assert metrics.latency_ms == 45.5
        assert metrics.error_type is None
        assert metrics.error_message is None

    def test_handler_metrics_with_error(self) -> None:
        """Test HandlerMetrics with error information."""
        metrics = HandlerMetrics(
            correlation_id="test-id",
            operation="store",
            handler="filesystem",
            status="failure",
            latency_ms=12.3,
            error_type="ValueError",
            error_message="Invalid input",
        )

        assert metrics.status == "failure"
        assert metrics.error_type == "ValueError"
        assert metrics.error_message == "Invalid input"


class TestStrictLabelsValidation:
    """Tests for strict_labels parameter on Counter, Histogram, and Gauge."""

    def test_counter_strict_labels_missing(self) -> None:
        """Test Counter raises ValueError when labels are missing in strict mode."""
        counter = Counter("test_counter", ["operation", "status"], strict_labels=True)
        with pytest.raises(ValueError) as exc_info:
            counter.inc(operation="store")  # Missing "status"
        assert "missing labels" in str(exc_info.value)
        assert "status" in str(exc_info.value)

    def test_counter_strict_labels_extra(self) -> None:
        """Test Counter raises ValueError when extra labels provided in strict mode."""
        counter = Counter("test_counter", ["operation"], strict_labels=True)
        with pytest.raises(ValueError) as exc_info:
            counter.inc(operation="store", extra="bad")  # Extra "extra" label
        assert "extra labels" in str(exc_info.value)
        assert "extra" in str(exc_info.value)

    def test_counter_strict_labels_valid(self) -> None:
        """Test Counter accepts valid labels in strict mode."""
        counter = Counter("test_counter", ["operation", "status"], strict_labels=True)
        counter.inc(operation="store", status="success")  # Should not raise
        assert counter.get(operation="store", status="success") == 1

    def test_histogram_strict_labels_missing(self) -> None:
        """Test Histogram raises ValueError when labels are missing in strict mode."""
        hist = Histogram("test_hist", ["operation", "handler"], strict_labels=True)
        with pytest.raises(ValueError) as exc_info:
            hist.observe(10.0, operation="store")  # Missing "handler"
        assert "missing labels" in str(exc_info.value)
        assert "handler" in str(exc_info.value)

    def test_histogram_strict_labels_extra(self) -> None:
        """Test Histogram raises ValueError when extra labels provided."""
        hist = Histogram("test_hist", ["operation"], strict_labels=True)
        with pytest.raises(ValueError) as exc_info:
            hist.observe(10.0, operation="store", extra="bad")
        assert "extra labels" in str(exc_info.value)

    def test_histogram_strict_labels_valid(self) -> None:
        """Test Histogram accepts valid labels in strict mode."""
        hist = Histogram("test_hist", ["operation", "handler"], strict_labels=True)
        hist.observe(10.0, operation="store", handler="fs")  # Should not raise
        snapshot = hist.get(operation="store", handler="fs")
        assert snapshot["count"] == 1

    def test_gauge_strict_labels_missing(self) -> None:
        """Test Gauge raises ValueError when labels are missing in strict mode."""
        gauge = Gauge("test_gauge", ["handler", "region"], strict_labels=True)
        with pytest.raises(ValueError) as exc_info:
            gauge.set(1.0, handler="fs")  # Missing "region"
        assert "missing labels" in str(exc_info.value)
        assert "region" in str(exc_info.value)

    def test_gauge_strict_labels_extra(self) -> None:
        """Test Gauge raises ValueError when extra labels provided in strict mode."""
        gauge = Gauge("test_gauge", ["handler"], strict_labels=True)
        with pytest.raises(ValueError) as exc_info:
            gauge.set(1.0, handler="fs", extra="bad")
        assert "extra labels" in str(exc_info.value)

    def test_gauge_strict_labels_valid(self) -> None:
        """Test Gauge accepts valid labels in strict mode."""
        gauge = Gauge("test_gauge", ["handler", "region"], strict_labels=True)
        gauge.set(1.0, handler="fs", region="us-east")  # Should not raise
        assert gauge.get(handler="fs", region="us-east") == 1.0

    def test_lenient_mode_ignores_missing(self) -> None:
        """Test lenient mode (default) doesn't raise on missing labels."""
        counter = Counter("test_counter", ["operation", "status"], strict_labels=False)
        counter.inc(operation="store")  # Missing "status" - should NOT raise
        # Missing status defaults to empty string
        assert counter.get(operation="store", status="") == 1

    def test_lenient_mode_ignores_extra(self) -> None:
        """Test lenient mode (default) doesn't raise on extra labels."""
        counter = Counter("test_counter", ["operation"], strict_labels=False)
        counter.inc(
            operation="store", extra="ignored"
        )  # Extra label - should NOT raise
        assert counter.get(operation="store") == 1


class TestGetHandlerStatsFiltering:
    """Tests for get_handler_stats() handler filtering to prevent false matches."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        MetricsRegistry.reset()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        MetricsRegistry.reset()

    @pytest.mark.asyncio
    async def test_handler_stats_no_cross_contamination(self) -> None:
        """Test get_handler_stats() only returns metrics for the correct handler.

        This verifies that label-based filtering prevents false matches where
        a handler name might appear in other label positions.
        """
        # Create two wrappers with different handler names
        wrapper_fs = HandlerObservabilityWrapper(handler_name="filesystem")
        wrapper_pg = HandlerObservabilityWrapper(handler_name="postgresql")

        # Record operations for both handlers
        async with wrapper_fs.observe_operation(operation="store"):
            pass
        async with wrapper_fs.observe_operation(operation="retrieve"):
            pass
        async with wrapper_pg.observe_operation(operation="store"):
            pass

        # Get stats for filesystem handler
        fs_stats = wrapper_fs.get_handler_stats()

        # Verify filesystem stats only contain filesystem operations
        # Should have 2 operations (store + retrieve success)
        assert fs_stats["handler"] == "filesystem"
        operation_counts = fs_stats["operation_counts"]
        for key, count in operation_counts.items():
            # Extract handler label from the key using labels_from_key
            # The counter has label_names=["operation", "status", "handler"]
            assert len(key) == 3, "Counter key should have 3 elements"
            assert key[2] == "filesystem", (
                f"Expected handler='filesystem', got key={key}"
            )

        # Get stats for postgresql handler
        pg_stats = wrapper_pg.get_handler_stats()
        pg_operation_counts = pg_stats["operation_counts"]
        for key, count in pg_operation_counts.items():
            assert key[2] == "postgresql", (
                f"Expected handler='postgresql', got key={key}"
            )

    @pytest.mark.asyncio
    async def test_handler_stats_histogram_filtering(self) -> None:
        """Test histogram filtering in get_handler_stats() is correct."""
        wrapper_a = HandlerObservabilityWrapper(handler_name="handler_a")
        wrapper_b = HandlerObservabilityWrapper(handler_name="handler_b")

        # Record latency for both handlers
        async with wrapper_a.observe_operation(operation="store"):
            pass
        async with wrapper_b.observe_operation(operation="store"):
            pass

        stats_a = wrapper_a.get_handler_stats()
        stats_b = wrapper_b.get_handler_stats()

        # Histogram label_names are ["operation", "handler"]
        for key in stats_a["storage_latency"].keys():
            assert len(key) == 2, "Histogram key should have 2 elements"
            assert key[1] == "handler_a", f"Expected handler='handler_a', got key={key}"

        for key in stats_b["storage_latency"].keys():
            assert key[1] == "handler_b", f"Expected handler='handler_b', got key={key}"

    @pytest.mark.asyncio
    async def test_handler_stats_with_similar_names(self) -> None:
        """Test filtering works correctly with similar handler names."""
        # Names that could potentially cause false matches if filtering is broken
        wrapper_store = HandlerObservabilityWrapper(
            handler_name="store"
        )  # Same as operation name
        wrapper_success = HandlerObservabilityWrapper(
            handler_name="success"
        )  # Same as status value

        async with wrapper_store.observe_operation(operation="store"):
            pass
        async with wrapper_success.observe_operation(operation="store"):
            pass

        # Get stats - should not have cross-contamination
        store_stats = wrapper_store.get_handler_stats()
        success_stats = wrapper_success.get_handler_stats()

        # Each should only have their own operations
        assert len(store_stats["operation_counts"]) == 1
        assert len(success_stats["operation_counts"]) == 1

        # Verify correct handler in keys
        for key in store_stats["operation_counts"].keys():
            assert key[2] == "store"
        for key in success_stats["operation_counts"].keys():
            assert key[2] == "success"


class TestLabelValidationError:
    """Tests for LabelValidationError exception."""

    def test_error_with_missing_labels(self) -> None:
        """Test error message includes missing labels."""
        error = LabelValidationError(
            metric_name="test_metric",
            missing_labels={"status", "handler"},
            extra_labels=set(),
            expected_labels={"operation", "status", "handler"},
            provided_labels={"operation"},
        )
        assert "test_metric" in str(error)
        assert "missing required labels" in str(error)
        assert "handler" in str(error)
        assert "status" in str(error)

    def test_error_with_extra_labels(self) -> None:
        """Test error message includes extra labels."""
        error = LabelValidationError(
            metric_name="test_metric",
            missing_labels=set(),
            extra_labels={"unknown", "extra"},
            expected_labels={"operation"},
            provided_labels={"operation", "unknown", "extra"},
        )
        assert "unexpected extra labels" in str(error)
        assert "unknown" in str(error)
        assert "extra" in str(error)

    def test_error_with_both_missing_and_extra(self) -> None:
        """Test error message includes both missing and extra labels."""
        error = LabelValidationError(
            metric_name="test_metric",
            missing_labels={"status"},
            extra_labels={"extra"},
            expected_labels={"operation", "status"},
            provided_labels={"operation", "extra"},
        )
        assert "missing required labels" in str(error)
        assert "unexpected extra labels" in str(error)

    def test_error_attributes(self) -> None:
        """Test error attributes are accessible."""
        error = LabelValidationError(
            metric_name="test_metric",
            missing_labels={"a"},
            extra_labels={"b"},
            expected_labels={"a", "c"},
            provided_labels={"b", "c"},
        )
        assert error.metric_name == "test_metric"
        assert error.missing_labels == {"a"}
        assert error.extra_labels == {"b"}
        assert error.expected_labels == {"a", "c"}
        assert error.provided_labels == {"b", "c"}


class TestValidateMetricLabels:
    """Tests for validate_metric_labels() function."""

    def test_valid_labels_exact_match(self) -> None:
        """Test validation passes with exact label match."""
        validate_metric_labels(
            labels={"operation": "store", "status": "success"},
            required_labels={"operation", "status"},
            metric_name="test_counter",
        )  # Should not raise

    def test_missing_required_labels_strict(self) -> None:
        """Test strict mode raises on missing required labels."""
        with pytest.raises(LabelValidationError) as exc_info:
            validate_metric_labels(
                labels={"operation": "store"},
                required_labels={"operation", "status"},
                metric_name="test_counter",
                strict=True,
            )
        assert exc_info.value.missing_labels == {"status"}

    def test_extra_labels_strict(self) -> None:
        """Test strict mode raises on extra labels."""
        with pytest.raises(LabelValidationError) as exc_info:
            validate_metric_labels(
                labels={"operation": "store", "unknown": "value"},
                required_labels={"operation"},
                metric_name="test_counter",
                strict=True,
            )
        assert exc_info.value.extra_labels == {"unknown"}

    def test_allowed_labels_superset(self) -> None:
        """Test allowed_labels can be a superset of required_labels."""
        validate_metric_labels(
            labels={"operation": "store", "region": "us-east"},
            required_labels={"operation"},
            allowed_labels={"operation", "region", "zone"},
            metric_name="test_counter",
        )  # Should not raise - region is allowed

    def test_allowed_labels_rejects_unknown(self) -> None:
        """Test extra labels not in allowed set are rejected."""
        with pytest.raises(LabelValidationError) as exc_info:
            validate_metric_labels(
                labels={"operation": "store", "unknown": "value"},
                required_labels={"operation"},
                allowed_labels={"operation", "region"},
                metric_name="test_counter",
            )
        assert "unknown" in exc_info.value.extra_labels

    def test_non_strict_mode_logs_but_doesnt_raise(self) -> None:
        """Test non-strict mode doesn't raise on issues."""
        # Should not raise even with missing and extra labels
        validate_metric_labels(
            labels={"unknown": "value"},
            required_labels={"operation"},
            metric_name="test_counter",
            strict=False,
        )

    def test_empty_required_labels_raises(self) -> None:
        """Test empty required_labels raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_metric_labels(
                labels={"operation": "store"},
                required_labels=set(),
                metric_name="test_counter",
            )
        assert "required_labels must not be empty" in str(exc_info.value)


class TestStructuredLogEntry:
    """Tests for StructuredLogEntry Pydantic model."""

    def test_valid_success_entry(self) -> None:
        """Test valid success log entry."""
        entry = StructuredLogEntry(
            correlation_id="abc123-def456",
            operation="store",
            handler="filesystem",
            status="success",
            latency_ms=45.23,
            timestamp="2025-01-19T12:34:56.789Z",
        )
        assert entry.correlation_id == "abc123-def456"
        assert entry.operation == "store"
        assert entry.handler == "filesystem"
        assert entry.status == "success"
        assert entry.latency_ms == 45.23
        assert entry.error_type is None
        assert entry.error_message is None

    def test_valid_failure_entry(self) -> None:
        """Test valid failure log entry with error fields."""
        entry = StructuredLogEntry(
            correlation_id="abc123",
            operation="store",
            handler="filesystem",
            status="failure",
            latency_ms=102.5,
            timestamp="2025-01-19T12:34:56.789Z",
            error_type="IOError",
            error_message="Permission denied",
        )
        assert entry.status == "failure"
        assert entry.error_type == "IOError"
        assert entry.error_message == "Permission denied"

    def test_invalid_correlation_id_pattern(self) -> None:
        """Test correlation_id pattern validation."""
        with pytest.raises(ValidationError):
            StructuredLogEntry(
                correlation_id="invalid<script>id",  # Invalid characters
                operation="store",
                handler="filesystem",
                status="success",
                latency_ms=45.0,
                timestamp="2025-01-19T12:34:56.789Z",
            )

    def test_invalid_status_value(self) -> None:
        """Test status must be 'success' or 'failure'."""
        with pytest.raises(ValidationError):
            StructuredLogEntry(
                correlation_id="abc123",
                operation="store",
                handler="filesystem",
                status="unknown",  # Invalid
                latency_ms=45.0,
                timestamp="2025-01-19T12:34:56.789Z",
            )

    def test_invalid_timestamp_format(self) -> None:
        """Test timestamp must match ISO8601 pattern."""
        with pytest.raises(ValidationError):
            StructuredLogEntry(
                correlation_id="abc123",
                operation="store",
                handler="filesystem",
                status="success",
                latency_ms=45.0,
                timestamp="2025-01-19 12:34:56",  # Wrong format
            )

    def test_negative_latency_rejected(self) -> None:
        """Test negative latency_ms is rejected."""
        with pytest.raises(ValidationError):
            StructuredLogEntry(
                correlation_id="abc123",
                operation="store",
                handler="filesystem",
                status="success",
                latency_ms=-10.0,
                timestamp="2025-01-19T12:34:56.789Z",
            )

    def test_extra_fields_rejected(self) -> None:
        """Test extra fields are rejected (extra='forbid')."""
        with pytest.raises(ValidationError):
            StructuredLogEntry(
                correlation_id="abc123",
                operation="store",
                handler="filesystem",
                status="success",
                latency_ms=45.0,
                timestamp="2025-01-19T12:34:56.789Z",
                unknown_field="value",  # Should be rejected
            )


class TestValidateLogEntry:
    """Tests for validate_log_entry() function."""

    def test_valid_entry_returns_model(self) -> None:
        """Test valid entry returns StructuredLogEntry."""
        log_data = {
            "correlation_id": "abc123",
            "operation": "store",
            "handler": "filesystem",
            "status": "success",
            "latency_ms": 45.23,
            "timestamp": "2025-01-19T12:34:56.789Z",
        }
        entry = validate_log_entry(log_data)
        assert isinstance(entry, StructuredLogEntry)
        assert entry.correlation_id == "abc123"

    def test_invalid_entry_raises_by_default(self) -> None:
        """Test invalid entry raises ValidationError by default."""
        log_data = {
            "correlation_id": "abc123",
            # Missing required fields
        }
        with pytest.raises(ValidationError):
            validate_log_entry(log_data)

    def test_invalid_entry_returns_none_when_not_raising(self) -> None:
        """Test invalid entry returns None when raise_on_error=False."""
        log_data = {"correlation_id": "abc123"}  # Missing fields
        result = validate_log_entry(log_data, raise_on_error=False)
        assert result is None


class TestCreateValidatedLogEntry:
    """Tests for create_validated_log_entry() function."""

    def test_creates_success_entry(self) -> None:
        """Test creating a success log entry."""
        entry = create_validated_log_entry(
            correlation_id="abc123",
            operation="store",
            handler="filesystem",
            status="success",
            latency_ms=45.23456,
        )
        assert entry.correlation_id == "abc123"
        assert entry.operation == "store"
        assert entry.handler == "filesystem"
        assert entry.status == "success"
        assert entry.latency_ms == 45.23  # Rounded to 2 decimal places
        assert entry.error_type is None
        assert entry.error_message is None

    def test_creates_failure_entry(self) -> None:
        """Test creating a failure log entry with error fields."""
        entry = create_validated_log_entry(
            correlation_id="abc123",
            operation="store",
            handler="filesystem",
            status="failure",
            latency_ms=102.5,
            error_type="IOError",
            error_message="Permission denied",
        )
        assert entry.status == "failure"
        assert entry.error_type == "IOError"
        assert entry.error_message == "Permission denied"

    def test_auto_generates_timestamp(self) -> None:
        """Test timestamp is auto-generated in correct format."""
        entry = create_validated_log_entry(
            correlation_id="abc123",
            operation="store",
            handler="filesystem",
            status="success",
            latency_ms=45.0,
        )
        # Timestamp should match ISO8601 pattern
        assert entry.timestamp.endswith("Z")
        assert "T" in entry.timestamp
        assert len(entry.timestamp) == 24  # YYYY-MM-DDTHH:MM:SS.sssZ

    def test_invalid_fields_raise(self) -> None:
        """Test invalid fields raise ValidationError."""
        with pytest.raises(ValidationError):
            create_validated_log_entry(
                correlation_id="invalid<>id",  # Invalid pattern
                operation="store",
                handler="filesystem",
                status="success",
                latency_ms=45.0,
            )


class TestHandlerObservabilityWrapperLogSchemaValidation:
    """Tests for HandlerObservabilityWrapper log schema validation."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        MetricsRegistry.reset()

    def teardown_method(self) -> None:
        """Reset registry after each test."""
        MetricsRegistry.reset()

    @pytest.mark.asyncio
    async def test_wrapper_with_schema_validation_enabled(self) -> None:
        """Test wrapper works with validate_log_schema=True."""
        wrapper = HandlerObservabilityWrapper(
            handler_name="test_handler",
            validate_log_schema=True,
        )

        # Should work without issues
        async with wrapper.observe_operation(
            operation="store",
            correlation_id="test-correlation-123",
        ):
            pass

        # Verify metrics were recorded
        registry = MetricsRegistry()
        count = registry.memory_operation_total.get(
            operation="store", status="success", handler="test_handler"
        )
        assert count == 1

    @pytest.mark.asyncio
    async def test_wrapper_schema_validation_on_failure(self) -> None:
        """Test schema validation works for failure cases."""
        wrapper = HandlerObservabilityWrapper(
            handler_name="test_handler",
            validate_log_schema=True,
        )

        with pytest.raises(ValueError):
            async with wrapper.observe_operation(operation="store"):
                raise ValueError("Test error")

        # Should still record metrics even with validation
        registry = MetricsRegistry()
        count = registry.memory_operation_total.get(
            operation="store", status="failure", handler="test_handler"
        )
        assert count == 1
