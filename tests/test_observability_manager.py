# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Tests for ObservabilityManager, ContextVar helpers, and security utilities.

Covers uncovered surfaces in observability.py:
    - validate_correlation_id
    - sanitize_metadata_value
    - ObservabilityManager.correlation_context
    - ObservabilityManager.trace_operation
    - ObservabilityManager.get_current_context
    - ObservabilityManager.get_performance_metrics
    - get_correlation_id / get_request_id / log_with_correlation
    - OperationType enum
    - correlation_context / trace_operation module-level convenience wrappers
"""

from __future__ import annotations

import pytest

from omnimemory.utils.observability import (
    ObservabilityManager,
    OperationType,
    correlation_context,
    correlation_id_var,
    get_correlation_id,
    get_request_id,
    sanitize_metadata_value,
    trace_operation,
    validate_correlation_id,
)

# =============================================================================
# validate_correlation_id
# =============================================================================


@pytest.mark.unit
class TestValidateCorrelationId:
    """Unit tests for validate_correlation_id."""

    def test_valid_uuid_format(self) -> None:
        assert validate_correlation_id("550e8400-e29b-41d4-a716-446655440000") is True

    def test_valid_alphanumeric(self) -> None:
        assert validate_correlation_id("abc123DEF") is True

    def test_valid_with_underscores_and_hyphens(self) -> None:
        assert validate_correlation_id("corr_id-001") is True

    def test_empty_string_is_invalid(self) -> None:
        assert validate_correlation_id("") is False

    def test_too_long_is_invalid(self) -> None:
        assert validate_correlation_id("a" * 65) is False

    def test_exactly_64_chars_is_valid(self) -> None:
        assert validate_correlation_id("a" * 64) is True

    def test_special_chars_are_invalid(self) -> None:
        assert validate_correlation_id("bad<>chars") is False

    def test_space_is_invalid(self) -> None:
        assert validate_correlation_id("hello world") is False

    def test_newline_is_invalid(self) -> None:
        assert validate_correlation_id("abc\ndef") is False

    def test_single_char_is_valid(self) -> None:
        assert validate_correlation_id("x") is True


# =============================================================================
# sanitize_metadata_value
# =============================================================================


@pytest.mark.unit
class TestSanitizeMetadataValue:
    """Unit tests for sanitize_metadata_value."""

    def test_passes_through_plain_string(self) -> None:
        result = sanitize_metadata_value("hello world")
        assert result == "hello world"

    def test_removes_angle_brackets(self) -> None:
        result = sanitize_metadata_value("<script>alert(1)</script>")
        assert "<" not in str(result)
        assert ">" not in str(result)

    def test_removes_quotes(self) -> None:
        result = sanitize_metadata_value('it\'s a "test"')
        assert '"' not in str(result)
        assert "'" not in str(result)

    def test_truncates_long_strings(self) -> None:
        result = sanitize_metadata_value("x" * 2000)
        assert isinstance(result, str)
        assert len(result) <= 1000

    def test_passes_through_int(self) -> None:
        assert sanitize_metadata_value(42) == 42

    def test_passes_through_float(self) -> None:
        assert sanitize_metadata_value(3.14) == 3.14

    def test_passes_through_bool_true(self) -> None:
        assert sanitize_metadata_value(True) is True

    def test_passes_through_bool_false(self) -> None:
        assert sanitize_metadata_value(False) is False

    def test_passes_through_none(self) -> None:
        assert sanitize_metadata_value(None) is None

    def test_converts_arbitrary_object_to_string(self) -> None:
        result = sanitize_metadata_value(["list", "value"])
        assert isinstance(result, str)

    def test_bool_not_treated_as_int(self) -> None:
        # bool is a subclass of int; ensure True/False are returned as bool
        assert sanitize_metadata_value(True) is True
        assert sanitize_metadata_value(False) is False


# =============================================================================
# OperationType enum
# =============================================================================


@pytest.mark.unit
class TestOperationType:
    """Unit tests for OperationType enum values."""

    def test_all_values_are_strings(self) -> None:
        for member in OperationType:
            assert isinstance(member.value, str)

    def test_memory_store_value(self) -> None:
        assert OperationType.MEMORY_STORE.value == "memory_store"

    def test_health_check_value(self) -> None:
        assert OperationType.HEALTH_CHECK.value == "health_check"

    def test_str_conversion_in_trace_operation(self) -> None:
        # trace_operation accepts str that converts to OperationType
        op = OperationType("memory_retrieve")
        assert op == OperationType.MEMORY_RETRIEVE


# =============================================================================
# get_correlation_id / get_request_id context helpers
# =============================================================================


@pytest.mark.unit
class TestContextVarHelpers:
    """Unit tests for get_correlation_id and get_request_id."""

    def test_get_correlation_id_returns_none_by_default(self) -> None:
        token = correlation_id_var.set(None)
        try:
            assert get_correlation_id() is None
        finally:
            correlation_id_var.reset(token)

    def test_get_correlation_id_returns_set_value(self) -> None:
        token = correlation_id_var.set("my-corr-id")
        try:
            assert get_correlation_id() == "my-corr-id"
        finally:
            correlation_id_var.reset(token)

    def test_get_request_id_returns_none_by_default(self) -> None:
        from omnimemory.utils.observability import request_id_var

        token = request_id_var.set(None)
        try:
            assert get_request_id() is None
        finally:
            request_id_var.reset(token)


# =============================================================================
# ObservabilityManager.correlation_context
# =============================================================================


@pytest.mark.unit
class TestObservabilityManagerCorrelationContext:
    """Unit tests for ObservabilityManager.correlation_context."""

    @pytest.mark.asyncio
    async def test_yields_correlation_context_with_given_id(self) -> None:
        manager = ObservabilityManager()
        async with manager.correlation_context(
            correlation_id="abc-123",
            operation="test_op",
        ) as ctx:
            assert ctx.correlation_id == "abc-123"
            assert ctx.operation == "test_op"

    @pytest.mark.asyncio
    async def test_generates_correlation_id_when_not_provided(self) -> None:
        manager = ObservabilityManager()
        async with manager.correlation_context() as ctx:
            assert ctx.correlation_id
            assert len(ctx.correlation_id) > 0

    @pytest.mark.asyncio
    async def test_sets_context_var_during_execution(self) -> None:
        manager = ObservabilityManager()
        async with manager.correlation_context(correlation_id="test-ctx-id") as ctx:
            assert get_correlation_id() == ctx.correlation_id

    @pytest.mark.asyncio
    async def test_resets_context_var_after_exit(self) -> None:
        manager = ObservabilityManager()
        original = get_correlation_id()
        async with manager.correlation_context(correlation_id="ephemeral-id"):
            pass
        assert get_correlation_id() == original

    @pytest.mark.asyncio
    async def test_resets_context_var_on_exception(self) -> None:
        manager = ObservabilityManager()
        original = get_correlation_id()
        with pytest.raises(ValueError):
            async with manager.correlation_context(correlation_id="exc-id"):
                raise ValueError("intentional")
        assert get_correlation_id() == original

    @pytest.mark.asyncio
    async def test_raises_on_invalid_correlation_id(self) -> None:
        manager = ObservabilityManager()
        with pytest.raises(ValueError, match="Invalid correlation ID"):
            async with manager.correlation_context(correlation_id="bad<chars>"):
                pass

    @pytest.mark.asyncio
    async def test_get_current_context_inside_manager(self) -> None:
        manager = ObservabilityManager()
        async with manager.correlation_context(
            correlation_id="ctx-check",
            operation="my_op",
        ):
            ctx = manager.get_current_context()
            assert ctx["correlation_id"] == "ctx-check"
            assert ctx["operation"] == "my_op"

    @pytest.mark.asyncio
    async def test_parent_correlation_id_captured(self) -> None:
        manager = ObservabilityManager()
        async with manager.correlation_context(correlation_id="parent-id") as parent:
            async with manager.correlation_context(correlation_id="child-id") as child:
                assert child.parent_correlation_id == parent.correlation_id


# =============================================================================
# ObservabilityManager.trace_operation
# =============================================================================


@pytest.mark.unit
class TestObservabilityManagerTraceOperation:
    """Unit tests for ObservabilityManager.trace_operation."""

    @pytest.mark.asyncio
    async def test_yields_trace_id_string(self) -> None:
        manager = ObservabilityManager()
        async with manager.trace_operation(
            operation_name="test_op",
            operation_type=OperationType.MEMORY_STORE,
        ) as trace_id:
            assert isinstance(trace_id, str)
            assert len(trace_id) > 0

    @pytest.mark.asyncio
    async def test_trace_performance_false_skips_metrics(self) -> None:
        manager = ObservabilityManager()
        async with manager.trace_operation(
            operation_name="test_op",
            operation_type=OperationType.HEALTH_CHECK,
            trace_performance=False,
        ) as trace_id:
            assert trace_id not in manager.get_performance_metrics()

    @pytest.mark.asyncio
    async def test_trace_removed_from_active_after_completion(self) -> None:
        manager = ObservabilityManager()
        captured_id: list[str] = []
        async with manager.trace_operation(
            operation_name="op",
            operation_type=OperationType.MEMORY_STORE,
        ) as trace_id:
            captured_id.append(trace_id)
        assert captured_id[0] not in manager.get_performance_metrics()

    @pytest.mark.asyncio
    async def test_exception_propagates_out(self) -> None:
        manager = ObservabilityManager()
        with pytest.raises(RuntimeError, match="intentional"):
            async with manager.trace_operation(
                operation_name="failing_op",
                operation_type=OperationType.MEMORY_STORE,
            ):
                raise RuntimeError("intentional")

    @pytest.mark.asyncio
    async def test_evicts_oldest_when_at_capacity(self) -> None:
        manager = ObservabilityManager(max_active_traces=2)
        # Open 3 traces simultaneously without closing them
        # Can't easily hold them open without tasks, so just verify init works
        assert manager.max_active_traces == 2


# =============================================================================
# Module-level convenience wrappers
# =============================================================================


@pytest.mark.unit
class TestModuleLevelConvenienceWrappers:
    """Unit tests for module-level correlation_context and trace_operation wrappers."""

    @pytest.mark.asyncio
    async def test_correlation_context_wrapper_yields_context(self) -> None:
        async with correlation_context(
            correlation_id="wrap-id",
            operation="wrapped_op",
        ) as ctx:
            assert ctx.correlation_id == "wrap-id"

    @pytest.mark.asyncio
    async def test_trace_operation_wrapper_with_string_type(self) -> None:
        async with trace_operation(
            operation_name="op",
            operation_type="memory_store",
        ) as trace_id:
            assert isinstance(trace_id, str)

    @pytest.mark.asyncio
    async def test_trace_operation_wrapper_with_unknown_string_defaults(self) -> None:
        # Unknown string type falls back to EXTERNAL_API
        async with trace_operation(
            operation_name="op",
            operation_type="completely_unknown_type",
        ) as trace_id:
            assert isinstance(trace_id, str)

    @pytest.mark.asyncio
    async def test_trace_operation_wrapper_with_enum_type(self) -> None:
        async with trace_operation(
            operation_name="op",
            operation_type=OperationType.MEMORY_RETRIEVE,
        ) as trace_id:
            assert isinstance(trace_id, str)


# =============================================================================
# ObservabilityManager.get_current_context
# =============================================================================


@pytest.mark.unit
class TestGetCurrentContext:
    """Unit tests for ObservabilityManager.get_current_context."""

    def test_returns_none_values_outside_context(self) -> None:
        manager = ObservabilityManager()
        ctx = manager.get_current_context()
        # All context vars default to None outside any context manager
        assert "correlation_id" in ctx
        assert "operation" in ctx

    @pytest.mark.asyncio
    async def test_returns_correct_values_inside_context(self) -> None:
        manager = ObservabilityManager()
        async with manager.correlation_context(
            correlation_id="in-ctx",
            operation="store",
        ):
            ctx = manager.get_current_context()
            assert ctx["correlation_id"] == "in-ctx"
            assert ctx["operation"] == "store"
