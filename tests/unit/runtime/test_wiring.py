# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for memory domain handler wiring.

Validates:
    - wire_memory_handlers() successfully imports all handler specs
    - Returns list of registered handler names
    - Handles import errors gracefully

Related:
    - OMN-2216: Phase 5 -- Runtime plugin PluginMemory
"""

from __future__ import annotations

import pytest

from omnimemory.runtime.wiring import wire_memory_handlers

from .conftest import StubConfig

# =============================================================================
# Tests
# =============================================================================


class TestWireMemoryHandlers:
    """Validate wire_memory_handlers imports and verifies handlers."""

    @pytest.mark.asyncio
    async def test_returns_handler_names(self) -> None:
        """wire_memory_handlers should return list of handler names."""
        config = StubConfig()

        result = await wire_memory_handlers(config=config)  # type: ignore[arg-type]

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_excludes_intent_consumer(self) -> None:
        """HandlerIntentEventConsumer must NOT be wired by omnimemory.

        node_intent_event_consumer_effect is canonically owned by omnimarket
        (OMN-13701). Its presence in omnimemory caused a duplicate UUID and
        dual-consumer race on the intent-classified Kafka topic.
        """
        config = StubConfig()

        result = await wire_memory_handlers(config=config)  # type: ignore[arg-type]

        assert "HandlerIntentEventConsumer" not in result

    @pytest.mark.asyncio
    async def test_includes_intent_query(self) -> None:
        """HandlerIntentQuery should be in the registered services."""
        config = StubConfig()

        result = await wire_memory_handlers(config=config)  # type: ignore[arg-type]

        assert "HandlerIntentQuery" in result

    @pytest.mark.asyncio
    async def test_includes_subscription(self) -> None:
        """HandlerSubscription should be in the registered services."""
        config = StubConfig()

        result = await wire_memory_handlers(config=config)  # type: ignore[arg-type]

        assert "HandlerSubscription" in result

    @pytest.mark.asyncio
    async def test_all_handlers_callable(self) -> None:
        """All registered handlers should be callable classes/functions."""
        config = StubConfig()

        # If any handler is not callable, wire_memory_handlers raises ImportError
        result = await wire_memory_handlers(config=config)  # type: ignore[arg-type]

        # Success means all handlers passed the callable check
        # 7 handlers (reduced from 8 in OMN-13701: HandlerIntentEventConsumer moved to omnimarket)
        assert len(result) == 7


class TestContractDrivenDiscovery:
    """Validate that handler discovery reads from contracts, not hardcoded list."""

    @pytest.mark.asyncio
    async def test_discovers_handlers_from_contracts(self) -> None:
        """wire_memory_handlers returns 7 handlers from contracts.

        Reduced from 8 (OMN-13701): HandlerIntentEventConsumer removed;
        node_intent_event_consumer_effect canonical owner is omnimarket.
        """
        config = StubConfig()
        result = await wire_memory_handlers(config=config)  # type: ignore[arg-type]
        assert len(result) == 7

    @pytest.mark.asyncio
    async def test_no_hardcoded_handler_specs(self) -> None:
        """_HANDLER_SPECS must not exist -- all specs come from contracts."""
        import omnimemory.runtime.wiring as wiring_mod

        assert not hasattr(wiring_mod, "_HANDLER_SPECS"), (
            "_HANDLER_SPECS still exists -- migration incomplete"
        )
