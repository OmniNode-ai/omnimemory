# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Validate that contract-driven discovery finds all expected handlers.

This test prevents silent regressions when contracts are added or removed.
It ensures the contract-driven wiring produces the exact same handler set
as the old hardcoded _HANDLER_SPECS list.

Related:
    - OMN-7153: Add handler completeness validation test
"""

from __future__ import annotations

import pytest

from omnimemory.runtime.wiring import wire_memory_handlers

from .conftest import StubConfig

# The canonical handler set — must match what the old _HANDLER_SPECS declared.
_EXPECTED_HANDLERS: frozenset[str] = frozenset(
    {
        "HandlerIntentEventConsumer",
        "HandlerIntentQuery",
        "HandlerSubscription",
        "HandlerSimilarityCompute",
        "AdapterIntentGraph",
        "HandlerNavigationHistoryReducer",
        "HandlerSemanticCompute",
        "HandlerMemoryLifecycle",
    }
)


@pytest.mark.unit
class TestContractHandlerCompleteness:
    @pytest.mark.asyncio
    async def test_all_expected_handlers_discovered(self) -> None:
        """Contract-driven discovery must find all expected handlers."""
        config = StubConfig()
        result = await wire_memory_handlers(config=config)  # type: ignore[arg-type]
        discovered = frozenset(result)
        assert discovered == _EXPECTED_HANDLERS, (
            f"Handler mismatch.\n"
            f"  Missing: {_EXPECTED_HANDLERS - discovered}\n"
            f"  Extra: {discovered - _EXPECTED_HANDLERS}"
        )
