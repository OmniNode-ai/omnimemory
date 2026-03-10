# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerMemoryEmbedding (OMN-4477).

Tests:
    1. test_handle_document_indexed_calls_index_operation
    2. test_handle_skips_non_indexed_events
    3. test_handle_skips_event_with_no_text

Ticket: OMN-4477
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnimemory.nodes.node_memory_embedding_effect.handlers.handler_memory_embedding import (
    HandlerMemoryEmbedding,
)


def _make_qdrant_mock() -> AsyncMock:
    """Return a mock HandlerQdrant with an async execute method."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.shutdown = AsyncMock()
    response = MagicMock()
    response.status = "success"
    response.total_count = 1
    mock.execute = AsyncMock(return_value=response)
    return mock


@pytest.mark.unit
class TestHandlerMemoryEmbedding:
    """Tests for HandlerMemoryEmbedding."""

    @pytest.mark.asyncio
    async def test_handle_document_indexed_calls_index_operation(self) -> None:
        """Valid document-indexed event triggers HandlerQdrant.execute with operation='index'."""
        qdrant = _make_qdrant_mock()
        handler = HandlerMemoryEmbedding(qdrant_handler=qdrant)
        handler._initialized = True

        event = SimpleNamespace(
            event_type="onex.evt.omnimemory.document-indexed.v1",
            payload={"document_id": "doc-abc", "extracted_text": "Hello ONEX world."},
        )
        await handler.handle(event)

        qdrant.execute.assert_awaited_once()
        call_request = qdrant.execute.call_args[0][0]
        assert call_request.operation == "index"
        assert call_request.document_id == "doc-abc"
        assert call_request.content == "Hello ONEX world."

    @pytest.mark.asyncio
    async def test_handle_skips_non_indexed_events(self) -> None:
        """Events with mismatching event_type must be skipped without calling execute."""
        qdrant = _make_qdrant_mock()
        handler = HandlerMemoryEmbedding(qdrant_handler=qdrant)
        handler._initialized = True

        event = SimpleNamespace(
            event_type="some.other.event.v1",
            payload={"document_id": "doc-abc", "extracted_text": "Hello."},
        )
        await handler.handle(event)

        qdrant.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_handle_skips_event_with_no_text(self) -> None:
        """Events with empty extracted_text must be skipped without calling execute."""
        qdrant = _make_qdrant_mock()
        handler = HandlerMemoryEmbedding(qdrant_handler=qdrant)
        handler._initialized = True

        event = SimpleNamespace(
            event_type="onex.evt.omnimemory.document-indexed.v1",
            payload={"document_id": "doc-abc", "extracted_text": ""},
        )
        await handler.handle(event)

        qdrant.execute.assert_not_awaited()
