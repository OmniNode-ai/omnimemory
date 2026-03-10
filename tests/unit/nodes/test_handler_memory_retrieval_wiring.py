# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit test for HandlerMemoryRetrieval production wiring (OMN-4476).

Verifies that use_stub_handlers=False wires HandlerQdrant (not a mock)
for the Qdrant semantic search path.

Ticket: OMN-4476
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_memory_retrieval import (
    HandlerMemoryRetrieval,
)
from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant import (
    HandlerQdrant,
)
from omnimemory.nodes.node_memory_retrieval_effect.models import (
    ModelHandlerMemoryRetrievalConfig,
    ModelHandlerQdrantConfig,
)


@pytest.mark.unit
class TestHandlerMemoryRetrievalWiring:
    """Tests for HandlerMemoryRetrieval production-mode handler wiring."""

    @pytest.mark.asyncio
    async def test_production_mode_wires_handler_qdrant(self) -> None:
        """use_stub_handlers=False must wire HandlerQdrant for the Qdrant path."""
        cfg = ModelHandlerMemoryRetrievalConfig(
            use_stub_handlers=False,
            qdrant_config=ModelHandlerQdrantConfig(
                embedding_server_url="http://localhost:8100"
            ),
        )
        with patch.object(HandlerQdrant, "initialize", new_callable=AsyncMock):
            handler = HandlerMemoryRetrieval(cfg)
            await handler.initialize()

        assert handler._initialized is True
        assert isinstance(handler._qdrant_handler, HandlerQdrant), (
            "_qdrant_handler must be HandlerQdrant in production mode, not a mock"
        )

    @pytest.mark.asyncio
    async def test_stub_mode_wires_mock_handlers(self) -> None:
        """use_stub_handlers=True must wire mock handlers."""
        from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant_mock import (
            HandlerQdrantMock,
        )

        cfg = ModelHandlerMemoryRetrievalConfig(use_stub_handlers=True)
        handler = HandlerMemoryRetrieval(cfg)
        await handler.initialize()

        assert handler._initialized is True
        assert isinstance(handler._qdrant_handler, HandlerQdrantMock), (
            "_qdrant_handler must be HandlerQdrantMock in stub mode"
        )
