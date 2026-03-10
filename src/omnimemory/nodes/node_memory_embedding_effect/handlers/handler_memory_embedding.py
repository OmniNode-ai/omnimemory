# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Handler for document-indexed.v1 events — chunks, embeds, and upserts to Qdrant.

Receives ``onex.evt.omnimemory.document-indexed.v1`` events, extracts the
document text, and delegates to ``HandlerQdrant.execute(operation="index")``
for chunking, embedding, and upserting.

Skips events that:
- Do not match the expected topic (``event_type`` mismatch)
- Carry empty or missing ``extracted_text``

TODO: This handler must be registered as a Kafka consumer for
``onex.evt.omnimemory.document-indexed.v1`` in the node runtime. Handler
creation alone is not sufficient for the ingestion pipeline to function
end-to-end. Registration wiring is a separate follow-up task.

Ticket: OMN-4477
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omnimemory.nodes.node_memory_retrieval_effect.handlers.handler_qdrant import (
        HandlerQdrant,
    )

logger = logging.getLogger(__name__)

__all__ = ["HandlerMemoryEmbedding"]

_INDEXED_TOPIC = "onex.evt.omnimemory.document-indexed.v1"


class HandlerMemoryEmbedding:
    """Handles document-indexed.v1 events by indexing content into Qdrant.

    Receives events from the ``onex.evt.omnimemory.document-indexed.v1``
    Kafka topic, extracts the document text, and delegates to
    ``HandlerQdrant.execute(operation="index")`` for chunking, embedding,
    and upserting.

    Attributes:
        _qdrant_handler: Initialized ``HandlerQdrant`` instance for indexing.
    """

    def __init__(self, qdrant_handler: HandlerQdrant) -> None:
        """Initialise with a pre-configured Qdrant handler.

        The caller is responsible for calling ``qdrant_handler.initialize()``
        before passing it here, or for calling this handler's ``initialize()``
        which will delegate to the Qdrant handler.

        Args:
            qdrant_handler: Configured and initializable HandlerQdrant instance.
        """
        self._qdrant_handler = qdrant_handler
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def is_initialized(self) -> bool:
        """Return True if the handler has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the handler and its Qdrant dependency.

        Double-checked locking prevents concurrent initialisation.
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            await self._qdrant_handler.initialize()
            self._initialized = True
            logger.info("HandlerMemoryEmbedding initialized")

    async def shutdown(self) -> None:
        """Shutdown the Qdrant handler and release resources."""
        if self._initialized:
            await self._qdrant_handler.shutdown()
            self._initialized = False
            logger.debug("HandlerMemoryEmbedding shutdown complete")

    async def handle(self, event: Any) -> None:
        """Process a document-indexed event by indexing content into Qdrant.

        Skips events whose ``event_type`` does not match the expected topic
        or whose ``extracted_text`` is empty.

        Args:
            event: Event object with ``event_type: str`` and
                   ``payload: dict`` containing ``document_id: str``
                   and ``extracted_text: str``.
        """
        event_type: str = getattr(event, "event_type", "")
        if event_type != _INDEXED_TOPIC:
            logger.debug(
                "HandlerMemoryEmbedding: skipping event_type=%r (expected %r)",
                event_type,
                _INDEXED_TOPIC,
            )
            return

        payload: dict[str, Any] = getattr(event, "payload", {}) or {}
        document_id: str = payload.get("document_id", "")
        extracted_text: str = payload.get("extracted_text", "")

        if not extracted_text:
            logger.warning(
                "HandlerMemoryEmbedding: skipping document_id=%r — empty extracted_text",
                document_id,
            )
            return

        index_request = SimpleNamespace(
            operation="index",
            document_id=document_id,
            content=extracted_text,
        )
        response = await self._qdrant_handler.execute(index_request)
        logger.info(
            "HandlerMemoryEmbedding: indexed document_id=%r — status=%s, chunks=%d",
            document_id,
            response.status,
            response.total_count or 0,
        )
