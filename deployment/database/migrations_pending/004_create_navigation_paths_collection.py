#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Create the ``navigation_paths`` Qdrant collection for OMN-2584.

Run once against the target Qdrant instance to create the collection used by
``NavigationHistoryWriter``.  Safe to run repeatedly — exits with success if
the collection already exists.

Usage::

    # Against localhost (default):
    python 004_create_navigation_paths_collection.py

    # Against a remote Qdrant:
    QDRANT_HOST=192.168.86.200 QDRANT_PORT=6333 \\
        python 004_create_navigation_paths_collection.py

Requires: qdrant-client
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = "navigation_paths"

# Qwen3-Embedding-8B produces 4096-dimensional vectors.
EMBEDDING_DIM = 4096


async def create_collection() -> None:
    """Create the ``navigation_paths`` Qdrant collection.

    Uses named vectors so the retrieval engine can query either the goal vector
    or the start-state vector independently.

    Named vectors:
    - ``goal``: Embedding of the goal condition text.
    - ``start_state``: Embedding of the start-state identifier.
    """
    client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)

    try:
        try:
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "goal": qdrant_models.VectorParams(
                        size=EMBEDDING_DIM,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                    "start_state": qdrant_models.VectorParams(
                        size=EMBEDDING_DIM,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                },
                # Payload indexes for filtered search
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=1000,
                ),
            )
            logger.info("Created collection '%s' with named vectors.", COLLECTION_NAME)
        except UnexpectedResponse as exc:
            # 409 Conflict means the collection was created concurrently — treat as success.
            if exc.status_code == 409:
                logger.info(
                    "Collection '%s' already exists (created concurrently) — no action required.",
                    COLLECTION_NAME,
                )
                return
            logger.error("Qdrant API error during collection creation: %s", exc)
            sys.exit(1)

        # Create payload indexes for common filter patterns
        try:
            await client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="outcome",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="graph_fingerprint",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="start_state_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            logger.info("Created payload indexes on outcome, graph_fingerprint, start_state_id.")
        except UnexpectedResponse as exc:
            logger.error("Qdrant API error during index creation: %s", exc)
            sys.exit(1)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(create_collection())
