# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory Storage Effect - ONEX Node (Core 8 Foundation).

CRUD operations to storage backends. P1A implements FileSystem backend,
with PostgreSQL, Redis, and Pinecone deferred to Phase 2.

This is a fully declarative ONEX node:
- Node behavior defined in contract.yaml
- Business logic implemented in adapters/
- No node.py class needed

Node Type: EFFECT
"""

from .adapters import HandlerFileSystemAdapter, ModelFileSystemAdapterConfig
from .models import ModelMemoryStorageRequest, ModelMemoryStorageResponse

__all__ = [
    # Models
    "ModelMemoryStorageRequest",
    "ModelMemoryStorageResponse",
    "ModelFileSystemAdapterConfig",
    # Handlers
    "HandlerFileSystemAdapter",
]
