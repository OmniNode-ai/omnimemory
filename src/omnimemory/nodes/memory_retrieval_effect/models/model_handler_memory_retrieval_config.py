# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Configuration model for the memory retrieval handler.

This module contains the Pydantic configuration model for
HandlerMemoryRetrieval.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1387.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .model_handler_db_mock_config import ModelHandlerDbMockConfig
from .model_handler_graph_mock_config import ModelHandlerGraphMockConfig
from .model_handler_qdrant_mock_config import ModelHandlerQdrantMockConfig

__all__ = [
    "ModelHandlerMemoryRetrievalConfig",
]


class ModelHandlerMemoryRetrievalConfig(BaseModel):
    """Configuration for the memory retrieval handler.

    Attributes:
        qdrant_config: Configuration for the Qdrant handler.
        db_config: Configuration for the Database handler.
        graph_config: Configuration for the Graph handler.
        use_mock_handlers: Whether to use mock handlers. When False, real
            handlers from omnibase_infra will be used (not yet implemented).
    """

    model_config = ConfigDict(frozen=True)

    qdrant_config: ModelHandlerQdrantMockConfig = Field(
        default_factory=ModelHandlerQdrantMockConfig,
        description="Configuration for Qdrant semantic search handler",
    )
    db_config: ModelHandlerDbMockConfig = Field(
        default_factory=ModelHandlerDbMockConfig,
        description="Configuration for Database full-text search handler",
    )
    graph_config: ModelHandlerGraphMockConfig = Field(
        default_factory=ModelHandlerGraphMockConfig,
        description="Configuration for Graph traversal handler",
    )
    use_mock_handlers: bool = Field(
        default=True,
        description="Use mock handlers (True) or real handlers (False)",
    )
