# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory Handler Adapters.

This package contains adapter layers that wrap omnibase_infra handlers
to provide memory-specific interfaces. Adapters translate between
memory domain concepts and underlying infrastructure operations.

IMPORTANT: These adapters are the ONLY allowed exit hatch for external HTTP
calls in OmniMemory. Direct use of httpx/requests is forbidden in business logic.

Available Adapters:
    - AdapterGraphMemory: Wraps HandlerGraph for relationship-based memory queries
    - EmbeddingHttpClient: Wraps HandlerHttp for embedding API calls
    - ProviderRateLimiter: Rate limiting keyed by (provider, model)

Example::

    from omnimemory.handlers.adapters import (
        AdapterGraphMemory,
        ModelGraphMemoryConfig,
        EmbeddingHttpClient,
        ModelEmbeddingHttpClientConfig,
    )

    # Graph adapter
    config = ModelGraphMemoryConfig(max_depth=3)
    adapter = AdapterGraphMemory(config)
    await adapter.initialize(connection_uri="bolt://localhost:7687")

    # Embedding client (contract boundary for HTTP)
    embed_config = ModelEmbeddingHttpClientConfig(
        provider="local",
        base_url="http://192.168.86.201:8002",
    )
    async with EmbeddingHttpClient(embed_config) as client:
        embedding = await client.get_embedding("Hello world")

.. versionadded:: 0.1.0
    Initial implementation for OMN-1389.

.. versionadded:: 0.2.0
    Added EmbeddingHttpClient and ProviderRateLimiter for OMN-1391.
"""

from omnimemory.handlers.adapters.adapter_embedding_http import (
    EmbeddingHttpClient,
    EnumEmbeddingProviderType,
    ModelEmbeddingHttpClientConfig,
)
from omnimemory.handlers.adapters.adapter_graph_memory import AdapterGraphMemory
from omnimemory.handlers.adapters.adapter_rate_limiter import (
    ProviderRateLimiter,
    RateLimiterRegistry,
)
from omnimemory.models.adapters import (
    ModelConnectionsResult,
    ModelGraphMemoryConfig,
    ModelGraphMemoryHealth,
    ModelMemoryConnection,
    ModelRelatedMemory,
    ModelRelatedMemoryResult,
)
from omnimemory.models.config import ModelRateLimiterConfig

__all__ = [
    # Graph memory adapter
    "AdapterGraphMemory",
    "ModelConnectionsResult",
    "ModelGraphMemoryConfig",
    "ModelGraphMemoryHealth",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
    # Embedding HTTP client (contract boundary)
    "EmbeddingHttpClient",
    "ModelEmbeddingHttpClientConfig",
    "EnumEmbeddingProviderType",
    # Rate limiter
    "ProviderRateLimiter",
    "ModelRateLimiterConfig",
    "RateLimiterRegistry",
]
