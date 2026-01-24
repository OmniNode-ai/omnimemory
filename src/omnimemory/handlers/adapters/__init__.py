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
    - AdapterIntentGraph: Wraps HandlerGraph for intent classification storage
    - AdapterValkey: Valkey/Redis adapter for subscription caching
    - EmbeddingHttpClient: Wraps HandlerHttp for embedding API calls
    - ProviderRateLimiter: Rate limiting keyed by (provider, model)

Example::

    from omnimemory.handlers.adapters import (
        AdapterGraphMemory,
        ModelGraphMemoryConfig,
        AdapterIntentGraph,
        ModelAdapterIntentGraphConfig,
        AdapterValkey,
        AdapterValkeyConfig,
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

    # Intent graph adapter
    intent_config = ModelAdapterIntentGraphConfig(timeout_seconds=30.0)
    intent_adapter = AdapterIntentGraph(intent_config)
    await intent_adapter.initialize(connection_uri="bolt://localhost:7687")

    # Valkey adapter for caching
    valkey_config = AdapterValkeyConfig(host="localhost", port=6379)
    valkey = AdapterValkey(valkey_config)
    await valkey.initialize()
    await valkey.set_key("key", "value")

.. versionadded:: 0.1.0
    Initial implementation with AdapterGraphMemory (OMN-1389).

.. versionadded:: 0.2.0
    Added EmbeddingHttpClient and ProviderRateLimiter for OMN-1391.
    Added AdapterIntentGraph for OMN-1457.
    Added AdapterValkey for OMN-1393.
"""

from omnimemory.handlers.adapters.adapter_embedding_http import (
    EmbeddingHttpClient,
    EnumEmbeddingProviderType,
    ModelEmbeddingHttpClientConfig,
)
from omnimemory.handlers.adapters.adapter_graph_memory import AdapterGraphMemory
from omnimemory.handlers.adapters.adapter_intent_graph import (
    AdapterIntentGraph,
    IntentCypherTemplates,
)
from omnimemory.handlers.adapters.adapter_rate_limiter import (
    ProviderRateLimiter,
    RateLimiterRegistry,
)
from omnimemory.handlers.adapters.adapter_valkey import (
    AdapterValkey,
    AdapterValkeyConfig,
    ModelValkeyHealth,
)
from omnimemory.handlers.adapters.models import (
    ModelAdapterIntentGraphConfig,
    ModelIntentClassificationOutput,
    ModelIntentDistributionResult,
    ModelIntentGraphHealth,
    ModelIntentQueryResult,
    ModelIntentRecord,
    ModelIntentStorageResult,
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
    # Intent graph adapter
    "AdapterIntentGraph",
    "ModelAdapterIntentGraphConfig",
    "IntentCypherTemplates",
    "ModelConnectionsResult",
    "ModelGraphMemoryConfig",
    "ModelGraphMemoryHealth",
    "ModelIntentClassificationOutput",
    "ModelIntentDistributionResult",
    "ModelIntentGraphHealth",
    "ModelIntentQueryResult",
    "ModelIntentRecord",
    "ModelIntentStorageResult",
    "ModelMemoryConnection",
    "ModelRelatedMemory",
    "ModelRelatedMemoryResult",
    # Valkey Adapter
    "AdapterValkey",
    "AdapterValkeyConfig",
    "ModelValkeyHealth",
    # Embedding HTTP client (contract boundary)
    "EmbeddingHttpClient",
    "ModelEmbeddingHttpClientConfig",
    "EnumEmbeddingProviderType",
    # Rate limiter
    "ProviderRateLimiter",
    "ModelRateLimiterConfig",
    "RateLimiterRegistry",
]
