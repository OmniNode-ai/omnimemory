"""
Configuration models for OmniMemory storage backends and policies.

This module provides composable configuration models for each storage backend,
following the ONEX pattern of focused, single-responsibility config objects.

Backend Configs:
- ModelFilesystemConfig: Local filesystem storage (required for Phase 1)
- ModelPostgresConfig: PostgreSQL persistent storage (optional)
- ModelQdrantConfig: Qdrant vector storage (optional)
- ModelEmbeddingHttpClientConfig: Embedding HTTP client (optional)

Service Config:
- ModelMemoryServiceConfig: Top-level config composing all backends

Policy Configs:
- ModelSemanticComputePolicyConfig: Semantic analysis policy configuration
- ModelRateLimiterConfig: Rate limiting configuration
"""

from omnimemory.models.config.model_embedding_config import (
    EnumEmbeddingProviderType,
    ModelEmbeddingHttpClientConfig,
)
from omnimemory.models.config.model_filesystem_config import ModelFilesystemConfig
from omnimemory.models.config.model_handler_semantic_compute_config import (
    ModelHandlerSemanticComputeConfig,
)
from omnimemory.models.config.model_memory_service_config import (
    ModelMemoryServiceConfig,
)
from omnimemory.models.config.model_postgres_config import ModelPostgresConfig
from omnimemory.models.config.model_qdrant_config import ModelQdrantConfig
from omnimemory.models.config.model_rate_limiter_config import (
    DEFAULT_REQUESTS_PER_MINUTE,
    ModelRateLimiterConfig,
)
from omnimemory.models.config.model_semantic_compute_policy_config import (
    ModelSemanticComputePolicyConfig,
)

__all__ = [
    "DEFAULT_REQUESTS_PER_MINUTE",
    "EnumEmbeddingProviderType",
    "ModelEmbeddingHttpClientConfig",
    "ModelFilesystemConfig",
    "ModelHandlerSemanticComputeConfig",
    "ModelMemoryServiceConfig",
    "ModelPostgresConfig",
    "ModelQdrantConfig",
    "ModelRateLimiterConfig",
    "ModelSemanticComputePolicyConfig",
]
