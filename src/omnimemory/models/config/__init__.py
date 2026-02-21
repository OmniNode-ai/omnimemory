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

Document Ingestion Configs (OMN-2426):
- ModelDocSourceConfig: Document token budget and similarity thresholds
- ModelPromotionThresholdSet: Per-source-type tier transition thresholds
- ModelScopeMappingConfig: Path/Linear scope_ref resolution with longest-prefix-match
- ModelPathScopeMapping: Single path-prefix to scope_ref entry
- ModelLinearScopeMapping: Single Linear (team, project) to scope_ref entry
"""

from omnimemory.models.config.model_doc_source_config import ModelDocSourceConfig
from omnimemory.models.config.model_embedding_config import (
    EnumEmbeddingProviderType,
    ModelEmbeddingHttpClientConfig,
)
from omnimemory.models.config.model_filesystem_config import ModelFilesystemConfig
from omnimemory.models.config.model_handler_semantic_compute_config import (
    ModelHandlerSemanticComputeConfig,
)
from omnimemory.models.config.model_linear_scope_mapping import (
    ModelLinearScopeMapping,
)
from omnimemory.models.config.model_memory_service_config import (
    ModelMemoryServiceConfig,
)
from omnimemory.models.config.model_path_scope_mapping import ModelPathScopeMapping
from omnimemory.models.config.model_postgres_config import ModelPostgresConfig
from omnimemory.models.config.model_promotion_threshold_set import (
    DEFAULT_PROMOTION_THRESHOLDS,
    PROMOTION_THRESHOLD_LINEAR_TICKET,
    PROMOTION_THRESHOLD_MEMORY_HOOK,
    PROMOTION_THRESHOLD_MEMORY_PATTERN,
    PROMOTION_THRESHOLD_REPO_DERIVED,
    PROMOTION_THRESHOLD_STATIC_STANDARDS,
    ModelPromotionThresholdSet,
)
from omnimemory.models.config.model_qdrant_config import ModelQdrantConfig
from omnimemory.models.config.model_rate_limiter_config import (
    DEFAULT_REQUESTS_PER_MINUTE,
    ModelRateLimiterConfig,
)
from omnimemory.models.config.model_scope_mapping_config import (
    ModelScopeMappingConfig,
)
from omnimemory.models.config.model_semantic_compute_policy_config import (
    ModelSemanticComputePolicyConfig,
)

__all__ = [
    "DEFAULT_PROMOTION_THRESHOLDS",
    "DEFAULT_REQUESTS_PER_MINUTE",
    "EnumEmbeddingProviderType",
    "ModelDocSourceConfig",
    "ModelEmbeddingHttpClientConfig",
    "ModelFilesystemConfig",
    "ModelHandlerSemanticComputeConfig",
    "ModelLinearScopeMapping",
    "ModelMemoryServiceConfig",
    "ModelPathScopeMapping",
    "ModelPostgresConfig",
    "ModelPromotionThresholdSet",
    "ModelQdrantConfig",
    "ModelRateLimiterConfig",
    "ModelScopeMappingConfig",
    "ModelSemanticComputePolicyConfig",
    "PROMOTION_THRESHOLD_LINEAR_TICKET",
    "PROMOTION_THRESHOLD_MEMORY_HOOK",
    "PROMOTION_THRESHOLD_MEMORY_PATTERN",
    "PROMOTION_THRESHOLD_REPO_DERIVED",
    "PROMOTION_THRESHOLD_STATIC_STANDARDS",
]
