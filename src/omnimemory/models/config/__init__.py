"""
Configuration models for OmniMemory storage backends.

This module provides composable configuration models for each storage backend,
following the ONEX pattern of focused, single-responsibility config objects.

Backend Configs:
- ModelFilesystemConfig: Local filesystem storage (required for Phase 1)
- ModelPostgresConfig: PostgreSQL persistent storage (optional)
- ModelQdrantConfig: Qdrant vector storage (optional)

Service Config:
- ModelMemoryServiceConfig: Top-level config composing all backends
"""

from omnimemory.models.config.model_filesystem_config import ModelFilesystemConfig
from omnimemory.models.config.model_memory_service_config import (
    ModelMemoryServiceConfig,
)
from omnimemory.models.config.model_postgres_config import ModelPostgresConfig
from omnimemory.models.config.model_qdrant_config import ModelQdrantConfig

__all__ = [
    "ModelFilesystemConfig",
    "ModelMemoryServiceConfig",
    "ModelPostgresConfig",
    "ModelQdrantConfig",
]
