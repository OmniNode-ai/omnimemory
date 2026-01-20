"""
Memory service configuration model composing all backend configs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnimemory.models.config.model_filesystem_config import ModelFilesystemConfig
from omnimemory.models.config.model_postgres_config import ModelPostgresConfig
from omnimemory.models.config.model_qdrant_config import ModelQdrantConfig


class ModelMemoryServiceConfig(BaseModel):
    """Top-level configuration composing all backend configs.

    This model follows the composable config pattern to avoid the
    "god config object" antipattern. Each backend has its own focused
    configuration model, composed here into a unified service config.

    Phase 1 requires only filesystem. Postgres and Qdrant are optional
    and can be enabled as needed for persistent and vector storage.
    """

    model_config = ConfigDict(extra="forbid")

    # Required backend (Phase 1)
    filesystem: ModelFilesystemConfig = Field(
        description="Filesystem storage configuration (required for Phase 1)",
    )

    # Optional backends
    postgres: ModelPostgresConfig | None = Field(
        default=None,
        description="PostgreSQL storage configuration (optional)",
    )
    qdrant: ModelQdrantConfig | None = Field(
        default=None,
        description="Qdrant vector storage configuration (optional)",
    )

    # Service-level settings
    service_name: str = Field(
        default="omnimemory",
        description="Name of the memory service instance",
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable performance metrics collection",
    )
    enable_logging: bool = Field(
        default=True,
        description="Enable operation logging",
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode for verbose output",
    )
