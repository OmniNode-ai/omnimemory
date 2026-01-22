"""
Health check configuration model for OmniMemory ONEX architecture.

This module contains the ModelHealthCheckConfig class and DependencyType enum.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ..foundation.model_health_metadata import HealthCheckMetadata

if TYPE_CHECKING:
    from .model_circuit_breaker_config import ModelCircuitBreakerConfig

__all__ = [
    "DependencyType",
    "ModelHealthCheckConfig",
]


class DependencyType(Enum):
    """Types of system dependencies."""

    DATABASE = "database"
    CACHE = "cache"
    VECTOR_DB = "vector_db"
    EXTERNAL_API = "external_api"
    MESSAGE_QUEUE = "message_queue"
    STORAGE = "storage"


class ModelHealthCheckConfig(BaseModel):
    """Configuration for individual health checks."""

    name: str = Field(description="Dependency name")
    dependency_type: DependencyType = Field(description="Type of dependency")
    timeout: float = Field(default=5.0, description="Health check timeout in seconds")
    critical: bool = Field(
        default=True, description="Whether failure affects overall health"
    )
    circuit_breaker_config: ModelCircuitBreakerConfig | None = Field(default=None)
    metadata: HealthCheckMetadata = Field(default_factory=HealthCheckMetadata)
