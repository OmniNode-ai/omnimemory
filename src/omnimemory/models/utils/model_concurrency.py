# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Concurrency-related Pydantic models for OmniMemory ONEX architecture.

This module contains models for connection pool configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ModelConnectionPoolConfig",
]


class ModelConnectionPoolConfig(BaseModel):
    """Configuration for connection pools."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(description="Pool name")
    min_connections: int = Field(default=1, ge=0, description="Minimum connections")
    max_connections: int = Field(
        default=50,
        ge=1,
        description="Maximum connections (increased for production load)",
    )
    connection_timeout: float = Field(
        default=30.0, gt=0, description="Connection timeout"
    )
    idle_timeout: float = Field(
        default=300.0, gt=0, description="Idle connection timeout"
    )
    health_check_interval: float = Field(
        default=60.0, gt=0, description="Health check interval"
    )
    retry_attempts: int = Field(
        default=3, ge=0, description="Retry attempts for failed connections"
    )

    @model_validator(mode="after")
    def validate_connection_bounds(self) -> ModelConnectionPoolConfig:
        """Validate that max_connections >= min_connections."""
        if self.max_connections < self.min_connections:
            raise ValueError(
                f"max_connections ({self.max_connections}) must be >= "
                f"min_connections ({self.min_connections})"
            )
        return self
