# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Qdrant vector storage configuration model following ONEX standards.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    field_validator,
    model_validator,
)

# Qdrant collection name maximum length
COLLECTION_NAME_MAX_LENGTH = 255


class ModelQdrantConfig(BaseModel):
    """Configuration for Qdrant vector storage.

    This config defines connection parameters for Qdrant-based
    vector memory storage. Optional for Phase 1.
    """

    model_config = ConfigDict(extra="forbid")

    # Connection configuration
    url: HttpUrl = Field(
        default=HttpUrl("http://localhost:6333"),
        description="Qdrant server URL",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant API key for authentication (optional, stored securely)",
        exclude=True,
    )

    # Collection configuration
    collection_name: str = Field(
        default="omnimemory",
        description="Default collection name for memory vectors",
    )
    vector_size: int = Field(
        default=1024,
        ge=1,
        le=65536,
        description="Vector embedding dimensions (1024 for MLX Qwen3)",
    )

    # Performance settings
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds",
    )
    grpc_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="gRPC port for high-performance operations (optional)",
    )
    prefer_grpc: bool = Field(
        default=False,
        description="Prefer gRPC over HTTP for operations",
    )

    # Search settings
    default_limit: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Default number of results to return",
    )
    score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold for results",
    )

    # Index settings
    distance_metric: str = Field(
        default="Cosine",
        description="Distance metric for vector similarity (Cosine, Euclid, Dot)",
    )
    on_disk: bool = Field(
        default=False,
        description="Store vectors on disk instead of RAM",
    )

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        """Validate collection name is a valid identifier."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "collection_name must contain only alphanumeric, underscore, or hyphen"
            )
        if len(v) > COLLECTION_NAME_MAX_LENGTH:
            raise ValueError(
                f"collection_name cannot exceed {COLLECTION_NAME_MAX_LENGTH} characters"
            )
        return v

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric is supported by Qdrant."""
        valid_metrics = {"Cosine", "Euclid", "Dot"}
        if v not in valid_metrics:
            raise ValueError(
                f"distance_metric must be one of: {', '.join(sorted(valid_metrics))}"
            )
        return v

    @model_validator(mode="after")
    def validate_grpc_configuration(self) -> ModelQdrantConfig:
        """Validate gRPC configuration consistency."""
        if self.prefer_grpc and self.grpc_port is None:
            raise ValueError("grpc_port must be specified when prefer_grpc is True")
        return self
