"""Qdrant Adapter Configuration Model for OmniMemory integration."""

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelQdrantAdapterConfig(BaseModel):
    """
    Configuration model for Qdrant adapter with environment-specific settings.

    Follows ONEX configuration patterns with validation, environment handling,
    and security-aware defaults.
    """

    # Qdrant Connection Settings
    qdrant_host: str = Field(
        default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"),
        description="Qdrant server host",
    )
    qdrant_port: int = Field(
        default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")),
        description="Qdrant server port",
    )
    qdrant_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("QDRANT_URL"),
        description="Full Qdrant URL (overrides host/port if provided)",
    )
    qdrant_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY"),
        description="Qdrant API key for authentication",
    )
    qdrant_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("QDRANT_TIMEOUT_SECONDS", "30")),
        description="Qdrant request timeout in seconds",
    )

    # Vector Operation Limits
    max_vector_dimensions: int = Field(
        default=4096, description="Maximum allowed vector dimensions"
    )
    max_batch_size: int = Field(
        default=100, description="Maximum batch size for bulk operations"
    )
    max_search_limit: int = Field(
        default=1000, description="Maximum number of results in search operations"
    )

    # Performance Settings
    connection_pool_size: int = Field(
        default=10, description="HTTP connection pool size"
    )
    max_retries: int = Field(
        default=3, description="Maximum retries for failed operations"
    )
    retry_delay_seconds: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )

    # Security Settings
    enable_error_sanitization: bool = Field(
        default=True,
        description="Enable error message sanitization to prevent information leakage",
    )
    enable_request_validation: bool = Field(
        default=True, description="Enable request validation for security"
    )

    # Circuit Breaker Settings
    circuit_breaker_failure_threshold: int = Field(
        default=5, description="Number of failures before opening circuit breaker"
    )
    circuit_breaker_timeout_seconds: int = Field(
        default=60, description="Circuit breaker timeout before attempting recovery"
    )
    circuit_breaker_half_open_max_calls: int = Field(
        default=3,
        description="Maximum calls allowed in half-open circuit breaker state",
    )

    # Default Collection Settings
    default_collection: str = Field(
        default="omnimemory_vectors",
        description="Default collection name for operations",
    )
    default_vector_size: int = Field(
        default=1536, description="Default vector size for new collections"
    )
    default_distance_metric: str = Field(
        default="Cosine",
        description="Default distance metric for new collections (Cosine, Dot, Euclidean)",
    )

    @classmethod
    def for_environment(cls, environment: str) -> "ModelQdrantAdapterConfig":
        """
        Create configuration optimized for specific environment.

        Args:
            environment: Target environment (development, staging, production)

        Returns:
            Environment-specific configuration
        """
        if environment in ["production", "prod"]:
            return cls(
                qdrant_timeout_seconds=10,
                max_retries=5,
                retry_delay_seconds=0.5,
                connection_pool_size=20,
                enable_error_sanitization=True,
                enable_request_validation=True,
                circuit_breaker_failure_threshold=3,
                circuit_breaker_timeout_seconds=30,
            )
        elif environment in ["staging", "stage"]:
            return cls(
                qdrant_timeout_seconds=15,
                max_retries=3,
                retry_delay_seconds=1.0,
                connection_pool_size=15,
                enable_error_sanitization=True,
                enable_request_validation=True,
                circuit_breaker_failure_threshold=5,
                circuit_breaker_timeout_seconds=45,
            )
        else:  # development
            return cls(
                qdrant_timeout_seconds=30,
                max_retries=2,
                retry_delay_seconds=2.0,
                connection_pool_size=5,
                enable_error_sanitization=False,  # Allow full errors in dev
                enable_request_validation=True,
                circuit_breaker_failure_threshold=10,
                circuit_breaker_timeout_seconds=60,
            )

    def get_qdrant_client_config(self) -> Dict[str, Any]:
        """
        Get Qdrant client configuration dictionary.

        Returns:
            Configuration dictionary for qdrant-client
        """
        config = {
            "timeout": self.qdrant_timeout_seconds,
        }

        if self.qdrant_url:
            config["url"] = self.qdrant_url
        else:
            config["host"] = self.qdrant_host
            config["port"] = self.qdrant_port

        if self.qdrant_api_key:
            config["api_key"] = self.qdrant_api_key

        return config

    def get_complexity_weights(self) -> Dict[str, int]:
        """
        Get complexity scoring weights for request validation.

        Returns:
            Dictionary of complexity weights
        """
        return {
            "vector_search": 2,
            "batch_upsert": 3,
            "filtered_search": 4,
            "scroll_search": 5,
            "create_collection": 10,
        }

    def validate_vector_dimensions(self, dimensions: int) -> bool:
        """
        Validate vector dimensions against limits.

        Args:
            dimensions: Number of vector dimensions

        Returns:
            True if valid, False otherwise
        """
        return 1 <= dimensions <= self.max_vector_dimensions

    def validate_search_limit(self, limit: int) -> bool:
        """
        Validate search limit against maximum.

        Args:
            limit: Search result limit

        Returns:
            True if valid, False otherwise
        """
        return 1 <= limit <= self.max_search_limit

    def validate_batch_size(self, batch_size: int) -> bool:
        """
        Validate batch size against limits.

        Args:
            batch_size: Size of batch operation

        Returns:
            True if valid, False otherwise
        """
        return 1 <= batch_size <= self.max_batch_size
