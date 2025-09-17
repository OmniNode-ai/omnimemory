"""Memory storage configuration model."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class ModelMemoryStorageEffectConfig(BaseModel):
    """Configuration for memory storage effect node."""

    # Core configuration
    max_timeout_seconds: float = Field(default=30.0, gt=0)
    enable_error_sanitization: bool = Field(default=True)

    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = Field(default=5, gt=0)
    circuit_breaker_recovery_timeout: float = Field(default=60.0, gt=0)

    # Performance configuration
    max_concurrent_operations: int = Field(default=100, gt=0)
    operation_queue_size: int = Field(default=1000, gt=0)

    # Memory-specific configuration
    default_memory_type: str = Field(default="persistent")
    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, gt=0)

    # Storage backend configuration
    enable_vector_storage: bool = Field(default=True)
    enable_persistent_storage: bool = Field(default=True)
    enable_temporal_storage: bool = Field(default=True)

    # Search configuration
    default_similarity_threshold: float = Field(default=0.8, ge=0, le=1)
    max_search_results: int = Field(default=100, gt=0)
    enable_semantic_search: bool = Field(default=True)

    # Batch operation configuration
    max_batch_size: int = Field(default=100, gt=0)
    enable_batch_operations: bool = Field(default=True)

    # External system configuration
    postgres_connection_timeout: float = Field(default=10.0, gt=0)
    redis_connection_timeout: float = Field(default=5.0, gt=0)
    pinecone_connection_timeout: float = Field(default=15.0, gt=0)

    # Health check configuration
    health_check_interval: float = Field(default=30.0, gt=0)
    enable_external_health_checks: bool = Field(default=True)

    @classmethod
    def for_environment(cls, environment: str) -> "ModelMemoryStorageConfig":
        """Load environment-specific configuration."""
        base_config = {
            "max_timeout_seconds": 30.0,
            "enable_error_sanitization": True,
            "default_memory_type": "persistent",
            "enable_caching": True,
        }

        if environment == "production":
            return cls(
                **base_config,
                max_timeout_seconds=10.0,
                max_concurrent_operations=200,
                operation_queue_size=2000,
                cache_ttl_seconds=7200,
                max_batch_size=50,
                postgres_connection_timeout=5.0,
                redis_connection_timeout=3.0,
                pinecone_connection_timeout=10.0,
            )
        elif environment == "staging":
            return cls(
                **base_config,
                max_timeout_seconds=15.0,
                max_concurrent_operations=100,
                cache_ttl_seconds=1800,
                max_batch_size=75,
            )
        elif environment == "development":
            dev_config = base_config.copy()
            dev_config.update(
                {
                    "enable_error_sanitization": False,
                    "max_timeout_seconds": 60.0,
                    "cache_ttl_seconds": 300,
                    "max_batch_size": 10,
                    "health_check_interval": 60.0,
                }
            )
            return cls(**dev_config)
        else:
            return cls(**base_config)
