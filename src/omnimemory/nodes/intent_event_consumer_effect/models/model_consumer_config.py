"""Consumer configuration model for intent event consumer."""

from pydantic import BaseModel, ConfigDict, Field


# omnimemory-model-exempt: handler config
class ModelIntentEventConsumerConfig(BaseModel):
    """Configuration for intent event consumer.

    Note: consumer_group is NOT configured here. It is derived from
    ModelNodeIdentity via compute_consumer_group_id() per ADR.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # Topic configuration (suffixes only - env prefix added at runtime)
    subscribe_topic_suffix: str = Field(
        default="onex.evt.omniintelligence.intent-classified.v1",
        description="Topic suffix to subscribe to",
    )
    publish_stored_topic_suffix: str = Field(
        default="onex.evt.omnimemory.intent-stored.v1",
        description="Topic suffix for storage events (success and error use same topic via status field)",
    )
    dlq_topic_suffix: str = Field(
        default="onex.evt.omniintelligence.intent-classified.v1.dlq",
        description="Dead letter queue topic suffix",
    )

    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        ge=1,
        description="Failures before circuit opens",
    )
    circuit_breaker_recovery_timeout_seconds: int = Field(
        default=60,
        ge=1,
        description="Seconds before circuit half-opens",
    )

    # Health check configuration
    staleness_threshold_seconds: float = Field(
        default=300.0,
        ge=0,
        description="Seconds without consumption before marked stale",
    )

    # Retry configuration
    retry_max_attempts: int = Field(
        default=3,
        ge=0,
        description="Max retries before DLQ (0 = no retries)",
    )
    retry_backoff_base_seconds: float = Field(
        default=1.0,
        ge=0.1,
        description="Base backoff seconds (exponential: base * 2^attempt)",
    )
