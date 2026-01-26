"""Subscription Handler for agent subscriptions and memory change notifications.

This module provides the core subscription management functionality:
- subscribe(): Register agent subscriptions to memory topics
- unsubscribe(): Remove agent subscriptions
- notify(): Publish notification events to Kafka for subscriber consumption
- list_subscriptions(): Get subscriptions for an agent (with optional pagination)

Architecture:
- Persistence: Valkey (fast lookups) + PostgreSQL (source of truth)
- Delivery: Kafka event bus (agents consume directly)
- Topic naming: memory.<entity>.<event> convention

Event Bus Strategy:
    Notifications are published to Kafka topics. Internal agents consume
    events directly via consumer groups. If external (non-Kafka) delivery
    is needed in the future, implement a WebhookEmitterEffect node that
    consumes bus events and handles HTTP delivery separately.

Known Limitations:
    Transaction Boundaries:
        The subscribe() and unsubscribe() operations persist to PostgreSQL
        first, then update the Valkey cache. These are NOT atomic operations.
        If the process crashes after DB write but before cache update, the
        cache may be inconsistent until the next cold start triggers
        _rebuild_cache_from_db().

        Mitigation: Cache rebuild on initialization ensures eventual consistency.
        The _rebuild_cache_from_db() method is called during initialize() and
        reconstructs the entire Valkey cache from PostgreSQL, which is the
        authoritative source of truth. This means any inconsistency is
        automatically resolved on the next handler startup.

        For strict consistency requirements, consider implementing:
        - A write-ahead log (WAL) pattern
        - Redis transactions with WATCH/MULTI/EXEC
        - Two-phase commit with compensation logic

        Current trade-off rationale: The eventual consistency model was chosen
        because (1) subscription changes are infrequent compared to reads,
        (2) cold start recovery is fast and automatic, and (3) the added
        complexity of distributed transactions was deemed unnecessary for
        the current use case.

Example::

    from omnibase_core.container import ModelONEXContainer
    from omnimemory.handlers import (
        HandlerSubscription,
        ModelHandlerSubscriptionConfig,
    )
    from omnimemory.models.subscription import (
        ModelNotificationEvent,
        ModelNotificationEventPayload,
    )

    container = ModelONEXContainer()
    config = ModelHandlerSubscriptionConfig(
        db_dsn="postgresql://user:pass@localhost:5432/omnimemory",
        valkey_host="localhost",
        valkey_port=6379,
        kafka_bootstrap_servers="localhost:9092",
    )
    handler = HandlerSubscription(container)
    await handler.initialize(config)

    # Subscribe an agent
    subscription = await handler.subscribe(
        agent_id="agent_123",
        topic="memory.item.created",
    )

    # Notify all subscribers (publishes to Kafka)
    event = ModelNotificationEvent(
        event_id="evt_456",
        topic="memory.item.created",
        payload=ModelNotificationEventPayload(
            entity_type="item",
            entity_id="item_789",
            action="created",
        ),
    )
    await handler.notify("memory.item.created", event)

    await handler.shutdown()

.. versionadded:: 0.1.0
    Initial implementation for OMN-1393.

.. versionchanged:: 0.2.0
    Removed webhook delivery in favor of Kafka event bus.
    Webhook delivery moved to optional WebhookEmitterEffect node.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from omnimemory.enums.enum_subscription_status import EnumSubscriptionStatus
from omnimemory.handlers.adapters.adapter_valkey import (
    AdapterValkey,
    AdapterValkeyConfig,
)
from omnimemory.models.subscription import (
    ModelNotificationEvent,
    ModelSubscription,
)
from omnimemory.models.subscription.constants import TOPIC_PATTERN

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

# Optional omnibase_infra imports for handler reuse
_OMNIBASE_INFRA_AVAILABLE = False
_OMNIBASE_INFRA_IMPORT_ERROR: str | None = None

try:
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.handlers.handler_db import HandlerDb

    _OMNIBASE_INFRA_AVAILABLE = True
except ImportError as e:
    _OMNIBASE_INFRA_IMPORT_ERROR = str(e)

    # Provide stubs for type checking
    class HandlerDb:  # type: ignore[no-redef]
        """Stub for HandlerDb when omnibase_infra is not installed."""

    class EventBusKafka:  # type: ignore[no-redef]
        """Stub for EventBusKafka when omnibase_infra is not installed."""


logger = logging.getLogger(__name__)

__all__ = [
    "HandlerSubscription",
    "ModelHandlerSubscriptionConfig",
    "ModelPaginatedSubscriptions",
    "ModelSubscriptionHealth",
    "ModelSubscriptionMetadata",
    "ModelSubscriptionMetrics",
]

# Cache key patterns
CACHE_KEY_TOPIC_SUBSCRIBERS = "topic:{topic}:subscribers"
CACHE_KEY_AGENT_SUBSCRIPTIONS = "agent:{agent_id}:subscriptions"
CACHE_KEY_SUBSCRIPTION = "subscription:{subscription_id}"

# Maximum cached entries for SQL placeholder generation.
# 256 entries covers common batch sizes (1-128 items) with various start offsets,
# providing ~95% hit rate for typical pagination and bulk operations.
# Larger batches are rare and regeneration overhead is minimal (~1us).
_SQL_PLACEHOLDERS_CACHE_SIZE = 256

# Default batch size for cache rebuild from database.
# This value is used as the default for config.cache_rebuild_batch_size.
# Configurable via ModelHandlerSubscriptionConfig for deployments with
# different memory constraints (100K+ subscriptions may need smaller batches).
_CACHE_REBUILD_BATCH_SIZE = 1000


@lru_cache(maxsize=_SQL_PLACEHOLDERS_CACHE_SIZE)
def _sql_placeholders(count: int, start: int = 1) -> str:
    """Generate SQL parameter placeholders for parameterized queries.

    Results are cached with LRU policy to avoid repeated string generation
    for common query sizes. See _SQL_PLACEHOLDERS_CACHE_SIZE for capacity.

    Args:
        count: Number of placeholders to generate. If <= 0, returns empty string.
               Must not exceed 10000 for safety.
        start: Starting index (default 1 for PostgreSQL $1, $2, ...).
               Must be >= 1.

    Returns:
        Comma-separated placeholder string (e.g., "$1, $2, $3").
        Returns empty string if count <= 0.

    Raises:
        ValueError: If start < 1 (PostgreSQL placeholders start at $1).
        ValueError: If count > 10000 (safety limit for batch operations).

    Example:
        >>> _sql_placeholders(3)
        '$1, $2, $3'
        >>> _sql_placeholders(2, start=5)
        '$5, $6'
        >>> _sql_placeholders(0)
        ''
    """
    if start < 1:
        raise ValueError(f"start must be >= 1 for PostgreSQL placeholders, got {start}")
    if count > 10000:
        raise ValueError(f"count exceeds maximum (10000) for safety, got {count}")
    if count <= 0:
        return ""
    return ", ".join(f"${i}" for i in range(start, start + count))


class ModelHandlerSubscriptionConfig(  # omnimemory-model-exempt: handler config
    BaseModel
):
    """Configuration for the Subscription Handler.

    Attributes:
        db_dsn: PostgreSQL connection string.
        valkey_host: Valkey server hostname.
        valkey_port: Valkey server port.
        valkey_db: Valkey database index.
        valkey_password: Optional Valkey password.
        kafka_bootstrap_servers: Kafka bootstrap servers.
        cache_ttl_seconds: TTL for cached subscription data.
        kafka_notification_topic: Kafka topic for memory notification events.
        pagination_max_limit: Maximum allowed limit for pagination queries.
        cache_rebuild_batch_size: Batch size for cache rebuild operations.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    db_dsn: SecretStr = Field(
        ...,
        description="PostgreSQL connection string",
    )
    valkey_host: str = Field(
        default="localhost",
        description="Valkey server hostname",
    )
    valkey_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Valkey server port",
    )
    valkey_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Valkey database index",
    )
    valkey_password: SecretStr | None = Field(
        default=None,
        description="Optional Valkey password",
    )
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers (comma-separated)",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="TTL for cached subscription data",
    )
    kafka_notification_topic: str = Field(
        default="omnimemory.memory.notification.v1",
        description="Kafka topic for memory notification events",
    )
    pagination_max_limit: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum allowed limit for pagination queries",
    )
    cache_rebuild_batch_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Batch size for cache rebuild from database (limits memory usage)",
    )


class ModelSubscriptionMetrics(  # omnimemory-model-exempt: handler internal
    BaseModel
):
    """Metrics for the Subscription Handler.

    Tracks counters for various operations to enable production monitoring
    and observability.

    Attributes:
        notifications_published: Count of notifications published to Kafka.
        subscriptions_created: Count of new subscriptions created.
        subscriptions_updated: Count of existing subscriptions updated (re-subscriptions).
        subscriptions_deleted: Count of subscriptions deleted.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )

    notifications_published: int = Field(
        default=0,
        ge=0,
        description="Count of notifications published to Kafka",
    )
    subscriptions_created: int = Field(
        default=0,
        ge=0,
        description="Count of new subscriptions created",
    )
    subscriptions_updated: int = Field(
        default=0,
        ge=0,
        description="Count of existing subscriptions updated (re-subscriptions)",
    )
    subscriptions_deleted: int = Field(
        default=0,
        ge=0,
        description="Count of subscriptions deleted",
    )


class ModelSubscriptionHealth(  # omnimemory-model-exempt: handler health
    BaseModel
):
    """Health status for the Subscription Handler.

    Attributes:
        is_healthy: Overall health status.
        initialized: Whether the handler has been initialized.
        db_healthy: Database connection health.
        valkey_healthy: Valkey connection health.
        kafka_healthy: Kafka connection health.
        error_message: Error details if unhealthy.
        metrics: Optional metrics for observability.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )

    is_healthy: bool = Field(
        ...,
        description="Overall health status",
    )
    initialized: bool = Field(
        ...,
        description="Whether the handler has been initialized",
    )
    db_healthy: bool | None = Field(
        default=None,
        description="Database connection health",
    )
    valkey_healthy: bool | None = Field(
        default=None,
        description="Valkey connection health",
    )
    kafka_healthy: bool | None = Field(
        default=None,
        description="Kafka connection health",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if unhealthy",
    )
    metrics: ModelSubscriptionMetrics | None = Field(
        default=None,
        description="Handler metrics for observability",
    )


class ModelPaginatedSubscriptions(  # omnimemory-model-exempt: handler result
    BaseModel
):
    """Paginated subscription list response.

    Returned by list_subscriptions() when pagination parameters are provided.
    Contains the subscription list along with pagination metadata for UI support.

    Attributes:
        subscriptions: List of subscriptions for the current page.
        total_count: Total number of subscriptions matching the query.
        limit: Requested limit (None if no pagination).
        offset: Requested offset.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    subscriptions: list[ModelSubscription] = Field(
        description="List of subscriptions for the current page",
    )
    total_count: int = Field(
        ge=0,
        description="Total number of subscriptions matching the query",
    )
    limit: int | None = Field(
        default=None,
        description="Requested limit (None if no pagination)",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Requested offset",
    )


class ModelSubscriptionMetadata(  # omnimemory-model-exempt: handler metadata
    BaseModel
):
    """Metadata describing handler capabilities and configuration.

    Returned by describe() method to provide introspection information
    about the handler's capabilities and current configuration.

    Attributes:
        handler_type: Type identifier for this handler.
        capabilities: List of supported operations and features.
        supports_transactions: Whether the handler supports transactional operations.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    handler_type: str = Field(
        ...,
        description="Type identifier for this handler",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="List of supported operations and features",
    )
    supports_transactions: bool = Field(
        default=False,
        description="Whether the handler supports transactional operations",
    )


class HandlerSubscription:
    """Handler for agent subscriptions and memory change notifications.

    Manages the lifecycle of subscriptions and publishes notification events
    to Kafka for consumption by subscribing agents.

    Architecture:
        - Subscription store: PostgreSQL (source of truth) + Valkey (cache)
        - Notification delivery: Kafka event bus
        - Agents consume events directly via consumer groups

    Note on External Delivery:
        If webhook delivery to external systems is needed, implement a
        WebhookEmitterEffect node that consumes Kafka events and handles
        HTTP delivery with its own retry/circuit breaker logic.

    ONEX Container Pattern:
        This handler follows the ONEX container-driven pattern:
        - Constructor takes only ModelONEXContainer
        - initialize() accepts configuration and creates dependencies
        - Handler owns all dependency lifecycles
        - describe() provides handler metadata

    Attributes:
        config: The handler configuration (set during initialize).
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize HandlerSubscription with ONEX container.

        Args:
            container: ONEX container providing dependency injection for
                services, configuration, and runtime context.

        Note:
            The container is stored for interface compliance with the standard
            ONEX handler pattern (def __init__(self, container: ModelONEXContainer))
            and to enable future DI-based service resolution.

        Raises:
            ImportError: If omnibase_infra is not installed.
        """
        if not _OMNIBASE_INFRA_AVAILABLE:
            raise ImportError(
                f"omnibase_infra is required for HandlerSubscription. "
                f"Install it with: poetry install --with dev. "
                f"Original error: {_OMNIBASE_INFRA_IMPORT_ERROR}"
            )

        self._container = container
        self._config: ModelHandlerSubscriptionConfig | None = None
        self._db_handler: HandlerDb | None = None
        self._kafka_handler: EventBusKafka | None = None
        self._valkey: AdapterValkey | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._cache_rebuild_lock = asyncio.Lock()
        self._metrics_lock = asyncio.Lock()

        # Metrics for observability (type-safe model instance)
        self._metrics = ModelSubscriptionMetrics()

    @property
    def handler_type(self) -> str:
        """Return the handler type identifier.

        Returns:
            String "subscription" identifying this handler type.
        """
        return "subscription"

    @property
    def config(self) -> ModelHandlerSubscriptionConfig:
        """Get the handler configuration.

        Raises:
            RuntimeError: If handler is not initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "HandlerSubscription is not initialized. "
                "Call initialize() before accessing config."
            )
        assert self._config is not None  # For type checker
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the handler has been initialized."""
        return self._initialized

    async def initialize(self, config: ModelHandlerSubscriptionConfig) -> None:
        """Initialize DB, Valkey, and Kafka handlers.

        Creates connections to all required services and optionally
        rebuilds the Valkey cache from PostgreSQL on cold start.

        Args:
            config: The handler configuration specifying database, Valkey,
                and Kafka connection details.

        Raises:
            RuntimeError: If initialization fails.
        """
        async with self._init_lock:
            if self._initialized:
                return

            # Store configuration
            self._config = config

            try:
                # Initialize Valkey adapter
                valkey_config = AdapterValkeyConfig(
                    host=self._config.valkey_host,
                    port=self._config.valkey_port,
                    db=self._config.valkey_db,
                    password=self._config.valkey_password,
                    key_prefix="omnimemory:subscription:",
                )
                self._valkey = AdapterValkey(valkey_config)
                await self._valkey.initialize()
                logger.info("Valkey adapter initialized")

                # Initialize DB handler
                self._db_handler = HandlerDb()
                await self._db_handler.initialize(
                    {
                        "dsn": self._config.db_dsn.get_secret_value(),
                    }
                )
                logger.info("Database handler initialized")

                # Initialize Kafka handler
                self._kafka_handler = EventBusKafka()
                await self._kafka_handler.initialize(
                    {
                        "bootstrap_servers": self._config.kafka_bootstrap_servers,
                    }
                )
                logger.info("Kafka handler initialized")

                # Rebuild cache from DB on cold start
                await self._rebuild_cache_from_db()

                self._initialized = True
                logger.info("HandlerSubscription initialized successfully")

            except Exception as e:
                logger.error("Failed to initialize HandlerSubscription: %s", e)
                await self._cleanup_partial_init()
                raise RuntimeError(f"Initialization failed: {e}") from e

    async def _cleanup_partial_init(self) -> None:
        """Cleanup partially initialized resources."""
        if self._valkey:
            try:
                await self._valkey.shutdown()
            except Exception as e:
                logger.warning("Failed to shutdown Valkey during cleanup: %s", e)
            self._valkey = None

        if self._db_handler:
            try:
                await self._db_handler.shutdown()
            except Exception as e:
                logger.warning("Failed to shutdown DB handler during cleanup: %s", e)
            self._db_handler = None

        if self._kafka_handler:
            try:
                await self._kafka_handler.shutdown()
            except Exception as e:
                logger.warning("Failed to shutdown Kafka handler during cleanup: %s", e)
            self._kafka_handler = None

    async def shutdown(self) -> None:
        """Cleanup all resources."""
        async with self._init_lock:
            if not self._initialized:
                return

            if self._valkey:
                await self._valkey.shutdown()
                self._valkey = None

            if self._db_handler:
                await self._db_handler.shutdown()
                self._db_handler = None

            if self._kafka_handler:
                await self._kafka_handler.shutdown()
                self._kafka_handler = None

            self._initialized = False
            logger.info("HandlerSubscription shutdown complete")

    def _ensure_initialized(
        self,
    ) -> tuple[AdapterValkey, HandlerDb, EventBusKafka, ModelHandlerSubscriptionConfig]:
        """Ensure handler is initialized and return components.

        Returns:
            Tuple of (valkey, db_handler, kafka_handler, config).

        Raises:
            RuntimeError: If handler is not initialized.
        """
        if (
            not self._initialized
            or self._valkey is None
            or self._db_handler is None
            or self._kafka_handler is None
            or self._config is None
        ):
            raise RuntimeError(
                "HandlerSubscription not initialized. Call initialize(config) first."
            )
        return self._valkey, self._db_handler, self._kafka_handler, self._config

    # =========================================================================
    # Core Operations
    # =========================================================================

    async def subscribe(
        self,
        agent_id: str,
        topic: str,
        metadata: dict[str, str] | None = None,
    ) -> ModelSubscription:
        """Register a new subscription.

        Workflow:
            1. Validate topic format (memory.<entity>.<event>)
            2. Check for existing subscription (upsert behavior)
            3. Create/update subscription record in Postgres
            4. Add to Valkey cache: topic:subscribers -> subscription_id
            5. Add to Valkey cache: agent:subscriptions -> subscription_id

        Args:
            agent_id: The subscribing agent's identifier.
            topic: Topic pattern (format: memory.<entity>.<event>).
            metadata: Optional subscription metadata.

        Returns:
            The created or updated subscription.

        Raises:
            ValueError: If topic format is invalid.
            RuntimeError: If handler is not initialized.
        """
        valkey, _, _, config = self._ensure_initialized()

        # Early topic validation with clear error message for better debugging
        # (ModelSubscription also validates, but this provides subscribe()-specific context)
        if not TOPIC_PATTERN.match(topic):
            raise ValueError(
                f"subscribe() received invalid topic format: '{topic}'. "
                f"Expected pattern: memory.<entity>.<event> (e.g., memory.item.created)"
            )

        # Generate subscription ID
        subscription_id = str(uuid4())
        now = datetime.now(timezone.utc)

        # Check for existing subscription (agent_id, topic unique constraint)
        existing = await self._get_subscription_by_agent_and_topic(agent_id, topic)
        if existing:
            # Update existing subscription
            subscription_id = existing.id
            logger.info(
                "Updating existing subscription %s for agent %s on topic %s",
                subscription_id,
                agent_id,
                topic,
            )

        # Create subscription model
        subscription = ModelSubscription(
            id=subscription_id,
            agent_id=agent_id,
            topic=topic,
            status=EnumSubscriptionStatus.ACTIVE,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            metadata=metadata,
        )

        # Persist to PostgreSQL (source of truth)
        # NOTE: Transaction boundary limitation - DB persist and cache update are NOT
        # atomic. If process crashes after this point but before cache update completes,
        # cache may be inconsistent. Mitigation: _rebuild_cache_from_db() on cold start
        # ensures eventual consistency. See module docstring "Known Limitations" section.
        await self._persist_subscription(subscription, is_update=existing is not None)
        if existing is not None:
            await self._increment_metric("subscriptions_updated")
        else:
            await self._increment_metric("subscriptions_created")

        # Update Valkey caches (best effort - DB is source of truth)
        # Cache update happens AFTER DB persist - not atomic with DB operation above.
        try:
            topic_key = CACHE_KEY_TOPIC_SUBSCRIBERS.format(topic=topic)
            agent_key = CACHE_KEY_AGENT_SUBSCRIPTIONS.format(agent_id=agent_id)
            sub_key = CACHE_KEY_SUBSCRIPTION.format(subscription_id=subscription_id)

            async with valkey.pipeline() as pipe:
                pipe.sadd(topic_key, subscription_id)
                pipe.expire(topic_key, config.cache_ttl_seconds)
                pipe.sadd(agent_key, subscription_id)
                pipe.expire(agent_key, config.cache_ttl_seconds)
                pipe.set_key(
                    sub_key,
                    subscription.model_dump_json(),
                    ttl=config.cache_ttl_seconds,
                )
        except Exception as e:
            logger.warning(
                "Failed to update cache for subscription %s (DB persisted successfully): %s",
                subscription_id,
                e,
            )

        logger.info(
            "Subscription %s created/updated for agent %s on topic %s",
            subscription_id,
            agent_id,
            topic,
        )

        return subscription

    async def unsubscribe(
        self,
        agent_id: str,
        topic: str,
    ) -> bool:
        """Remove a subscription.

        Workflow:
            1. Find subscription in Postgres by (agent_id, topic)
            2. Mark as deleted (soft delete)
            3. Remove from Valkey caches

        Args:
            agent_id: The agent's identifier.
            topic: The topic to unsubscribe from.

        Returns:
            True if subscription was found and removed, False otherwise.

        Raises:
            RuntimeError: If handler is not initialized.
        """
        valkey, _, _, _ = self._ensure_initialized()

        # Find existing subscription
        subscription = await self._get_subscription_by_agent_and_topic(agent_id, topic)
        if not subscription:
            logger.warning(
                "No subscription found for agent %s on topic %s",
                agent_id,
                topic,
            )
            return False

        # Soft delete in PostgreSQL
        # NOTE: Transaction boundary limitation - DB delete and cache eviction are NOT
        # atomic. If process crashes after this point but before cache eviction completes,
        # cache may be inconsistent (stale subscription in cache). Mitigation:
        # _rebuild_cache_from_db() on cold start ensures eventual consistency.
        # See module docstring "Known Limitations" section.
        await self._soft_delete_subscription(subscription.id)
        await self._increment_metric("subscriptions_deleted")

        # Remove from Valkey caches (best effort - DB is source of truth)
        # Cache eviction happens AFTER DB delete - not atomic with DB operation above.
        topic_key = CACHE_KEY_TOPIC_SUBSCRIBERS.format(topic=topic)
        agent_key = CACHE_KEY_AGENT_SUBSCRIPTIONS.format(agent_id=agent_id)
        sub_key = CACHE_KEY_SUBSCRIPTION.format(subscription_id=subscription.id)

        try:
            await valkey.srem(topic_key, subscription.id)
            await valkey.srem(agent_key, subscription.id)
            await valkey.delete(sub_key)
        except Exception as e:
            logger.warning(
                "Failed to evict cache for subscription %s (DB delete succeeded): %s",
                subscription.id,
                e,
            )

        logger.info(
            "Subscription %s removed for agent %s on topic %s",
            subscription.id,
            agent_id,
            topic,
        )

        return True

    async def notify(
        self,
        topic: str,
        event: ModelNotificationEvent,
    ) -> int:
        """Publish notification event to Kafka for subscriber consumption.

        Agents subscribe to Kafka topics and consume events via consumer groups.
        This method publishes the event to the event bus - actual delivery to
        agents happens through their Kafka consumers.

        Args:
            topic: The topic to notify (format: memory.<entity>.<event>).
            event: The notification event to publish.

        Returns:
            Number of active subscribers for this topic.

        Raises:
            RuntimeError: If handler is not initialized.
            ValueError: If event.topic does not match the topic argument.
        """
        _, _, kafka_handler, config = self._ensure_initialized()

        # Validate that event topic matches the topic argument
        if event.topic != topic:
            raise ValueError(
                f"Event topic mismatch: event.topic='{event.topic}' does not match "
                f"topic argument='{topic}'. Ensure the event is being sent to the "
                f"correct topic."
            )

        # Get subscriber count for metrics/logging
        subscriber_ids = await self._get_subscribers_for_topic(topic)
        subscriber_count = len(subscriber_ids)

        if subscriber_count == 0:
            logger.debug("No subscribers for topic %s", topic)
            return 0

        # Publish event to Kafka
        # Agents consume from this topic via consumer groups keyed by agent_id
        kafka_topic = config.kafka_notification_topic
        envelope = {
            "operation": "kafka.produce",
            "payload": {
                "topic": kafka_topic,
                "key": topic,  # Partition by topic for ordering
                "value": event.model_dump_json(),
                "headers": {
                    "event_id": event.event_id,
                    "topic": topic,
                    "subscriber_count": str(subscriber_count),
                },
            },
        }
        result = await kafka_handler.execute(envelope)

        # Validate Kafka publish succeeded
        if result is None:
            logger.warning(
                "Kafka publish returned None for topic %s, event %s",
                kafka_topic,
                event.event_id,
            )
        elif hasattr(result, "result") and not result.result.get("success", True):
            logger.warning(
                "Kafka publish may have failed for topic %s, event %s: %s",
                kafka_topic,
                event.event_id,
                result.result.get("error", "unknown error"),
            )

        await self._increment_metric("notifications_published")

        logger.info(
            "Published notification for topic %s, event %s, %d subscribers",
            topic,
            event.event_id,
            subscriber_count,
        )

        return subscriber_count

    async def list_subscriptions(
        self,
        agent_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ModelSubscription] | ModelPaginatedSubscriptions:
        """Get subscriptions for an agent with optional pagination.

        When pagination parameters are provided (limit is not None), this method
        queries the database directly to ensure consistent ordering and returns
        a ModelPaginatedSubscriptions response with total_count for UI pagination.
        When no pagination is requested, it uses the Valkey cache for performance
        and returns a plain list for backward compatibility.

        Args:
            agent_id: The agent's identifier.
            limit: Maximum number of subscriptions to return. None means no limit
                (returns all subscriptions, backward compatible default).
            offset: Number of subscriptions to skip (default 0). Only meaningful
                when limit is provided.

        Returns:
            - When limit is None: list[ModelSubscription] (backward compatible)
            - When limit is provided: ModelPaginatedSubscriptions with total_count

        Raises:
            RuntimeError: If handler is not initialized.
            ValueError: If offset is negative or limit is non-positive.

        Example:
            # Get all subscriptions (backward compatible, returns list)
            all_subs = await handler.list_subscriptions("agent_123")

            # Get first 10 subscriptions with pagination info
            first_page = await handler.list_subscriptions("agent_123", limit=10)
            # first_page.subscriptions = [...]
            # first_page.total_count = 25
            # first_page.limit = 10
            # first_page.offset = 0

            # Get next 10 subscriptions
            second_page = await handler.list_subscriptions("agent_123", limit=10, offset=10)

        Note:
            **Ordering Behavior Difference**:

            - **Without pagination** (``limit=None``, the default): Results are retrieved
              from the Valkey cache for performance. The order of returned subscriptions
              is **NOT guaranteed** since Redis/Valkey sets are unordered collections.

            - **With pagination** (``limit`` provided): Results are retrieved directly
              from the database with ``ORDER BY created_at DESC``, ensuring consistent
              and deterministic ordering across calls.

            **Recommendation**: For consistent ordering across multiple calls or when
            implementing UI pagination, always provide a ``limit`` parameter even if
            you want all results (e.g., ``limit=1000``).
        """
        valkey, _, _, config = self._ensure_initialized()

        # Validate pagination parameters
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")
        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be positive when provided, got {limit}")
        if limit is not None and limit > config.pagination_max_limit:
            raise ValueError(
                f"limit exceeds maximum ({config.pagination_max_limit}), got {limit}"
            )

        # For paginated queries, always use database to ensure consistent ordering
        # Cache-based retrieval doesn't guarantee order, making pagination unreliable
        # Use atomic query with window function to prevent race condition between
        # fetching subscriptions and counting total
        if limit is not None:
            (
                subscriptions,
                total_count,
            ) = await self._get_subscriptions_with_count_from_db(
                agent_id, limit=limit, offset=offset
            )
            return ModelPaginatedSubscriptions(
                subscriptions=subscriptions,
                total_count=total_count,
                limit=limit,
                offset=offset,
            )

        # Non-paginated: Try Valkey cache first for performance
        agent_key = CACHE_KEY_AGENT_SUBSCRIPTIONS.format(agent_id=agent_id)
        try:
            subscription_ids = await valkey.smembers(agent_key)

            if subscription_ids:
                subscriptions = await self._load_subscriptions(subscription_ids)
                # Filter to only active subscriptions
                return [
                    s
                    for s in subscriptions
                    if s.status == EnumSubscriptionStatus.ACTIVE
                ]
        except Exception as e:
            logger.warning(
                "Valkey smembers failed for agent %s, falling back to database: %s",
                agent_id,
                e,
            )

        # Fallback to database
        return await self._get_subscriptions_from_db(agent_id)

    # =========================================================================
    # Internal Helpers - Database Operations
    # =========================================================================

    async def _persist_subscription(
        self,
        subscription: ModelSubscription,
        is_update: bool = False,
    ) -> None:
        """Persist subscription to PostgreSQL.

        Args:
            subscription: The subscription to persist.
            is_update: Whether this is an update (upsert).
        """
        _, db_handler, _, _ = self._ensure_initialized()

        if is_update:
            sql = """
                UPDATE subscriptions SET
                    status = $1,
                    updated_at = $2,
                    metadata = $3
                WHERE id = $4
            """
            params = [
                subscription.status.value,
                subscription.updated_at.isoformat(),
                json.dumps(subscription.metadata) if subscription.metadata else None,
                subscription.id,
            ]
        else:
            sql = """
                INSERT INTO subscriptions (
                    id, agent_id, topic, status,
                    created_at, updated_at, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (agent_id, topic) DO UPDATE SET
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata
            """
            params = [
                subscription.id,
                subscription.agent_id,
                subscription.topic,
                subscription.status.value,
                subscription.created_at.isoformat(),
                subscription.updated_at.isoformat(),
                json.dumps(subscription.metadata) if subscription.metadata else None,
            ]

        envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": sql,
                "parameters": params,
            },
        }
        await db_handler.execute(envelope)

    async def _soft_delete_subscription(self, subscription_id: str) -> None:
        """Soft delete a subscription by marking status as deleted.

        Args:
            subscription_id: The subscription ID to delete.
        """
        _, db_handler, _, _ = self._ensure_initialized()

        sql = """
            UPDATE subscriptions
            SET status = $1, updated_at = $2
            WHERE id = $3
        """
        envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": sql,
                "parameters": [
                    EnumSubscriptionStatus.DELETED.value,
                    datetime.now(timezone.utc).isoformat(),
                    subscription_id,
                ],
            },
        }
        await db_handler.execute(envelope)

    async def _get_subscription_by_agent_and_topic(
        self,
        agent_id: str,
        topic: str,
    ) -> ModelSubscription | None:
        """Get subscription by agent_id and topic.

        Args:
            agent_id: The agent's identifier.
            topic: The topic.

        Returns:
            The subscription if found, None otherwise.
        """
        _, db_handler, _, _ = self._ensure_initialized()

        sql = """
            SELECT id, agent_id, topic, status,
                   created_at, updated_at, metadata
            FROM subscriptions
            WHERE agent_id = $1 AND topic = $2 AND status != $3
        """
        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": sql,
                "parameters": [agent_id, topic, EnumSubscriptionStatus.DELETED.value],
            },
        }
        result = await db_handler.execute(envelope)

        rows = result.result.get("payload", {}).get("rows", [])
        if not rows:
            return None

        return self._row_to_subscription(rows[0])

    async def _get_subscriptions_from_db(
        self,
        agent_id: str,
    ) -> list[ModelSubscription]:
        """Get all active subscriptions for an agent from database.

        This method is used for non-paginated queries (cache fallback).
        For paginated queries with total count, use
        _get_subscriptions_with_count_from_db() which returns both results
        and total count atomically using a window function.

        Args:
            agent_id: The agent's identifier.

        Returns:
            List of all active subscriptions ordered by created_at DESC.
        """
        _, db_handler, _, _ = self._ensure_initialized()

        sql = """
            SELECT id, agent_id, topic, status,
                   created_at, updated_at, metadata
            FROM subscriptions
            WHERE agent_id = $1 AND status = $2
            ORDER BY created_at DESC
        """
        parameters: list[str | int] = [agent_id, EnumSubscriptionStatus.ACTIVE.value]

        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": sql,
                "parameters": parameters,
            },
        }
        result = await db_handler.execute(envelope)

        rows = result.result.get("payload", {}).get("rows", [])
        return [self._row_to_subscription(row) for row in rows]

    async def _get_subscriptions_with_count_from_db(
        self,
        agent_id: str,
        limit: int,
        offset: int = 0,
    ) -> tuple[list[ModelSubscription], int]:
        """Get paginated subscriptions with total count atomically.

        Uses PostgreSQL window function COUNT(*) OVER() to retrieve both
        the paginated subscription list and the total count in a single
        atomic query. This prevents race conditions where the count could
        become inconsistent with the results if subscriptions change
        between separate queries.

        Args:
            agent_id: The agent's identifier.
            limit: Maximum number of subscriptions to return.
            offset: Number of subscriptions to skip (default 0).

        Returns:
            Tuple of (subscriptions, total_count) where:
            - subscriptions: List of subscriptions for the requested page
            - total_count: Total number of matching subscriptions (before pagination)
        """
        _, db_handler, _, _ = self._ensure_initialized()

        # Use window function to get total count in the same query
        # COUNT(*) OVER() returns the total count for each row without grouping
        sql = """
            SELECT id, agent_id, topic, status,
                   created_at, updated_at, metadata,
                   COUNT(*) OVER() as total_count
            FROM subscriptions
            WHERE agent_id = $1 AND status = $2
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
        """
        parameters: list[str | int] = [
            agent_id,
            EnumSubscriptionStatus.ACTIVE.value,
            limit,
            offset,
        ]

        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": sql,
                "parameters": parameters,
            },
        }
        result = await db_handler.execute(envelope)

        rows = result.result.get("payload", {}).get("rows", [])

        if not rows:
            # No results - total count is 0
            return [], 0

        # Extract total_count from the first row (same for all rows due to OVER())
        total_count = int(rows[0]["total_count"])

        # Convert rows to subscriptions
        subscriptions = [self._row_to_subscription(row) for row in rows]

        return subscriptions, total_count

    def _row_to_subscription(self, row: dict[str, object]) -> ModelSubscription:
        """Convert database row to ModelSubscription.

        Handles various return types from different DB drivers:
        - JSONB fields may return dict (asyncpg) or str (other drivers)
        - Datetime fields may return datetime objects or ISO strings

        Args:
            row: Database row dict.

        Returns:
            ModelSubscription instance.
        """
        # Parse JSON/JSONB fields with explicit type handling
        # Some DB drivers (asyncpg) return JSONB as dict, others return str
        metadata: dict[str, str] | None = None
        metadata_raw = row.get("metadata")
        if metadata_raw is not None:
            if isinstance(metadata_raw, dict):
                # JSONB already parsed by driver (e.g., asyncpg)
                metadata = cast(dict[str, str], metadata_raw)
            elif isinstance(metadata_raw, str):
                # JSON string needs parsing
                metadata = json.loads(metadata_raw)
            else:
                # Fallback: convert to string and parse
                logger.warning(
                    "Unexpected metadata type %s, attempting string conversion",
                    type(metadata_raw).__name__,
                )
                metadata = json.loads(str(metadata_raw))

        # Parse datetime fields with proper type handling
        created_at_raw = row["created_at"]
        if isinstance(created_at_raw, str):
            created_at_parsed = datetime.fromisoformat(
                created_at_raw.replace("Z", "+00:00")
            )
        else:
            created_at_parsed = cast(datetime, created_at_raw)

        updated_at_raw = row["updated_at"]
        if isinstance(updated_at_raw, str):
            updated_at_parsed = datetime.fromisoformat(
                updated_at_raw.replace("Z", "+00:00")
            )
        else:
            updated_at_parsed = cast(datetime, updated_at_raw)

        return ModelSubscription(
            id=str(row["id"]),
            agent_id=str(row["agent_id"]),
            topic=str(row["topic"]),
            status=EnumSubscriptionStatus(str(row["status"])),
            created_at=created_at_parsed,
            updated_at=updated_at_parsed,
            metadata=metadata,
        )

    # =========================================================================
    # Internal Helpers - Cache Operations
    # =========================================================================

    async def _rebuild_cache_from_db(self) -> None:
        """Cold start recovery: rebuild Valkey from Postgres.

        Processes subscriptions in batches to limit memory usage for large
        subscription counts (100K+). Each batch is processed atomically via
        pipeline before loading the next batch.

        This method is the primary mitigation for the transaction boundary
        limitation documented in the module docstring. Since subscribe() and
        unsubscribe() operations persist to PostgreSQL before updating the
        Valkey cache (non-atomic), a crash between these operations can leave
        the cache inconsistent. This method rebuilds the entire cache from
        PostgreSQL (the source of truth), ensuring eventual consistency on
        every handler restart.

        Note:
            This method accesses components directly rather than using
            _ensure_initialized() because it is called during initialize()
            before _initialized is set to True. The components are already
            initialized by the time this method is called.

        Raises:
            RuntimeError: If required components are not available.
        """
        async with self._cache_rebuild_lock:
            # Access components directly - this is called during initialize()
            # before _initialized is set, so we can't use _ensure_initialized()
            if self._valkey is None or self._db_handler is None or self._config is None:
                raise RuntimeError(
                    "_rebuild_cache_from_db called before components initialized"
                )
            valkey = self._valkey
            db_handler = self._db_handler
            config = self._config

            logger.info("Rebuilding Valkey cache from PostgreSQL...")

            # Count total for progress logging
            count_sql = """
                SELECT COUNT(*) as count FROM subscriptions WHERE status = $1
            """
            count_envelope = {
                "operation": "db.query",
                "payload": {
                    "sql": count_sql,
                    "parameters": [EnumSubscriptionStatus.ACTIVE.value],
                },
            }
            count_result = await db_handler.execute(count_envelope)
            total_count = int(
                count_result.result.get("payload", {})
                .get("rows", [{}])[0]
                .get("count", 0)
            )

            if total_count == 0:
                logger.info("No subscriptions to cache, skipping rebuild")
                return

            logger.info("Found %d active subscriptions to cache", total_count)

            # Process in batches for memory efficiency
            offset = 0
            processed = 0
            while offset < total_count:
                sql = """
                    SELECT id, agent_id, topic, status,
                           created_at, updated_at, metadata
                    FROM subscriptions
                    WHERE status = $1
                    ORDER BY id
                    LIMIT $2 OFFSET $3
                """
                envelope = {
                    "operation": "db.query",
                    "payload": {
                        "sql": sql,
                        "parameters": [
                            EnumSubscriptionStatus.ACTIVE.value,
                            config.cache_rebuild_batch_size,
                            offset,
                        ],
                    },
                }
                result = await db_handler.execute(envelope)
                rows = result.result.get("payload", {}).get("rows", [])

                if not rows:
                    break

                # Use pipeline for atomic batch update
                async with valkey.pipeline() as pipe:
                    for row in rows:
                        subscription = self._row_to_subscription(row)

                        # Cache subscription data
                        sub_key = CACHE_KEY_SUBSCRIPTION.format(
                            subscription_id=subscription.id
                        )
                        pipe.set_key(
                            sub_key,
                            subscription.model_dump_json(),
                            ttl=config.cache_ttl_seconds,
                        )

                        # Add to topic->subscribers mapping
                        topic_key = CACHE_KEY_TOPIC_SUBSCRIBERS.format(
                            topic=subscription.topic
                        )
                        pipe.sadd(topic_key, subscription.id)
                        pipe.expire(topic_key, config.cache_ttl_seconds)

                        # Add to agent->subscriptions mapping
                        agent_key = CACHE_KEY_AGENT_SUBSCRIPTIONS.format(
                            agent_id=subscription.agent_id
                        )
                        pipe.sadd(agent_key, subscription.id)
                        pipe.expire(agent_key, config.cache_ttl_seconds)

                processed += len(rows)
                offset += config.cache_rebuild_batch_size
                logger.info(
                    "Cache rebuild progress: %d/%d subscriptions",
                    processed,
                    total_count,
                )

            logger.info("Valkey cache rebuilt with %d subscriptions", processed)

    async def _get_subscribers_for_topic(self, topic: str) -> set[str]:
        """Get subscriber IDs for a topic.

        Tries Valkey first, falls back to Postgres. Valkey failures are
        handled gracefully with automatic DB fallback.

        Args:
            topic: The topic.

        Returns:
            Set of subscription IDs.
        """
        valkey, db_handler, _, config = self._ensure_initialized()

        # Try cache first (best-effort - DB is authoritative)
        topic_key = CACHE_KEY_TOPIC_SUBSCRIBERS.format(topic=topic)
        try:
            subscriber_ids = await valkey.smembers(topic_key)

            if subscriber_ids:
                # Refresh TTL on cache hit to prevent expiry during active usage
                try:
                    await valkey.expire(topic_key, config.cache_ttl_seconds)
                except Exception as e:
                    logger.debug(
                        "Failed to refresh cache TTL for topic %s: %s", topic, e
                    )
                return subscriber_ids
        except Exception as e:
            logger.warning(
                "Valkey smembers failed for topic %s, falling back to database: %s",
                topic,
                e,
            )

        # Cache miss or failure - log for monitoring cache effectiveness
        logger.info(
            "Cache miss for topic subscribers: %s, falling back to database",
            topic,
        )

        # Fallback to database (authoritative source)
        sql = """
            SELECT id FROM subscriptions
            WHERE topic = $1 AND status = $2
        """
        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": sql,
                "parameters": [topic, EnumSubscriptionStatus.ACTIVE.value],
            },
        }
        result = await db_handler.execute(envelope)

        rows = result.result.get("payload", {}).get("rows", [])
        subscription_ids = {str(row["id"]) for row in rows}

        # Rebuild cache for this topic (best-effort - don't fail if cache write fails)
        if subscription_ids:
            try:
                async with valkey.pipeline() as pipe:
                    pipe.sadd(topic_key, *subscription_ids)
                    pipe.expire(topic_key, config.cache_ttl_seconds)
            except Exception as e:
                logger.warning(
                    "Failed to rebuild cache for topic %s (DB query succeeded): %s",
                    topic,
                    e,
                )

        return subscription_ids

    def _build_batch_select_query(
        self,
        table: str,
        columns: list[str],
        id_column: str,
        id_count: int,
    ) -> str:
        """Build a batch SELECT query with parameterized IN clause.

        This helper encapsulates the safe f-string pattern for batch queries.
        The placeholders are generated by _sql_placeholders() which only produces
        $1, $2, ... strings (never user data), making this SQL injection safe.

        Security Note:
            The table name, columns, and id_column parameters MUST be hardcoded
            string literals from the codebase, NOT user input. This method does
            not validate these parameters against SQL injection because they are
            expected to be trusted values from the calling code. The actual query
            values are always passed via parameterized placeholders.

        Args:
            table: Table name (must be a known table, not user input).
            columns: List of column names to select.
            id_column: Column name for the IN clause.
            id_count: Number of IDs in the batch.

        Returns:
            SQL query string with parameterized placeholders.

        Example:
            >>> handler._build_batch_select_query(
            ...     table="subscriptions",
            ...     columns=["id", "agent_id", "topic"],
            ...     id_column="id",
            ...     id_count=3,
            ... )
            'SELECT id, agent_id, topic FROM subscriptions WHERE id IN ($1, $2, $3)'
        """
        placeholders = _sql_placeholders(id_count)
        columns_str = ", ".join(columns)
        return (
            f"SELECT {columns_str} FROM {table} WHERE {id_column} IN ({placeholders})"  # noqa: S608  # nosec B608
        )

    async def _load_subscriptions(
        self,
        subscription_ids: set[str],
    ) -> list[ModelSubscription]:
        """Load subscription details from cache or database.

        Tries Valkey cache first for performance, falls back to database
        if cache fails. Cache writes after DB fallback are best-effort.

        Args:
            subscription_ids: Set of subscription IDs to load.

        Returns:
            List of subscriptions.
        """
        valkey, db_handler, _, config = self._ensure_initialized()

        subscriptions: list[ModelSubscription] = []
        missing_ids: list[str] = list(subscription_ids)

        # Try cache first using batch retrieval (best-effort - DB is authoritative)
        sub_id_list = list(subscription_ids)
        if sub_id_list:
            try:
                cache_keys = [
                    CACHE_KEY_SUBSCRIPTION.format(subscription_id=sub_id)
                    for sub_id in sub_id_list
                ]
                cached_values = await valkey.mget(*cache_keys)

                # Reset missing_ids since we successfully read from cache
                missing_ids = []

                for sub_id, cached in zip(sub_id_list, cached_values, strict=True):
                    if cached:
                        try:
                            subscription = ModelSubscription.model_validate_json(cached)
                            subscriptions.append(subscription)
                        except Exception as e:
                            logger.warning(
                                "Failed to parse cached subscription %s: %s", sub_id, e
                            )
                            missing_ids.append(sub_id)
                    else:
                        missing_ids.append(sub_id)

                # Refresh TTL on successfully loaded cache entries to prevent
                # expiry during active usage (consistent with _get_subscribers_for_topic)
                if subscriptions:
                    try:
                        async with valkey.pipeline() as pipe:
                            for sub in subscriptions:
                                sub_key = CACHE_KEY_SUBSCRIPTION.format(
                                    subscription_id=sub.id
                                )
                                pipe.expire(sub_key, config.cache_ttl_seconds)
                    except Exception as e:
                        logger.debug("Failed to refresh cache TTL: %s", e)

            except Exception as e:
                logger.warning(
                    "Valkey mget failed for subscriptions, falling back to database: %s",
                    e,
                )
                # All IDs need to be loaded from DB
                missing_ids = sub_id_list

        # Load missing from database
        if missing_ids:
            placeholders = _sql_placeholders(len(missing_ids))
            # Security: Safe f-string usage - _sql_placeholders() only generates
            # parameterized placeholders ($1, $2, ...), not user data. Actual values
            # are passed via the parameters list below (parameterized query).  # nosec B608
            sql = f"""
                SELECT id, agent_id, topic, status,
                       created_at, updated_at, metadata
                FROM subscriptions
                WHERE id IN ({placeholders})
            """
            envelope = {
                "operation": "db.query",
                "payload": {
                    "sql": sql,
                    "parameters": missing_ids,
                },
            }
            result = await db_handler.execute(envelope)

            rows = result.result.get("payload", {}).get("rows", [])
            for row in rows:
                subscription = self._row_to_subscription(row)
                subscriptions.append(subscription)

                # Update cache (best-effort - don't fail if cache write fails)
                try:
                    sub_key = CACHE_KEY_SUBSCRIPTION.format(
                        subscription_id=subscription.id
                    )
                    await valkey.set_key(
                        sub_key,
                        subscription.model_dump_json(),
                        ttl=config.cache_ttl_seconds,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to cache subscription %s (DB query succeeded): %s",
                        subscription.id,
                        e,
                    )

        return subscriptions

    # =========================================================================
    # Metrics
    # =========================================================================

    async def _increment_metric(self, key: str, amount: int = 1) -> None:
        """Thread-safe metric increment.

        Args:
            key: The metric key to increment (must be a valid ModelSubscriptionMetrics field).
            amount: The amount to increment by (default 1).

        Raises:
            AttributeError: If key is not a valid metric field.
        """
        async with self._metrics_lock:
            current = getattr(self._metrics, key)
            setattr(self._metrics, key, current + amount)

    async def get_metrics(self) -> ModelSubscriptionMetrics:
        """Get handler metrics for observability.

        Returns a copy of current metrics as a Pydantic model for production
        monitoring and alerting. Uses async lock for thread-safe access.

        Returns:
            ModelSubscriptionMetrics copy with current counter values.
        """
        async with self._metrics_lock:
            return self._metrics.model_copy()

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> ModelSubscriptionHealth:
        """Check if all handler components are healthy.

        Performs health checks on Valkey, database, and Kafka handlers.
        Component health is reported as:
        - True: Component is healthy and responding
        - False: Component is unhealthy or failed health check

        The overall is_healthy is True when ALL components are healthy.

        Returns:
            ModelSubscriptionHealth with detailed status for each component.
        """
        if not self._initialized:
            return ModelSubscriptionHealth(
                is_healthy=False,
                initialized=False,
                error_message="Handler not initialized",
            )

        errors: list[str] = []

        # Check Valkey
        valkey_healthy = False
        if self._valkey:
            try:
                health = await self._valkey.health_check()
                valkey_healthy = health.is_healthy
                if not valkey_healthy:
                    errors.append(f"Valkey: {health.error_message}")
            except Exception as e:
                errors.append(f"Valkey check failed: {e}")

        # Check DB
        db_healthy = False
        if self._db_handler:
            try:
                envelope = {
                    "operation": "db.query",
                    "payload": {
                        "sql": "SELECT 1",
                        "parameters": [],
                    },
                }
                await self._db_handler.execute(envelope)
                db_healthy = True
            except Exception as e:
                errors.append(f"Database check failed: {e}")

        # Check Kafka with robust type handling
        # EventBusKafka.health_check() may return:
        #   - dict: {"healthy": bool, "circuit_state": str, ...}
        #   - bool: Direct healthy status
        #   - Pydantic model: Object with is_healthy attribute
        kafka_healthy = False
        if self._kafka_handler:
            try:
                health_result = await self._kafka_handler.health_check()
                # Handle different return types from EventBusKafka
                if isinstance(health_result, dict):
                    kafka_healthy = bool(health_result.get("healthy", False))
                    if not kafka_healthy:
                        circuit_state = health_result.get("circuit_state", "unknown")
                        errors.append(
                            f"Kafka: unhealthy (circuit_state={circuit_state})"
                        )
                elif isinstance(health_result, bool):
                    kafka_healthy = health_result
                    if not kafka_healthy:
                        errors.append("Kafka: unhealthy (returned False)")
                elif hasattr(health_result, "is_healthy"):
                    # Handle Pydantic model response (e.g., ModelHealthStatus)
                    kafka_healthy = bool(health_result.is_healthy)
                    if not kafka_healthy:
                        error_detail = getattr(health_result, "error", None) or getattr(
                            health_result, "error_message", "unknown"
                        )
                        errors.append(f"Kafka: unhealthy ({error_detail})")
                else:
                    # Unexpected return type - log warning and treat as unhealthy
                    logger.warning(
                        "Unexpected Kafka health_check return type: %s, treating as unhealthy",
                        type(health_result).__name__,
                    )
                    errors.append(
                        f"Kafka: unexpected health_check return type "
                        f"({type(health_result).__name__})"
                    )
            except Exception as e:
                errors.append(f"Kafka check failed: {e}")

        # Only fully healthy if all components are explicitly True
        is_healthy = (
            valkey_healthy is True and db_healthy is True and kafka_healthy is True
        )

        return ModelSubscriptionHealth(
            is_healthy=is_healthy,
            initialized=True,
            db_healthy=db_healthy,
            valkey_healthy=valkey_healthy,
            kafka_healthy=kafka_healthy,
            error_message="; ".join(errors) if errors else None,
            metrics=await self.get_metrics(),
        )

    async def describe(self) -> ModelSubscriptionMetadata:
        """Return handler metadata and capabilities.

        Provides introspection information about the handler's type,
        supported operations, and configuration.

        Returns:
            ModelSubscriptionMetadata with handler information.

        Note:
            This method is async per ONEX protocol specification.
        """
        capabilities = [
            "subscribe",
            "unsubscribe",
            "list_subscriptions",
            "notify",
            "pagination",
            "metrics",
            "health_check",
            "kafka_delivery",
            "valkey_caching",
            "postgresql_persistence",
        ]

        return ModelSubscriptionMetadata(
            handler_type=self.handler_type,
            capabilities=capabilities,
            supports_transactions=False,  # Operations are not transactional across services
        )
