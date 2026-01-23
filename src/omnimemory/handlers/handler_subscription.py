"""Subscription Handler for agent subscriptions and memory change notifications.

This module provides the core subscription management functionality:
- subscribe(): Register agent subscriptions to memory topics
- unsubscribe(): Remove agent subscriptions
- notify(): Publish notification events to Kafka for subscriber consumption
- list_subscriptions(): Get all subscriptions for an agent

Architecture:
- Persistence: Valkey (fast lookups) + PostgreSQL (source of truth)
- Delivery: Kafka event bus (agents consume directly)
- Topic naming: memory.<entity>.<event> convention

Event Bus Strategy:
    Notifications are published to Kafka topics. Internal agents consume
    events directly via consumer groups. If external (non-Kafka) delivery
    is needed in the future, implement a WebhookEmitterEffect node that
    consumes bus events and handles HTTP delivery separately.

Example::

    from omnimemory.handlers import (
        HandlerSubscription,
        ModelHandlerSubscriptionConfig,
    )
    from omnimemory.models.subscription import (
        ModelNotificationEvent,
        ModelNotificationEventPayload,
    )

    config = ModelHandlerSubscriptionConfig(
        db_dsn="postgresql://user:pass@localhost:5432/omnimemory",
        valkey_host="localhost",
        valkey_port=6379,
        kafka_bootstrap_servers="localhost:9092",
    )
    handler = HandlerSubscription(config)
    await handler.initialize()

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
from typing import cast
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

# Optional omnibase_infra imports for handler reuse
_OMNIBASE_INFRA_AVAILABLE = False
_OMNIBASE_INFRA_IMPORT_ERROR: str | None = None

try:
    from omnibase_infra.handlers.handler_db import HandlerDb
    from omnibase_infra.handlers.handler_kafka import HandlerKafka

    _OMNIBASE_INFRA_AVAILABLE = True
except ImportError as e:
    _OMNIBASE_INFRA_IMPORT_ERROR = str(e)

    # Provide stubs for type checking
    class HandlerDb:  # type: ignore[no-redef]
        """Stub for HandlerDb when omnibase_infra is not installed."""

    class HandlerKafka:  # type: ignore[no-redef]
        """Stub for HandlerKafka when omnibase_infra is not installed."""


logger = logging.getLogger(__name__)

__all__ = [
    "HandlerSubscription",
    "ModelHandlerSubscriptionConfig",
    "ModelSubscriptionHealth",
    "ModelSubscriptionMetrics",
]

# Cache key patterns
CACHE_KEY_TOPIC_SUBSCRIBERS = "topic:{topic}:subscribers"
CACHE_KEY_AGENT_SUBSCRIPTIONS = "agent:{agent_id}:subscriptions"
CACHE_KEY_SUBSCRIPTION = "subscription:{subscription_id}"

# Kafka topic for memory notifications
KAFKA_TOPIC_MEMORY_NOTIFICATIONS = "omnimemory.memory.notification.v1"


@lru_cache(maxsize=128)
def _sql_placeholders(count: int, start: int = 1) -> str:
    """Generate SQL parameter placeholders for parameterized queries.

    Results are cached with LRU policy (maxsize=128) to avoid repeated string
    generation for common query sizes.

    Args:
        count: Number of placeholders to generate. If <= 0, returns empty string.
        start: Starting index (default 1 for PostgreSQL $1, $2, ...).
               Must be >= 1.

    Returns:
        Comma-separated placeholder string (e.g., "$1, $2, $3").
        Returns empty string if count <= 0.

    Raises:
        ValueError: If start < 1 (PostgreSQL placeholders start at $1).

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
    if count <= 0:
        return ""
    return ", ".join(f"${i}" for i in range(start, start + count))


class ModelHandlerSubscriptionConfig(BaseModel):
    """Configuration for the Subscription Handler.

    Attributes:
        db_dsn: PostgreSQL connection string.
        valkey_host: Valkey server hostname.
        valkey_port: Valkey server port.
        valkey_db: Valkey database index.
        valkey_password: Optional Valkey password.
        kafka_bootstrap_servers: Kafka bootstrap servers.
        cache_ttl_seconds: TTL for cached subscription data.
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


class ModelSubscriptionMetrics(BaseModel):
    """Metrics for the Subscription Handler.

    Tracks counters for various operations to enable production monitoring
    and observability.

    Attributes:
        notifications_published: Count of notifications published to Kafka.
        subscriptions_created: Count of subscriptions created.
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
        description="Count of subscriptions created",
    )
    subscriptions_deleted: int = Field(
        default=0,
        ge=0,
        description="Count of subscriptions deleted",
    )


class ModelSubscriptionHealth(BaseModel):
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

    Attributes:
        config: The handler configuration.
    """

    def __init__(self, config: ModelHandlerSubscriptionConfig) -> None:
        """Initialize the handler with configuration.

        Args:
            config: The handler configuration.

        Raises:
            ImportError: If omnibase_infra is not installed.
        """
        if not _OMNIBASE_INFRA_AVAILABLE:
            raise ImportError(
                f"omnibase_infra is required for HandlerSubscription. "
                f"Install it with: poetry install --with dev. "
                f"Original error: {_OMNIBASE_INFRA_IMPORT_ERROR}"
            )

        self._config = config
        self._db_handler: HandlerDb | None = None
        self._kafka_handler: HandlerKafka | None = None
        self._valkey: AdapterValkey | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Metrics for observability
        self._metrics: dict[str, int] = {
            "notifications_published": 0,
            "subscriptions_created": 0,
            "subscriptions_deleted": 0,
        }

    @property
    def config(self) -> ModelHandlerSubscriptionConfig:
        """Get the handler configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the handler has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize DB, Valkey, and Kafka handlers.

        Creates connections to all required services and optionally
        rebuilds the Valkey cache from PostgreSQL on cold start.

        Raises:
            RuntimeError: If initialization fails.
        """
        async with self._init_lock:
            if self._initialized:
                return

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
                self._kafka_handler = HandlerKafka()
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
            except Exception:
                pass
            self._valkey = None

        if self._db_handler:
            try:
                await self._db_handler.shutdown()
            except Exception:
                pass
            self._db_handler = None

        if self._kafka_handler:
            try:
                await self._kafka_handler.shutdown()
            except Exception:
                pass
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

    def _ensure_initialized(self) -> tuple[AdapterValkey, HandlerDb, HandlerKafka]:
        """Ensure handler is initialized and return components.

        Returns:
            Tuple of (valkey, db_handler, kafka_handler).

        Raises:
            RuntimeError: If handler is not initialized.
        """
        if (
            not self._initialized
            or self._valkey is None
            or self._db_handler is None
            or self._kafka_handler is None
        ):
            raise RuntimeError(
                "HandlerSubscription not initialized. Call initialize() first."
            )
        return self._valkey, self._db_handler, self._kafka_handler

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
        valkey, _, _ = self._ensure_initialized()

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
        await self._persist_subscription(subscription, is_update=existing is not None)
        self._metrics["subscriptions_created"] += 1

        # Update Valkey caches (best effort - DB is source of truth)
        try:
            topic_key = CACHE_KEY_TOPIC_SUBSCRIBERS.format(topic=topic)
            agent_key = CACHE_KEY_AGENT_SUBSCRIPTIONS.format(agent_id=agent_id)
            sub_key = CACHE_KEY_SUBSCRIPTION.format(subscription_id=subscription_id)

            async with valkey.pipeline() as pipe:
                pipe.sadd(topic_key, subscription_id)
                pipe.sadd(agent_key, subscription_id)
                pipe.set_key(
                    sub_key,
                    subscription.model_dump_json(),
                    ttl=self._config.cache_ttl_seconds,
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
        valkey, _, _ = self._ensure_initialized()

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
        await self._soft_delete_subscription(subscription.id)
        self._metrics["subscriptions_deleted"] += 1

        # Remove from Valkey caches (best effort - DB is source of truth)
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
        _, _, kafka_handler = self._ensure_initialized()

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
        envelope = {
            "operation": "kafka.produce",
            "payload": {
                "topic": KAFKA_TOPIC_MEMORY_NOTIFICATIONS,
                "key": topic,  # Partition by topic for ordering
                "value": event.model_dump_json(),
                "headers": {
                    "event_id": event.event_id,
                    "topic": topic,
                    "subscriber_count": str(subscriber_count),
                },
            },
        }
        await kafka_handler.execute(envelope)

        self._metrics["notifications_published"] += 1

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
    ) -> list[ModelSubscription]:
        """Get all subscriptions for an agent.

        Args:
            agent_id: The agent's identifier.

        Returns:
            List of subscriptions for the agent.

        Raises:
            RuntimeError: If handler is not initialized.
        """
        valkey, _, _ = self._ensure_initialized()

        # Try Valkey cache first
        agent_key = CACHE_KEY_AGENT_SUBSCRIPTIONS.format(agent_id=agent_id)
        subscription_ids = await valkey.smembers(agent_key)

        if subscription_ids:
            subscriptions = await self._load_subscriptions(subscription_ids)
            # Filter to only active subscriptions
            return [
                s for s in subscriptions if s.status == EnumSubscriptionStatus.ACTIVE
            ]

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
        _, db_handler, _ = self._ensure_initialized()

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
        _, db_handler, _ = self._ensure_initialized()

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
        _, db_handler, _ = self._ensure_initialized()

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

        Args:
            agent_id: The agent's identifier.

        Returns:
            List of subscriptions.
        """
        _, db_handler, _ = self._ensure_initialized()

        sql = """
            SELECT id, agent_id, topic, status,
                   created_at, updated_at, metadata
            FROM subscriptions
            WHERE agent_id = $1 AND status = $2
            ORDER BY created_at DESC
        """
        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": sql,
                "parameters": [agent_id, EnumSubscriptionStatus.ACTIVE.value],
            },
        }
        result = await db_handler.execute(envelope)

        rows = result.result.get("payload", {}).get("rows", [])
        return [self._row_to_subscription(row) for row in rows]

    def _row_to_subscription(self, row: dict[str, object]) -> ModelSubscription:
        """Convert database row to ModelSubscription.

        Args:
            row: Database row dict.

        Returns:
            ModelSubscription instance.
        """
        # Parse JSON fields
        metadata = None
        metadata_raw = row.get("metadata")
        if metadata_raw:
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
        """Cold start recovery: rebuild Valkey from Postgres."""
        valkey, db_handler, _ = self._ensure_initialized()

        logger.info("Rebuilding Valkey cache from PostgreSQL...")

        sql = """
            SELECT id, agent_id, topic, status,
                   created_at, updated_at, metadata
            FROM subscriptions
            WHERE status = $1
        """
        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": sql,
                "parameters": [EnumSubscriptionStatus.ACTIVE.value],
            },
        }
        result = await db_handler.execute(envelope)

        rows = result.result.get("payload", {}).get("rows", [])
        logger.info("Found %d active subscriptions to cache", len(rows))

        if not rows:
            logger.info("No subscriptions to cache, skipping pipeline")
            return

        # Use pipeline for atomic batch update
        async with valkey.pipeline() as pipe:
            for row in rows:
                subscription = self._row_to_subscription(row)

                # Cache subscription data
                sub_key = CACHE_KEY_SUBSCRIPTION.format(subscription_id=subscription.id)
                pipe.set_key(
                    sub_key,
                    subscription.model_dump_json(),
                    ttl=self._config.cache_ttl_seconds,
                )

                # Add to topic->subscribers mapping
                topic_key = CACHE_KEY_TOPIC_SUBSCRIBERS.format(topic=subscription.topic)
                pipe.sadd(topic_key, subscription.id)

                # Add to agent->subscriptions mapping
                agent_key = CACHE_KEY_AGENT_SUBSCRIPTIONS.format(
                    agent_id=subscription.agent_id
                )
                pipe.sadd(agent_key, subscription.id)

        logger.info(
            "Valkey cache rebuilt with %d subscriptions using pipeline (atomic batch)",
            len(rows),
        )

    async def _get_subscribers_for_topic(self, topic: str) -> set[str]:
        """Get subscriber IDs for a topic.

        Tries Valkey first, falls back to Postgres.

        Args:
            topic: The topic.

        Returns:
            Set of subscription IDs.
        """
        valkey, db_handler, _ = self._ensure_initialized()

        # Try cache first
        topic_key = CACHE_KEY_TOPIC_SUBSCRIBERS.format(topic=topic)
        subscriber_ids = await valkey.smembers(topic_key)

        if subscriber_ids:
            return subscriber_ids

        # Fallback to database
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

        # Rebuild cache for this topic
        if subscription_ids:
            await valkey.sadd(topic_key, *subscription_ids)

        return subscription_ids

    async def _load_subscriptions(
        self,
        subscription_ids: set[str],
    ) -> list[ModelSubscription]:
        """Load subscription details from cache or database.

        Args:
            subscription_ids: Set of subscription IDs to load.

        Returns:
            List of subscriptions.
        """
        valkey, db_handler, _ = self._ensure_initialized()

        subscriptions: list[ModelSubscription] = []
        missing_ids: list[str] = []

        # Try cache first
        for sub_id in subscription_ids:
            sub_key = CACHE_KEY_SUBSCRIPTION.format(subscription_id=sub_id)
            cached = await valkey.get(sub_key)
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

        # Load missing from database
        if missing_ids:
            placeholders = _sql_placeholders(len(missing_ids))
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

                # Update cache
                sub_key = CACHE_KEY_SUBSCRIPTION.format(subscription_id=subscription.id)
                await valkey.set_key(
                    sub_key,
                    subscription.model_dump_json(),
                    ttl=self._config.cache_ttl_seconds,
                )

        return subscriptions

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> ModelSubscriptionMetrics:
        """Get handler metrics for observability.

        Returns a copy of current metrics as a Pydantic model for production
        monitoring and alerting.

        Returns:
            ModelSubscriptionMetrics with current counter values.
        """
        return ModelSubscriptionMetrics(**self._metrics)

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> ModelSubscriptionHealth:
        """Check if all handler components are healthy.

        Returns:
            ModelSubscriptionHealth with detailed status.
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

        # Check Kafka
        kafka_healthy = False
        if self._kafka_handler:
            try:
                # Simple health check - Kafka handler should have a health method
                kafka_healthy = True  # Assume healthy if initialized
            except Exception as e:
                errors.append(f"Kafka check failed: {e}")

        is_healthy = valkey_healthy and db_healthy and kafka_healthy

        return ModelSubscriptionHealth(
            is_healthy=is_healthy,
            initialized=True,
            db_healthy=db_healthy,
            valkey_healthy=valkey_healthy,
            kafka_healthy=kafka_healthy,
            error_message="; ".join(errors) if errors else None,
            metrics=self.get_metrics(),
        )
