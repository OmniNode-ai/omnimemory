# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Subscription Handler for agent subscriptions and memory change notifications.

This module provides the core subscription management functionality:
- subscribe(): Register agent subscriptions to memory topics
- unsubscribe(): Remove agent subscriptions
- notify(): Send notifications to all subscribers of a topic
- list_subscriptions(): Get all subscriptions for an agent

Architecture:
- Persistence: Valkey (fast lookups) + PostgreSQL (source of truth)
- Delivery: Webhook only (HTTP POST with retry/DLQ)
- Topic naming: memory.<entity>.<event> convention

Note on Retry Worker:
    This handler records retry schedules (``next_retry_at`` field) and persists
    failed delivery attempts to the ``delivery_attempts`` table, but it does NOT
    implement the background retry worker. A separate component (e.g., a scheduled
    task, cron job, or Kafka consumer) is responsible for polling the
    ``delivery_attempts`` table for rows where ``status = 'failed'`` and
    ``next_retry_at <= NOW()``, then re-invoking delivery. This separation follows
    the single-responsibility principle: the handler manages subscriptions and
    immediate delivery attempts, while retry orchestration is a distinct concern.

Example::

    from omnimemory.handlers import (
        HandlerSubscription,
        ModelHandlerSubscriptionConfig,
    )
    from omnimemory.models.subscription import (
        ModelSubscriptionDeliveryWebhook,
        ModelNotificationEvent,
        ModelNotificationEventPayload,
    )

    config = ModelHandlerSubscriptionConfig(
        db_dsn="postgresql://user:pass@localhost:5432/omnimemory",
        valkey_host="localhost",
        valkey_port=6379,
    )
    handler = HandlerSubscription(config)
    await handler.initialize()

    # Subscribe an agent
    delivery = ModelSubscriptionDeliveryWebhook(
        webhook_url="https://agent.example.com/webhook",
        secret="my-hmac-secret",
    )
    subscription = await handler.subscribe(
        agent_id="agent_123",
        topic="memory.item.created",
        delivery=delivery,
    )

    # Notify all subscribers
    event = ModelNotificationEvent(
        event_id="evt_456",
        topic="memory.item.created",
        payload=ModelNotificationEventPayload(
            entity_type="item",
            entity_id="item_789",
            action="created",
        ),
    )
    attempts = await handler.notify("memory.item.created", event)

    await handler.shutdown()

.. versionadded:: 0.1.0
    Initial implementation for OMN-1393.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr

from omnimemory.enums.enum_subscription_status import (
    EnumCircuitBreakerState,
    EnumDeliveryStatus,
    EnumSubscriptionStatus,
)
from omnimemory.handlers.adapters.adapter_valkey import (
    AdapterValkey,
    AdapterValkeyConfig,
)
from omnimemory.models.subscription import (
    ModelNotificationDeliveryAttempt,
    ModelNotificationEvent,
    ModelSubscription,
    ModelSubscriptionDeliveryWebhook,
)
from omnimemory.models.subscription.constants import (
    DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
    TOPIC_PATTERN,
)

# Optional omnibase_infra imports for handler reuse
_OMNIBASE_INFRA_AVAILABLE = False
_OMNIBASE_INFRA_IMPORT_ERROR: str | None = None

try:
    from omnibase_infra.handlers.handler_db import HandlerDb
    from omnibase_infra.handlers.handler_http import HandlerHttpRest

    _OMNIBASE_INFRA_AVAILABLE = True
except ImportError as e:
    _OMNIBASE_INFRA_IMPORT_ERROR = str(e)

    # Provide stubs for type checking
    class HandlerDb:  # type: ignore[no-redef]
        """Stub for HandlerDb when omnibase_infra is not installed."""

    class HandlerHttpRest:  # type: ignore[no-redef]
        """Stub for HandlerHttpRest when omnibase_infra is not installed."""


logger = logging.getLogger(__name__)

__all__ = [
    "HandlerSubscription",
    "ModelHandlerSubscriptionConfig",
    "ModelSubscriptionHealth",
]

# Cache key patterns
CACHE_KEY_TOPIC_SUBSCRIBERS = "topic:{topic}:subscribers"
CACHE_KEY_AGENT_SUBSCRIPTIONS = "agent:{agent_id}:subscriptions"
CACHE_KEY_SUBSCRIPTION = "subscription:{subscription_id}"
CACHE_KEY_CIRCUIT_BREAKER = "circuit_breaker:{endpoint_hash}"

# Default retry configuration
DEFAULT_MAX_RETRY_ATTEMPTS = 5
DEFAULT_RETRY_BASE_DELAY_MS = 1000
DEFAULT_RETRY_MAX_DELAY_MS = 60000  # 1 minute cap

# Circuit breaker cache TTL in seconds.
# TTL Refresh Strategy: This TTL is refreshed on every state update to prevent state
# loss for long-lived circuits. If a circuit stays open/half-open beyond this TTL
# without any updates, the state would be lost and the circuit would reset to closed.
# By refreshing on every update, we ensure state persists as long as the circuit is
# actively being used. PostgreSQL serves as durable backup if cache expires.
CIRCUIT_BREAKER_CACHE_TTL_SECONDS = 3600  # 1 hour


def _sql_placeholders(count: int, start: int = 1) -> str:
    """Generate SQL parameter placeholders for parameterized queries.

    Args:
        count: Number of placeholders to generate.
        start: Starting index (default 1 for PostgreSQL $1, $2, ...).

    Returns:
        Comma-separated placeholder string (e.g., "$1, $2, $3").

    Example:
        >>> _sql_placeholders(3)
        '$1, $2, $3'
        >>> _sql_placeholders(2, start=5)
        '$5, $6'
    """
    return ", ".join(f"${i}" for i in range(start, start + count))


class ModelHandlerSubscriptionConfig(BaseModel):
    """Configuration for the Subscription Handler.

    Attributes:
        db_dsn: PostgreSQL connection string.
        valkey_host: Valkey server hostname.
        valkey_port: Valkey server port.
        valkey_db: Valkey database index.
        valkey_password: Optional Valkey password.
        max_retry_attempts: Maximum delivery retry attempts before DLQ.
        retry_base_delay_ms: Base delay for exponential backoff.
        retry_max_delay_ms: Maximum delay cap for retries.
        circuit_breaker_threshold: Failures before circuit opens.
        circuit_breaker_success_threshold: Successes in half_open before closing.
        circuit_breaker_cooldown_seconds: Time before half-open transition.
        http_timeout_seconds: Webhook delivery timeout.
        cache_ttl_seconds: TTL for cached subscription data.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
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
    max_retry_attempts: int = Field(
        default=DEFAULT_MAX_RETRY_ATTEMPTS,
        ge=1,
        le=10,
        description="Maximum delivery retry attempts",
    )
    retry_base_delay_ms: int = Field(
        default=DEFAULT_RETRY_BASE_DELAY_MS,
        ge=100,
        le=10000,
        description="Base delay for exponential backoff (ms)",
    )
    retry_max_delay_ms: int = Field(
        default=DEFAULT_RETRY_MAX_DELAY_MS,
        ge=1000,
        le=300000,
        description="Maximum delay cap for retries (ms)",
    )
    circuit_breaker_threshold: int = Field(
        default=DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        ge=1,
        le=100,
        description="Consecutive failures before circuit opens",
    )
    circuit_breaker_success_threshold: int = Field(
        default=DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
        ge=1,
        le=100,
        description="Number of consecutive successes in half_open state before closing circuit",
    )
    circuit_breaker_cooldown_seconds: int = Field(
        default=DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
        ge=10,
        le=600,
        description="Seconds before circuit transitions to half-open",
    )
    http_timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        le=30.0,
        description="Webhook delivery timeout in seconds",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="TTL for cached subscription data",
    )


class ModelSubscriptionHealth(BaseModel):
    """Health status for the Subscription Handler.

    Attributes:
        is_healthy: Overall health status.
        initialized: Whether the handler has been initialized.
        db_healthy: Database connection health.
        valkey_healthy: Valkey connection health.
        http_healthy: HTTP handler health.
        error_message: Error details if unhealthy.
    """

    model_config = ConfigDict(
        extra="forbid",
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
    http_healthy: bool | None = Field(
        default=None,
        description="HTTP handler health",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if unhealthy",
    )


class HandlerSubscription:
    """Handler for agent subscriptions and memory change notifications.

    Manages the lifecycle of subscriptions and handles notification delivery
    with retry logic, circuit breakers, and dead letter queue support.

    Persistence Strategy:
        - Valkey: Fast lookups for topic->subscribers mapping
        - PostgreSQL: Source of truth for subscription data and delivery history

    Delivery Features:
        - Webhook delivery with HMAC signature verification
        - Exponential backoff retry with configurable attempts
        - Circuit breaker per endpoint to prevent cascade failures
        - Dead letter queue for failed deliveries

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
        self._http_handler: HandlerHttpRest | None = None
        self._valkey: AdapterValkey | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def config(self) -> ModelHandlerSubscriptionConfig:
        """Get the handler configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if the handler has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize DB, Valkey, and HTTP handlers.

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

                # Initialize HTTP handler
                self._http_handler = HandlerHttpRest()
                await self._http_handler.initialize(
                    {
                        "timeout": self._config.http_timeout_seconds,
                    }
                )
                logger.info("HTTP handler initialized")

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

        if self._http_handler:
            try:
                await self._http_handler.shutdown()
            except Exception:
                pass
            self._http_handler = None

    async def shutdown(self) -> None:
        """Cleanup all resources."""
        if self._valkey:
            await self._valkey.shutdown()
            self._valkey = None

        if self._db_handler:
            await self._db_handler.shutdown()
            self._db_handler = None

        if self._http_handler:
            await self._http_handler.shutdown()
            self._http_handler = None

        self._initialized = False
        logger.info("HandlerSubscription shutdown complete")

    def _ensure_initialized(self) -> tuple[AdapterValkey, HandlerDb, HandlerHttpRest]:
        """Ensure handler is initialized and return components.

        Returns:
            Tuple of (valkey, db_handler, http_handler).

        Raises:
            RuntimeError: If handler is not initialized.
        """
        if (
            not self._initialized
            or self._valkey is None
            or self._db_handler is None
            or self._http_handler is None
        ):
            raise RuntimeError(
                "HandlerSubscription not initialized. Call initialize() first."
            )
        return self._valkey, self._db_handler, self._http_handler

    # =========================================================================
    # Core Operations
    # =========================================================================

    async def subscribe(
        self,
        agent_id: str,
        topic: str,
        delivery: ModelSubscriptionDeliveryWebhook,
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
            delivery: Webhook delivery configuration.
            metadata: Optional subscription metadata.

        Returns:
            The created or updated subscription.

        Raises:
            ValueError: If topic format is invalid.
            RuntimeError: If handler is not initialized.
        """
        valkey, _, _ = self._ensure_initialized()

        # Validate topic format
        if not TOPIC_PATTERN.match(topic):
            raise ValueError(
                f"Topic must match pattern 'memory.<entity>.<event>', got: {topic}"
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
            delivery=delivery,
            status=EnumSubscriptionStatus.ACTIVE,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            metadata=metadata,
        )

        # Persist to PostgreSQL (source of truth)
        await self._persist_subscription(subscription, is_update=existing is not None)

        # Update Valkey caches (best effort - DB is source of truth)
        try:
            topic_key = CACHE_KEY_TOPIC_SUBSCRIBERS.format(topic=topic)
            agent_key = CACHE_KEY_AGENT_SUBSCRIPTIONS.format(agent_id=agent_id)
            sub_key = CACHE_KEY_SUBSCRIPTION.format(subscription_id=subscription_id)

            await valkey.sadd(topic_key, subscription_id)
            await valkey.sadd(agent_key, subscription_id)
            await valkey.set(
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

        # Remove from Valkey caches
        topic_key = CACHE_KEY_TOPIC_SUBSCRIBERS.format(topic=topic)
        agent_key = CACHE_KEY_AGENT_SUBSCRIPTIONS.format(agent_id=agent_id)
        sub_key = CACHE_KEY_SUBSCRIPTION.format(subscription_id=subscription.id)

        await valkey.srem(topic_key, subscription.id)
        await valkey.srem(agent_key, subscription.id)
        await valkey.delete(sub_key)

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
    ) -> list[ModelNotificationDeliveryAttempt]:
        """Send notification to all subscribers.

        Workflow:
            1. Get subscribers from Valkey cache (fallback to Postgres)
            2. For each active subscriber:
               a. Check circuit breaker state
               b. If closed/half-open: attempt delivery via HandlerHttpRest
               c. Record delivery attempt
               d. On failure: schedule retry with exponential backoff
            3. Return list of delivery attempts

        Args:
            topic: The topic to notify (format: memory.<entity>.<event>).
            event: The notification event to send.

        Returns:
            List of delivery attempt records.

        Raises:
            RuntimeError: If handler is not initialized.
        """
        self._ensure_initialized()

        # Get subscribers for topic
        subscriber_ids = await self._get_subscribers_for_topic(topic)
        if not subscriber_ids:
            logger.debug("No subscribers for topic %s", topic)
            return []

        # Load subscription details
        subscriptions = await self._load_subscriptions(subscriber_ids)
        active_subscriptions = [
            s for s in subscriptions if s.status == EnumSubscriptionStatus.ACTIVE
        ]

        if not active_subscriptions:
            logger.debug("No active subscribers for topic %s", topic)
            return []

        # Deliver to each subscriber
        attempts: list[ModelNotificationDeliveryAttempt] = []
        for subscription in active_subscriptions:
            attempt = await self._deliver_notification(subscription, event)
            attempts.append(attempt)

        logger.info(
            "Notification sent to %d subscribers for topic %s, event %s",
            len(attempts),
            topic,
            event.event_id,
        )

        return attempts

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
                    webhook_url = $1,
                    webhook_secret = $2,
                    webhook_headers = $3,
                    webhook_timeout_ms = $4,
                    status = $5,
                    updated_at = $6,
                    metadata = $7
                WHERE id = $8
            """
            params = [
                str(subscription.delivery.webhook_url),
                subscription.delivery.secret,
                json.dumps(subscription.delivery.headers)
                if subscription.delivery.headers
                else None,
                subscription.delivery.timeout_ms,
                subscription.status.value,
                subscription.updated_at.isoformat(),
                json.dumps(subscription.metadata) if subscription.metadata else None,
                subscription.id,
            ]
        else:
            sql = """
                INSERT INTO subscriptions (
                    id, agent_id, topic, webhook_url, webhook_secret,
                    webhook_headers, webhook_timeout_ms, status,
                    created_at, updated_at, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (agent_id, topic) DO UPDATE SET
                    webhook_url = EXCLUDED.webhook_url,
                    webhook_secret = EXCLUDED.webhook_secret,
                    webhook_headers = EXCLUDED.webhook_headers,
                    webhook_timeout_ms = EXCLUDED.webhook_timeout_ms,
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at,
                    metadata = EXCLUDED.metadata
            """
            params = [
                subscription.id,
                subscription.agent_id,
                subscription.topic,
                str(subscription.delivery.webhook_url),
                subscription.delivery.secret,
                json.dumps(subscription.delivery.headers)
                if subscription.delivery.headers
                else None,
                subscription.delivery.timeout_ms,
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
            SELECT id, agent_id, topic, webhook_url, webhook_secret,
                   webhook_headers, webhook_timeout_ms, status,
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
            SELECT id, agent_id, topic, webhook_url, webhook_secret,
                   webhook_headers, webhook_timeout_ms, status,
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
        headers = None
        webhook_headers_raw = row.get("webhook_headers")
        if webhook_headers_raw:
            headers = json.loads(str(webhook_headers_raw))

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

        # Extract webhook secret with proper typing
        webhook_secret_raw = row.get("webhook_secret")
        webhook_secret: str | None = (
            str(webhook_secret_raw) if webhook_secret_raw is not None else None
        )

        # Extract timeout with proper typing
        timeout_raw = row.get("webhook_timeout_ms")
        timeout_ms: int = int(str(timeout_raw)) if timeout_raw is not None else 5000

        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=HttpUrl(str(row["webhook_url"])),
            secret=webhook_secret,
            headers=headers,
            timeout_ms=timeout_ms,
        )

        return ModelSubscription(
            id=str(row["id"]),
            agent_id=str(row["agent_id"]),
            topic=str(row["topic"]),
            delivery=delivery,
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
            SELECT id, agent_id, topic, webhook_url, webhook_secret,
                   webhook_headers, webhook_timeout_ms, status,
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

        for row in rows:
            subscription = self._row_to_subscription(row)

            # Cache subscription data
            sub_key = CACHE_KEY_SUBSCRIPTION.format(subscription_id=subscription.id)
            await valkey.set(
                sub_key,
                subscription.model_dump_json(),
                ttl=self._config.cache_ttl_seconds,
            )

            # Add to topic->subscribers mapping
            topic_key = CACHE_KEY_TOPIC_SUBSCRIBERS.format(topic=subscription.topic)
            await valkey.sadd(topic_key, subscription.id)

            # Add to agent->subscriptions mapping
            agent_key = CACHE_KEY_AGENT_SUBSCRIPTIONS.format(
                agent_id=subscription.agent_id
            )
            await valkey.sadd(agent_key, subscription.id)

        logger.info("Valkey cache rebuilt with %d subscriptions", len(rows))

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
        subscription_ids = {row["id"] for row in rows}

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
                SELECT id, agent_id, topic, webhook_url, webhook_secret,
                       webhook_headers, webhook_timeout_ms, status,
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
                await valkey.set(
                    sub_key,
                    subscription.model_dump_json(),
                    ttl=self._config.cache_ttl_seconds,
                )

        return subscriptions

    # =========================================================================
    # Internal Helpers - Delivery
    # =========================================================================

    async def _deliver_notification(
        self,
        subscription: ModelSubscription,
        event: ModelNotificationEvent,
    ) -> ModelNotificationDeliveryAttempt:
        """Deliver notification to a single subscriber.

        Args:
            subscription: The subscription to deliver to.
            event: The notification event.

        Returns:
            Delivery attempt record.
        """
        _, _, http_handler = self._ensure_initialized()

        delivery_id = str(uuid4())
        webhook_url = str(subscription.delivery.webhook_url)
        start_time = time.perf_counter()

        # Check circuit breaker
        circuit_allowed = await self._check_circuit_breaker(webhook_url)
        if not circuit_allowed:
            logger.warning(
                "Circuit breaker OPEN for endpoint %s, skipping delivery",
                webhook_url,
            )
            return ModelNotificationDeliveryAttempt(
                delivery_id=delivery_id,
                subscription_id=subscription.id,
                event_id=event.event_id,
                attempt_number=1,
                status=EnumDeliveryStatus.FAILED,
                error_message="Circuit breaker open - endpoint temporarily unavailable",
                created_at=datetime.now(timezone.utc),
            )

        # Build request payload
        payload_json = event.model_dump_json()

        # Build headers
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "X-Event-ID": event.event_id,
            "X-Subscription-ID": subscription.id,
        }

        # Add custom headers from subscription
        if subscription.delivery.headers:
            headers.update(subscription.delivery.headers)

        # Add HMAC signature if secret is configured
        if subscription.delivery.secret:
            signature = self._compute_hmac_signature(
                payload_json, subscription.delivery.secret
            )
            headers["X-Signature-256"] = signature

        # Execute webhook delivery
        try:
            envelope = {
                "operation": "http.post",
                "payload": {
                    "url": webhook_url,
                    "headers": headers,
                    "body": json.loads(payload_json),  # dict for httpx
                },
            }
            result = await http_handler.execute(envelope)

            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)

            # Parse response
            response_payload = result.result.get("payload", {})
            status_code = response_payload.get("status_code", 0)
            response_body = response_payload.get("body", "")

            # Determine success (2xx status codes)
            is_success = 200 <= status_code < 300

            # Update circuit breaker (pass error message on failure)
            error_msg = None if is_success else f"HTTP {status_code}"
            await self._update_circuit_breaker(
                webhook_url, success=is_success, error_message=error_msg
            )

            if is_success:
                attempt = ModelNotificationDeliveryAttempt(
                    delivery_id=delivery_id,
                    subscription_id=subscription.id,
                    event_id=event.event_id,
                    attempt_number=1,
                    status=EnumDeliveryStatus.SUCCESS,
                    status_code=status_code,
                    response_body=str(response_body)[:4096] if response_body else None,
                    latency_ms=latency_ms,
                    created_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                )
            else:
                # Schedule retry
                attempt = await self._handle_delivery_failure(
                    delivery_id=delivery_id,
                    subscription=subscription,
                    event=event,
                    attempt_number=1,
                    status_code=status_code,
                    error_message=f"HTTP {status_code}",
                    response_body=str(response_body)[:4096] if response_body else None,
                    latency_ms=latency_ms,
                )

            # Persist delivery attempt
            await self._persist_delivery_attempt(attempt)

            return attempt

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = int((end_time - start_time) * 1000)

            # Update circuit breaker on failure with error message
            await self._update_circuit_breaker(
                webhook_url, success=False, error_message=str(e)[:2048]
            )

            attempt = await self._handle_delivery_failure(
                delivery_id=delivery_id,
                subscription=subscription,
                event=event,
                attempt_number=1,
                status_code=None,
                error_message=str(e)[:2048],
                response_body=None,
                latency_ms=latency_ms,
            )

            await self._persist_delivery_attempt(attempt)

            return attempt

    async def _handle_delivery_failure(
        self,
        delivery_id: str,
        subscription: ModelSubscription,
        event: ModelNotificationEvent,
        attempt_number: int,
        status_code: int | None,
        error_message: str,
        response_body: str | None,
        latency_ms: int,
    ) -> ModelNotificationDeliveryAttempt:
        """Handle a failed delivery attempt.

        Schedules retry with exponential backoff or moves to DLQ.

        Args:
            delivery_id: The delivery attempt ID.
            subscription: The subscription.
            event: The notification event.
            attempt_number: Current attempt number.
            status_code: HTTP status code (if available).
            error_message: Error message.
            response_body: Response body (if available).
            latency_ms: Request latency.

        Returns:
            Delivery attempt record.
        """
        if attempt_number >= self._config.max_retry_attempts:
            # Move to DLQ
            logger.warning(
                "Delivery to %s failed after %d attempts, moving to DLQ",
                subscription.delivery.webhook_url,
                attempt_number,
            )
            return ModelNotificationDeliveryAttempt(
                delivery_id=delivery_id,
                subscription_id=subscription.id,
                event_id=event.event_id,
                attempt_number=attempt_number,
                status=EnumDeliveryStatus.DLQ,
                status_code=status_code,
                error_message=error_message,
                response_body=response_body,
                latency_ms=latency_ms,
                created_at=datetime.now(timezone.utc),
            )

        # Calculate next retry time with exponential backoff.
        # NOTE: This handler only records the scheduled retry time; it does NOT
        # execute the retry. A separate background worker must poll the
        # delivery_attempts table for pending retries (see module docstring).
        #
        # TODO(OMN-XXXX): Implement RetryWorker that polls delivery_attempts table
        # for rows where status='failed' AND next_retry_at <= NOW() and re-invokes
        # delivery. See also: module docstring for architectural rationale.
        delay_ms = min(
            self._config.retry_base_delay_ms * (2 ** (attempt_number - 1)),
            self._config.retry_max_delay_ms,
        )
        next_retry_at = datetime.now(timezone.utc) + timedelta(milliseconds=delay_ms)

        logger.info(
            "Scheduling retry for subscription %s, attempt %d in %d ms",
            subscription.id,
            attempt_number + 1,
            delay_ms,
        )

        return ModelNotificationDeliveryAttempt(
            delivery_id=delivery_id,
            subscription_id=subscription.id,
            event_id=event.event_id,
            attempt_number=attempt_number,
            status=EnumDeliveryStatus.FAILED,
            status_code=status_code,
            error_message=error_message,
            response_body=response_body,
            latency_ms=latency_ms,
            next_retry_at=next_retry_at,
            created_at=datetime.now(timezone.utc),
        )

    async def _persist_delivery_attempt(
        self,
        attempt: ModelNotificationDeliveryAttempt,
    ) -> None:
        """Persist delivery attempt to database.

        Args:
            attempt: The delivery attempt to persist.
        """
        _, db_handler, _ = self._ensure_initialized()

        sql = """
            INSERT INTO delivery_attempts (
                id, subscription_id, event_id, attempt_number, status,
                status_code, error_message, response_body, latency_ms,
                next_retry_at, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (subscription_id, event_id, attempt_number) DO UPDATE SET
                status = EXCLUDED.status,
                status_code = EXCLUDED.status_code,
                error_message = EXCLUDED.error_message,
                response_body = EXCLUDED.response_body,
                latency_ms = EXCLUDED.latency_ms,
                next_retry_at = EXCLUDED.next_retry_at
        """
        envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": sql,
                "parameters": [
                    attempt.delivery_id,
                    attempt.subscription_id,
                    attempt.event_id,
                    attempt.attempt_number,
                    attempt.status.value,
                    attempt.status_code,
                    attempt.error_message,
                    attempt.response_body,
                    attempt.latency_ms,
                    attempt.next_retry_at.isoformat()
                    if attempt.next_retry_at
                    else None,
                    attempt.created_at.isoformat(),
                ],
            },
        }
        await db_handler.execute(envelope)

    def _compute_hmac_signature(self, payload: str, secret: str) -> str:
        """Compute HMAC-SHA256 signature for webhook payload.

        Args:
            payload: The JSON payload string.
            secret: The shared secret.

        Returns:
            Hex-encoded signature with "sha256=" prefix.
        """
        signature = hmac.new(
            secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"sha256={signature}"

    # =========================================================================
    # Internal Helpers - Circuit Breaker
    # =========================================================================

    def _endpoint_hash(self, endpoint: str) -> str:
        """Generate hash for endpoint to use as cache key.

        Args:
            endpoint: The webhook URL.

        Returns:
            SHA256 hash of the endpoint (first 16 chars).
        """
        return hashlib.sha256(endpoint.encode()).hexdigest()[:16]

    async def _check_circuit_breaker(self, endpoint: str) -> bool:
        """Check if a request should be allowed through circuit breaker.

        Tries Valkey cache first for speed, falls back to PostgreSQL if cache misses.
        On DB fallback, repopulates the Valkey cache for subsequent requests.

        Args:
            endpoint: The webhook endpoint URL.

        Returns:
            True if request should proceed, False if circuit is open.
        """
        valkey, _, _ = self._ensure_initialized()

        endpoint_hash = self._endpoint_hash(endpoint)
        cb_key = CACHE_KEY_CIRCUIT_BREAKER.format(endpoint_hash=endpoint_hash)

        # Try Valkey cache first
        cached = await valkey.get(cb_key)
        state_dict: dict[str, object] | None = None

        if cached:
            try:
                state_dict = json.loads(cached)
            except Exception as e:
                logger.warning("Failed to parse cached circuit breaker state: %s", e)
                state_dict = None

        # Fallback to PostgreSQL if Valkey misses
        if state_dict is None:
            state_dict = await self._load_circuit_breaker_from_db(endpoint)
            if state_dict:
                # Repopulate Valkey cache from DB with TTL refresh
                await valkey.set(
                    cb_key,
                    json.dumps(state_dict),
                    ttl=CIRCUIT_BREAKER_CACHE_TTL_SECONDS,
                )
                logger.debug(
                    "Circuit breaker state for %s loaded from DB and cached",
                    endpoint,
                )

        if not state_dict:
            # No circuit breaker state = allow (first request to this endpoint)
            return True

        try:
            state = EnumCircuitBreakerState(state_dict.get("state", "closed"))

            if state == EnumCircuitBreakerState.CLOSED:
                return True

            if state == EnumCircuitBreakerState.OPEN:
                # Check if cooldown has passed
                open_until = state_dict.get("open_until")
                if open_until:
                    open_until_str = str(open_until)
                    open_until_dt = datetime.fromisoformat(
                        open_until_str.replace("Z", "+00:00")
                    )
                    if datetime.now(timezone.utc) >= open_until_dt:
                        # Transition to half-open state. TTL is refreshed to prevent
                        # state loss during the half-open testing period.
                        state_dict["state"] = EnumCircuitBreakerState.HALF_OPEN.value
                        await valkey.set(
                            cb_key,
                            json.dumps(state_dict),
                            ttl=CIRCUIT_BREAKER_CACHE_TTL_SECONDS,
                        )
                        # Also persist the state transition to DB
                        await self._persist_circuit_breaker_to_db(endpoint, state_dict)
                        return True
                return False

            # HALF_OPEN: allow request for testing
            return True

        except Exception as e:
            logger.warning("Failed to check circuit breaker: %s", e)
            return True  # Fail open

    async def _load_circuit_breaker_from_db(
        self,
        endpoint: str,
    ) -> dict[str, object] | None:
        """Load circuit breaker state from PostgreSQL.

        Args:
            endpoint: The webhook endpoint URL.

        Returns:
            Circuit breaker state dict if found, None otherwise.
        """
        _, db_handler, _ = self._ensure_initialized()

        sql = """
            SELECT state, failure_count, success_count, last_failure_at,
                   last_success_at, last_error_message, open_until,
                   total_requests, total_failures, updated_at
            FROM circuit_breaker_states
            WHERE endpoint = $1
        """
        envelope = {
            "operation": "db.query",
            "payload": {
                "sql": sql,
                "parameters": [endpoint],
            },
        }

        try:
            result = await db_handler.execute(envelope)
            rows = result.result.get("payload", {}).get("rows", [])

            if not rows:
                return None

            row = rows[0]

            # Convert DB row to state dict format used by Valkey
            state_dict: dict[str, object] = {
                "state": row.get("state", "closed"),
                "failure_count": row.get("failure_count", 0),
                "success_count": row.get("success_count", 0),
                "total_requests": row.get("total_requests", 0),
                "total_failures": row.get("total_failures", 0),
            }

            # Handle nullable datetime fields
            if row.get("last_failure_at"):
                last_failure = row["last_failure_at"]
                if isinstance(last_failure, datetime):
                    state_dict["last_failure_at"] = last_failure.isoformat()
                else:
                    state_dict["last_failure_at"] = str(last_failure)

            if row.get("last_success_at"):
                last_success = row["last_success_at"]
                if isinstance(last_success, datetime):
                    state_dict["last_success_at"] = last_success.isoformat()
                else:
                    state_dict["last_success_at"] = str(last_success)

            if row.get("open_until"):
                open_until = row["open_until"]
                if isinstance(open_until, datetime):
                    state_dict["open_until"] = open_until.isoformat()
                else:
                    state_dict["open_until"] = str(open_until)

            if row.get("last_error_message"):
                state_dict["last_error_message"] = row["last_error_message"]

            if row.get("updated_at"):
                updated_at = row["updated_at"]
                if isinstance(updated_at, datetime):
                    state_dict["updated_at"] = updated_at.isoformat()
                else:
                    state_dict["updated_at"] = str(updated_at)

            return state_dict

        except Exception as e:
            logger.warning("Failed to load circuit breaker from DB: %s", e)
            return None

    async def _update_circuit_breaker(
        self,
        endpoint: str,
        success: bool,
        error_message: str | None = None,
    ) -> None:
        """Update circuit breaker state after a delivery attempt.

        Persists state to both Valkey (primary cache) and PostgreSQL (durable backup).

        Args:
            endpoint: The webhook endpoint URL.
            success: Whether the delivery was successful.
            error_message: Error message if delivery failed (optional).
        """
        valkey, _, _ = self._ensure_initialized()

        endpoint_hash = self._endpoint_hash(endpoint)
        cb_key = CACHE_KEY_CIRCUIT_BREAKER.format(endpoint_hash=endpoint_hash)

        cached = await valkey.get(cb_key)
        now = datetime.now(timezone.utc)

        if cached:
            try:
                state_dict = json.loads(cached)
            except Exception:
                state_dict = {
                    "state": "closed",
                    "failure_count": 0,
                    "success_count": 0,
                    "total_requests": 0,
                    "total_failures": 0,
                }
        else:
            # Try to load from DB if not in cache
            state_dict = await self._load_circuit_breaker_from_db(endpoint)
            if not state_dict:
                state_dict = {
                    "state": "closed",
                    "failure_count": 0,
                    "success_count": 0,
                    "total_requests": 0,
                    "total_failures": 0,
                }

        current_state = EnumCircuitBreakerState(state_dict.get("state", "closed"))

        # Update total requests counter
        state_dict["total_requests"] = state_dict.get("total_requests", 0) + 1

        if success:
            state_dict["success_count"] = state_dict.get("success_count", 0) + 1
            state_dict["failure_count"] = 0
            state_dict["last_success_at"] = now.isoformat()

            if current_state == EnumCircuitBreakerState.HALF_OPEN:
                # Successful test - close the circuit
                if (
                    state_dict["success_count"]
                    >= self._config.circuit_breaker_success_threshold
                ):
                    state_dict["state"] = EnumCircuitBreakerState.CLOSED.value
                    state_dict["success_count"] = 0
                    logger.info("Circuit breaker CLOSED for endpoint %s", endpoint)

        else:
            state_dict["failure_count"] = state_dict.get("failure_count", 0) + 1
            state_dict["success_count"] = 0
            state_dict["last_failure_at"] = now.isoformat()
            state_dict["total_failures"] = state_dict.get("total_failures", 0) + 1

            # Store the error message for debugging
            if error_message:
                state_dict["last_error_message"] = error_message[:2048]

            if current_state == EnumCircuitBreakerState.CLOSED:
                if (
                    state_dict["failure_count"]
                    >= self._config.circuit_breaker_threshold
                ):
                    state_dict["state"] = EnumCircuitBreakerState.OPEN.value
                    open_until = now + timedelta(
                        seconds=self._config.circuit_breaker_cooldown_seconds
                    )
                    state_dict["open_until"] = open_until.isoformat()
                    logger.warning(
                        "Circuit breaker OPEN for endpoint %s until %s",
                        endpoint,
                        open_until.isoformat(),
                    )

            elif current_state == EnumCircuitBreakerState.HALF_OPEN:
                # Failed test - reopen the circuit
                state_dict["state"] = EnumCircuitBreakerState.OPEN.value
                open_until = now + timedelta(
                    seconds=self._config.circuit_breaker_cooldown_seconds
                )
                state_dict["open_until"] = open_until.isoformat()
                logger.warning(
                    "Circuit breaker reopened (HALF_OPEN -> OPEN) for endpoint %s",
                    endpoint,
                )

        state_dict["updated_at"] = now.isoformat()

        # Persist to Valkey (primary cache).
        # TTL is refreshed on every update to prevent state loss for long-lived circuits.
        # This ensures that actively-used circuits never expire from cache.
        await valkey.set(
            cb_key,
            json.dumps(state_dict),
            ttl=CIRCUIT_BREAKER_CACHE_TTL_SECONDS,
        )

        # Persist to PostgreSQL (durable backup)
        await self._persist_circuit_breaker_to_db(endpoint, state_dict)

    async def _persist_circuit_breaker_to_db(
        self,
        endpoint: str,
        state_dict: dict[str, object],
    ) -> None:
        """Persist circuit breaker state to PostgreSQL for durability.

        Uses UPSERT pattern (ON CONFLICT ... DO UPDATE) since endpoint is the
        primary key.

        Args:
            endpoint: The webhook endpoint URL (primary key, max 512 chars).
            state_dict: Circuit breaker state dictionary.
        """
        _, db_handler, _ = self._ensure_initialized()

        # Truncate endpoint if too long (VARCHAR(512) limit)
        endpoint_to_store = endpoint[:512]

        # Parse datetime fields from ISO format strings
        last_failure_at = None
        if state_dict.get("last_failure_at"):
            try:
                last_failure_str = str(state_dict["last_failure_at"])
                last_failure_at = datetime.fromisoformat(
                    last_failure_str.replace("Z", "+00:00")
                ).isoformat()
            except (ValueError, TypeError):
                pass

        last_success_at = None
        if state_dict.get("last_success_at"):
            try:
                last_success_str = str(state_dict["last_success_at"])
                last_success_at = datetime.fromisoformat(
                    last_success_str.replace("Z", "+00:00")
                ).isoformat()
            except (ValueError, TypeError):
                pass

        open_until = None
        if state_dict.get("open_until"):
            try:
                open_until_str = str(state_dict["open_until"])
                open_until = datetime.fromisoformat(
                    open_until_str.replace("Z", "+00:00")
                ).isoformat()
            except (ValueError, TypeError):
                pass

        sql = """
            INSERT INTO circuit_breaker_states (
                endpoint, state, failure_count, success_count,
                last_failure_at, last_success_at, last_error_message,
                open_until, total_requests, total_failures
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (endpoint) DO UPDATE SET
                state = EXCLUDED.state,
                failure_count = EXCLUDED.failure_count,
                success_count = EXCLUDED.success_count,
                last_failure_at = EXCLUDED.last_failure_at,
                last_success_at = EXCLUDED.last_success_at,
                last_error_message = EXCLUDED.last_error_message,
                open_until = EXCLUDED.open_until,
                total_requests = EXCLUDED.total_requests,
                total_failures = EXCLUDED.total_failures
        """

        envelope = {
            "operation": "db.execute",
            "payload": {
                "sql": sql,
                "parameters": [
                    endpoint_to_store,
                    state_dict.get("state", "closed"),
                    state_dict.get("failure_count", 0),
                    state_dict.get("success_count", 0),
                    last_failure_at,
                    last_success_at,
                    state_dict.get("last_error_message"),
                    open_until,
                    state_dict.get("total_requests", 0),
                    state_dict.get("total_failures", 0),
                ],
            },
        }

        try:
            await db_handler.execute(envelope)
            logger.debug(
                "Circuit breaker state persisted to DB for endpoint %s",
                endpoint_to_store,
            )
        except Exception as e:
            # Log but don't fail - Valkey is the primary cache
            logger.warning(
                "Failed to persist circuit breaker to DB for endpoint %s: %s",
                endpoint_to_store,
                e,
            )

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
                # Simple query to test connection
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

        # HTTP handler doesn't have a health check, assume healthy if initialized
        http_healthy = self._http_handler is not None

        is_healthy = valkey_healthy and db_healthy and http_healthy

        return ModelSubscriptionHealth(
            is_healthy=is_healthy,
            initialized=True,
            db_healthy=db_healthy,
            valkey_healthy=valkey_healthy,
            http_healthy=http_healthy,
            error_message="; ".join(errors) if errors else None,
        )
