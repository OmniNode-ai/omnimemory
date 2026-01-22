# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerSubscription.

This module tests the subscription handler against real or mocked external
services to verify end-to-end functionality.

Test Categories:
    - Subscribe: Agent subscription creation and idempotency
    - Unsubscribe: Subscription removal
    - Notify: Notification delivery with webhooks
    - Persistence: Subscriptions survive service restart
    - Retry/CircuitBreaker: Failed notification handling

Prerequisites:
    - PostgreSQL running at TEST_DB_DSN
    - Valkey running at TEST_VALKEY_HOST:TEST_VALKEY_PORT
    - omnibase_infra installed (dev dependency)

Usage:
    # Run subscription handler tests
    pytest tests/integration/test_handler_subscription.py -v

    # Run with markers
    pytest -m "integration and subscription" -v

Environment Variables:
    TEST_DB_DSN: PostgreSQL connection string
    TEST_VALKEY_HOST: Valkey hostname (default: localhost)
    TEST_VALKEY_PORT: Valkey port (default: 6379)

.. versionadded:: 0.1.0
    Initial implementation for OMN-1393.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest

# Check if omnibase_infra is available
_OMNIBASE_INFRA_AVAILABLE = False
_SKIP_REASON = "omnibase_infra not installed"

try:
    from omnimemory.enums.enum_subscription_status import (
        EnumDeliveryStatus,
        EnumSubscriptionStatus,
    )
    from omnimemory.handlers import (
        HandlerSubscription,
        ModelHandlerSubscriptionConfig,
    )
    from omnimemory.models.subscription import (
        ModelNotificationEvent,
        ModelNotificationEventPayload,
        ModelSubscriptionDeliveryWebhook,
    )

    _OMNIBASE_INFRA_AVAILABLE = True
    _SKIP_REASON = ""
except ImportError as e:
    _SKIP_REASON = f"Required dependencies not available: {e}"


# =============================================================================
# Skip Conditions
# =============================================================================

# Skip all tests if dependencies are not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.subscription,
    pytest.mark.skipif(
        not _OMNIBASE_INFRA_AVAILABLE,
        reason=_SKIP_REASON,
    ),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def subscription_handler(
    test_db_dsn: str,
    test_valkey_host: str,
    test_valkey_port: int,
    services_available: bool,
) -> AsyncGenerator[HandlerSubscription, None]:
    """Create and initialize a HandlerSubscription for tests.

    Yields:
        Initialized HandlerSubscription instance.
    """
    if not services_available:
        pytest.skip("Required services (PostgreSQL, Valkey) not available")

    config = ModelHandlerSubscriptionConfig(
        db_dsn=test_db_dsn,
        valkey_host=test_valkey_host,
        valkey_port=test_valkey_port,
        max_retry_attempts=3,
        retry_base_delay_ms=100,  # Fast retries for tests
        circuit_breaker_threshold=5,
        circuit_breaker_cooldown_seconds=10,
        http_timeout_seconds=5.0,
    )
    handler = HandlerSubscription(config)
    await handler.initialize()

    yield handler

    await handler.shutdown()


@pytest.fixture
def unique_agent_id() -> str:
    """Generate a unique agent ID for test isolation.

    Returns:
        Unique agent identifier.
    """
    return f"test_agent_{uuid4().hex[:8]}"


@pytest.fixture
def unique_topic() -> str:
    """Generate a unique topic for test isolation.

    Returns:
        Unique topic in memory.<entity>.<event> format.
    """
    return f"memory.test_{uuid4().hex[:8]}.created"


# =============================================================================
# Subscribe Tests
# =============================================================================


class TestSubscribe:
    """Tests for subscribe operation."""

    @pytest.mark.asyncio
    async def test_subscribe_creates_subscription(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
        webhook_url: str,
    ) -> None:
        """Agent can subscribe to a memory topic."""
        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=webhook_url,
        )

        subscription = await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )

        assert subscription.agent_id == unique_agent_id
        assert subscription.topic == unique_topic
        assert subscription.status == EnumSubscriptionStatus.ACTIVE
        assert subscription.id is not None

    @pytest.mark.asyncio
    async def test_subscribe_validates_topic_format(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        webhook_url: str,
    ) -> None:
        """Topic must match memory.<entity>.<event> pattern."""
        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=webhook_url,
        )

        with pytest.raises(ValueError, match="Topic must match pattern"):
            await subscription_handler.subscribe(
                agent_id=unique_agent_id,
                topic="invalid-topic",
                delivery=delivery,
            )

    @pytest.mark.asyncio
    async def test_subscribe_duplicate_is_idempotent(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
        webhook_url: str,
    ) -> None:
        """Subscribing twice to same topic returns existing subscription."""
        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=webhook_url,
        )

        sub1 = await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )

        sub2 = await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )

        assert sub1.id == sub2.id

    @pytest.mark.asyncio
    async def test_subscribe_with_metadata(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
        webhook_url: str,
    ) -> None:
        """Subscription can include metadata."""
        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=webhook_url,
        )
        metadata = {"environment": "test", "version": "1.0"}

        subscription = await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
            metadata=metadata,
        )

        assert subscription.metadata == metadata

    @pytest.mark.asyncio
    async def test_subscribe_with_custom_headers(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
        webhook_url: str,
    ) -> None:
        """Subscription can include custom webhook headers."""
        custom_headers = {"X-Custom-Header": "test-value"}
        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=webhook_url,
            headers=custom_headers,
        )

        subscription = await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )

        assert subscription.delivery.headers == custom_headers


# =============================================================================
# Unsubscribe Tests
# =============================================================================


class TestUnsubscribe:
    """Tests for unsubscribe operation."""

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_subscription(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
        webhook_url: str,
    ) -> None:
        """Agent can unsubscribe from a topic."""
        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=webhook_url,
        )

        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )

        result = await subscription_handler.unsubscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        assert result is True

        # Verify subscription is gone
        subs = await subscription_handler.list_subscriptions(unique_agent_id)
        assert len(subs) == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_returns_false(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
    ) -> None:
        """Unsubscribing from non-existent subscription returns False."""
        result = await subscription_handler.unsubscribe(
            agent_id=unique_agent_id,
            topic="memory.nonexistent.topic",
        )

        assert result is False


# =============================================================================
# List Subscriptions Tests
# =============================================================================


class TestListSubscriptions:
    """Tests for list_subscriptions operation."""

    @pytest.mark.asyncio
    async def test_list_subscriptions_returns_all(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        webhook_url: str,
    ) -> None:
        """List subscriptions returns all agent subscriptions."""
        topics = [
            f"memory.test_{uuid4().hex[:8]}.created",
            f"memory.test_{uuid4().hex[:8]}.updated",
            f"memory.test_{uuid4().hex[:8]}.deleted",
        ]
        delivery = ModelSubscriptionDeliveryWebhook(webhook_url=webhook_url)

        for topic in topics:
            await subscription_handler.subscribe(
                agent_id=unique_agent_id,
                topic=topic,
                delivery=delivery,
            )

        subs = await subscription_handler.list_subscriptions(unique_agent_id)

        assert len(subs) == 3
        returned_topics = {s.topic for s in subs}
        assert returned_topics == set(topics)

    @pytest.mark.asyncio
    async def test_list_subscriptions_empty_for_new_agent(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
    ) -> None:
        """List subscriptions returns empty for agent with no subscriptions."""
        subs = await subscription_handler.list_subscriptions(unique_agent_id)

        assert len(subs) == 0


# =============================================================================
# Notify Tests
# =============================================================================


class TestNotify:
    """Tests for notify operation."""

    @pytest.mark.asyncio
    async def test_notify_sends_to_subscribers(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
        webhook_server_with_events: tuple[str, list[dict[str, object]]],
    ) -> None:
        """Memory changes trigger notifications to subscribers."""
        webhook_url, received_events = webhook_server_with_events

        # Subscribe
        delivery = ModelSubscriptionDeliveryWebhook(webhook_url=webhook_url)
        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )

        # Notify
        event_id = str(uuid4())
        event = ModelNotificationEvent(
            event_id=event_id,
            topic=unique_topic,
            payload=ModelNotificationEventPayload(
                entity_type="test_item",
                entity_id="item_123",
                action="created",
            ),
        )

        attempts = await subscription_handler.notify(
            topic=unique_topic,
            event=event,
        )

        assert len(attempts) == 1
        assert attempts[0].status == EnumDeliveryStatus.SUCCESS
        assert len(received_events) == 1
        assert received_events[0]["body"]["event_id"] == event_id

    @pytest.mark.asyncio
    async def test_notify_includes_hmac_signature(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
        webhook_server_with_events: tuple[str, list[dict[str, object]]],
    ) -> None:
        """Webhook includes HMAC signature when secret is provided."""
        webhook_url, received_events = webhook_server_with_events
        secret = "test-secret-123"

        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=webhook_url,
            secret=secret,
        )
        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )

        event = ModelNotificationEvent(
            event_id=str(uuid4()),
            topic=unique_topic,
            payload=ModelNotificationEventPayload(
                entity_type="test_item",
                entity_id="item_123",
                action="created",
            ),
        )

        await subscription_handler.notify(
            topic=unique_topic,
            event=event,
        )

        assert len(received_events) == 1
        signature = received_events[0]["signature"]
        assert signature is not None
        assert signature.startswith("sha256=")

    @pytest.mark.asyncio
    async def test_notify_no_subscribers_returns_empty(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Notify with no subscribers returns empty list."""
        event = ModelNotificationEvent(
            event_id=str(uuid4()),
            topic="memory.orphan.topic",
            payload=ModelNotificationEventPayload(
                entity_type="orphan",
                entity_id="orphan_123",
                action="created",
            ),
        )

        attempts = await subscription_handler.notify(
            topic="memory.orphan.topic",
            event=event,
        )

        assert len(attempts) == 0

    @pytest.mark.asyncio
    async def test_notify_multiple_subscribers(
        self,
        subscription_handler: HandlerSubscription,
        unique_topic: str,
        webhook_server_with_events: tuple[str, list[dict[str, object]]],
    ) -> None:
        """Notify delivers to all subscribers of a topic."""
        webhook_url, received_events = webhook_server_with_events

        # Subscribe multiple agents
        agents = [f"agent_{uuid4().hex[:8]}" for _ in range(3)]
        delivery = ModelSubscriptionDeliveryWebhook(webhook_url=webhook_url)

        for agent_id in agents:
            await subscription_handler.subscribe(
                agent_id=agent_id,
                topic=unique_topic,
                delivery=delivery,
            )

        # Notify
        event = ModelNotificationEvent(
            event_id=str(uuid4()),
            topic=unique_topic,
            payload=ModelNotificationEventPayload(
                entity_type="test_item",
                entity_id="item_123",
                action="created",
            ),
        )

        attempts = await subscription_handler.notify(
            topic=unique_topic,
            event=event,
        )

        assert len(attempts) == 3
        assert all(a.status == EnumDeliveryStatus.SUCCESS for a in attempts)
        assert len(received_events) == 3


# =============================================================================
# Survive Restart Tests
# =============================================================================


class TestSurviveRestart:
    """Tests for subscription persistence across restarts."""

    @pytest.mark.asyncio
    async def test_subscriptions_survive_restart(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
        webhook_url: str,
    ) -> None:
        """Subscriptions persist across handler restarts."""
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
        )

        # Create subscription with first handler instance
        handler1 = HandlerSubscription(config)
        await handler1.initialize()

        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=webhook_url,
        )
        sub = await handler1.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )
        original_id = sub.id

        await handler1.shutdown()

        # Create new handler instance (simulates restart)
        handler2 = HandlerSubscription(config)
        await handler2.initialize()

        # Verify subscription still exists
        subs = await handler2.list_subscriptions(unique_agent_id)

        assert len(subs) == 1
        assert subs[0].id == original_id
        assert subs[0].topic == unique_topic

        await handler2.shutdown()

    @pytest.mark.asyncio
    async def test_cache_rebuilds_from_db_on_restart(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        webhook_url: str,
    ) -> None:
        """Valkey cache is rebuilt from PostgreSQL on cold start."""
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
        )

        # Create multiple subscriptions
        topics = [
            f"memory.restart_{uuid4().hex[:8]}.created",
            f"memory.restart_{uuid4().hex[:8]}.updated",
        ]

        handler1 = HandlerSubscription(config)
        await handler1.initialize()

        delivery = ModelSubscriptionDeliveryWebhook(webhook_url=webhook_url)
        for topic in topics:
            await handler1.subscribe(
                agent_id=unique_agent_id,
                topic=topic,
                delivery=delivery,
            )

        await handler1.shutdown()

        # New handler should rebuild cache from DB
        handler2 = HandlerSubscription(config)
        await handler2.initialize()

        # All subscriptions should be available
        subs = await handler2.list_subscriptions(unique_agent_id)
        assert len(subs) == 2

        returned_topics = {s.topic for s in subs}
        assert returned_topics == set(topics)

        await handler2.shutdown()


# =============================================================================
# Retry Behavior Tests
# =============================================================================


class TestRetryBehavior:
    """Tests for retry and circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_failed_notification_is_retried(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Failed notifications are scheduled for retry."""
        # Subscribe with unreachable webhook
        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url="http://localhost:59999/unreachable",
            timeout_ms=100,
        )
        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )

        event = ModelNotificationEvent(
            event_id=str(uuid4()),
            topic=unique_topic,
            payload=ModelNotificationEventPayload(
                entity_type="test_item",
                entity_id="item_123",
                action="created",
            ),
        )

        attempts = await subscription_handler.notify(
            topic=unique_topic,
            event=event,
        )

        assert len(attempts) == 1
        assert attempts[0].status == EnumDeliveryStatus.FAILED
        assert attempts[0].next_retry_at is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Circuit breaker opens after consecutive failures."""
        # Subscribe with unreachable webhook
        unreachable_url = "http://localhost:59999/unreachable"
        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=unreachable_url,
            timeout_ms=100,
        )
        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )

        # Trigger multiple failures to open circuit breaker
        # Default threshold is 5
        for i in range(6):
            event = ModelNotificationEvent(
                event_id=str(uuid4()),
                topic=unique_topic,
                payload=ModelNotificationEventPayload(
                    entity_type="test_item",
                    entity_id=f"item_{i}",
                    action="created",
                ),
            )
            await subscription_handler.notify(
                topic=unique_topic,
                event=event,
            )

        # Check circuit breaker is open
        is_allowed = await subscription_handler._check_circuit_breaker(unreachable_url)
        assert is_allowed is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_after_cooldown(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Circuit breaker transitions to half-open after cooldown."""
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        # Use very short cooldown for testing
        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown_seconds=10,  # Minimum allowed
            http_timeout_seconds=1.0,
        )
        handler = HandlerSubscription(config)
        await handler.initialize()

        try:
            unreachable_url = "http://localhost:59999/unreachable"
            delivery = ModelSubscriptionDeliveryWebhook(
                webhook_url=unreachable_url,
                timeout_ms=100,
            )
            await handler.subscribe(
                agent_id=unique_agent_id,
                topic=unique_topic,
                delivery=delivery,
            )

            # Trigger failures to open circuit
            for i in range(3):
                event = ModelNotificationEvent(
                    event_id=str(uuid4()),
                    topic=unique_topic,
                    payload=ModelNotificationEventPayload(
                        entity_type="test_item",
                        entity_id=f"item_{i}",
                        action="created",
                    ),
                )
                await handler.notify(topic=unique_topic, event=event)

            # Circuit should be open now
            is_allowed = await handler._check_circuit_breaker(unreachable_url)
            assert is_allowed is False

        finally:
            await handler.shutdown()


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for handler health check."""

    @pytest.mark.asyncio
    async def test_health_check_returns_healthy(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Health check returns healthy for initialized handler."""
        health = await subscription_handler.health_check()

        assert health.is_healthy is True
        assert health.initialized is True
        assert health.db_healthy is True
        assert health.valkey_healthy is True
        assert health.http_healthy is True
        assert health.error_message is None

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
    ) -> None:
        """Health check returns unhealthy when not initialized."""
        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
        )
        handler = HandlerSubscription(config)

        # Not initialized
        health = await handler.health_check()

        assert health.is_healthy is False
        assert health.initialized is False
        assert health.error_message is not None
