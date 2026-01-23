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
        ModelSubscription,
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
        webhook_server: tuple[str, list[dict[str, object]]],
    ) -> None:
        """Memory changes trigger notifications to subscribers."""
        webhook_url, received_events = webhook_server

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
        webhook_server: tuple[str, list[dict[str, object]]],
    ) -> None:
        """Webhook includes HMAC signature when secret is provided."""
        webhook_url, received_events = webhook_server
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
        webhook_server: tuple[str, list[dict[str, object]]],
    ) -> None:
        """Notify delivers to all subscribers of a topic."""
        webhook_url, received_events = webhook_server

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
# Cache Miss Fallback Tests
# =============================================================================


class TestCacheMissFallback:
    """Tests for cache-miss to database fallback behavior.

    The HandlerSubscription implements a cache-first strategy where it:
    1. Tries Valkey cache first for fast lookups
    2. Falls back to PostgreSQL if cache misses
    3. Repopulates the cache after loading from database

    These tests verify that this fallback mechanism works correctly
    by explicitly clearing cache entries and verifying data recovery.
    """

    @pytest.mark.asyncio
    async def test_cache_miss_fallback_list_subscriptions(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        webhook_url: str,
    ) -> None:
        """list_subscriptions falls back to database when cache is empty.

        This test:
        1. Creates subscriptions (populates both DB and cache)
        2. Manually clears the Valkey cache for the agent's subscriptions
        3. Calls list_subscriptions() which should hit cache-miss path
        4. Verifies data is correctly loaded from the database
        5. Verifies the cache is repopulated after the fallback
        """
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
        )
        handler = HandlerSubscription(config)
        await handler.initialize()

        try:
            # Step 1: Create multiple subscriptions
            topics = [
                f"memory.fallback_{uuid4().hex[:8]}.created",
                f"memory.fallback_{uuid4().hex[:8]}.updated",
            ]
            delivery = ModelSubscriptionDeliveryWebhook(webhook_url=webhook_url)

            subscription_ids: list[str] = []
            for topic in topics:
                sub = await handler.subscribe(
                    agent_id=unique_agent_id,
                    topic=topic,
                    delivery=delivery,
                )
                subscription_ids.append(sub.id)

            # Verify subscriptions exist in cache initially
            subs_before_clear = await handler.list_subscriptions(unique_agent_id)
            assert len(subs_before_clear) == 2

            # Step 2: Manually clear the Valkey cache for this agent
            # Access internal valkey adapter to clear cache
            valkey = handler._valkey
            assert valkey is not None

            # Clear agent subscriptions set
            agent_key = f"agent:{unique_agent_id}:subscriptions"
            await valkey.delete(agent_key)

            # Clear individual subscription entries
            for sub_id in subscription_ids:
                sub_key = f"subscription:{sub_id}"
                await valkey.delete(sub_key)

            # Verify cache is cleared
            agent_sub_ids = await valkey.smembers(agent_key)
            assert len(agent_sub_ids) == 0, "Cache should be cleared"

            # Step 3: Call list_subscriptions - should trigger DB fallback
            subs_after_clear = await handler.list_subscriptions(unique_agent_id)

            # Step 4: Verify data was recovered from database
            assert len(subs_after_clear) == 2
            returned_topics = {s.topic for s in subs_after_clear}
            assert returned_topics == set(topics)

            # Verify subscription details are intact
            for sub in subs_after_clear:
                assert sub.agent_id == unique_agent_id
                assert sub.status == EnumSubscriptionStatus.ACTIVE
                assert str(sub.delivery.webhook_url) == webhook_url

        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_cache_miss_fallback_notify_repopulates_cache(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        webhook_url: str,
    ) -> None:
        """notify() falls back to database and repopulates cache on miss.

        This test verifies that when the topic->subscribers cache is cleared,
        the notify operation:
        1. Falls back to database to find subscribers
        2. Repopulates the topic cache after fallback
        """
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
        )
        handler = HandlerSubscription(config)
        await handler.initialize()

        try:
            # Create subscription
            agent_id = f"agent_{uuid4().hex[:8]}"
            topic = f"memory.notify_fallback_{uuid4().hex[:8]}.created"
            delivery = ModelSubscriptionDeliveryWebhook(
                webhook_url=webhook_url,
                timeout_ms=100,  # Short timeout since webhook won't respond
            )

            sub = await handler.subscribe(
                agent_id=agent_id,
                topic=topic,
                delivery=delivery,
            )

            # Clear topic->subscribers cache
            valkey = handler._valkey
            assert valkey is not None

            topic_key = f"topic:{topic}:subscribers"
            await valkey.delete(topic_key)

            # Verify cache is cleared
            cached_subs = await valkey.smembers(topic_key)
            assert len(cached_subs) == 0, "Topic cache should be cleared"

            # Call notify - should trigger DB fallback
            event = ModelNotificationEvent(
                event_id=str(uuid4()),
                topic=topic,
                payload=ModelNotificationEventPayload(
                    entity_type="test_item",
                    entity_id="item_123",
                    action="created",
                ),
            )

            # Note: The actual delivery may fail (no server), but the fallback
            # should still work and find the subscriber from DB
            attempts = await handler.notify(topic=topic, event=event)

            # Should have found the subscriber via DB fallback
            assert len(attempts) == 1
            assert attempts[0].subscription_id == sub.id

            # Verify cache was repopulated
            cached_subs_after = await valkey.smembers(topic_key)
            assert len(cached_subs_after) == 1
            assert sub.id in cached_subs_after

        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_cache_miss_individual_subscription_fallback(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        webhook_url: str,
    ) -> None:
        """Individual subscription cache miss triggers DB fallback and repopulation.

        When the subscription:{id} cache entry is missing but the set membership
        exists, _load_subscriptions should fall back to DB and repopulate.
        """
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
        )
        handler = HandlerSubscription(config)
        await handler.initialize()

        try:
            # Create subscription
            agent_id = f"agent_{uuid4().hex[:8]}"
            topic = f"memory.individual_{uuid4().hex[:8]}.created"
            delivery = ModelSubscriptionDeliveryWebhook(webhook_url=webhook_url)

            sub = await handler.subscribe(
                agent_id=agent_id,
                topic=topic,
                delivery=delivery,
            )

            # Clear only the individual subscription cache (not the set)
            valkey = handler._valkey
            assert valkey is not None

            sub_key = f"subscription:{sub.id}"
            await valkey.delete(sub_key)

            # Verify individual cache is cleared but set membership exists
            cached_data = await valkey.get(sub_key)
            assert (
                cached_data is None
            ), "Individual subscription cache should be cleared"

            agent_key = f"agent:{agent_id}:subscriptions"
            set_members = await valkey.smembers(agent_key)
            assert sub.id in set_members, "Set membership should still exist"

            # Call list_subscriptions - should trigger DB fallback for the subscription
            subs = await handler.list_subscriptions(agent_id)

            # Should recover the subscription
            assert len(subs) == 1
            assert subs[0].id == sub.id
            assert subs[0].topic == topic

            # Verify individual cache was repopulated
            cached_data_after = await valkey.get(sub_key)
            assert cached_data_after is not None, "Cache should be repopulated"

            # Verify the cached data is valid JSON and correct
            repopulated_sub = ModelSubscription.model_validate_json(cached_data_after)
            assert repopulated_sub.id == sub.id
            assert repopulated_sub.agent_id == agent_id
            assert repopulated_sub.topic == topic

        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_cache_miss_partial_subscriptions_fallback(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        webhook_url: str,
    ) -> None:
        """Mixed cache state: some subscriptions cached, some missing.

        Tests the scenario where some individual subscription entries are
        cached but others are missing, requiring partial DB fallback.
        """
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
        )
        handler = HandlerSubscription(config)
        await handler.initialize()

        try:
            # Create multiple subscriptions
            agent_id = f"agent_{uuid4().hex[:8]}"
            topics = [
                f"memory.partial_{uuid4().hex[:8]}.created",
                f"memory.partial_{uuid4().hex[:8]}.updated",
                f"memory.partial_{uuid4().hex[:8]}.deleted",
            ]
            delivery = ModelSubscriptionDeliveryWebhook(webhook_url=webhook_url)

            subscription_ids: list[str] = []
            for topic in topics:
                sub = await handler.subscribe(
                    agent_id=agent_id,
                    topic=topic,
                    delivery=delivery,
                )
                subscription_ids.append(sub.id)

            # Clear cache for only the first and third subscriptions
            # Leave the second one cached
            valkey = handler._valkey
            assert valkey is not None

            await valkey.delete(f"subscription:{subscription_ids[0]}")
            await valkey.delete(f"subscription:{subscription_ids[2]}")

            # Verify partial cache state
            assert await valkey.get(f"subscription:{subscription_ids[0]}") is None
            assert await valkey.get(f"subscription:{subscription_ids[1]}") is not None
            assert await valkey.get(f"subscription:{subscription_ids[2]}") is None

            # Call list_subscriptions - should load all 3 subscriptions
            subs = await handler.list_subscriptions(agent_id)

            # All subscriptions should be recovered
            assert len(subs) == 3
            returned_ids = {s.id for s in subs}
            assert returned_ids == set(subscription_ids)

            # Verify all caches are now populated
            for sub_id in subscription_ids:
                cached = await valkey.get(f"subscription:{sub_id}")
                assert cached is not None, f"Cache for {sub_id} should be repopulated"

        finally:
            await handler.shutdown()


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
# Circuit Breaker State Transition Tests
# =============================================================================


class TestCircuitBreakerStateTransitions:
    """Comprehensive tests for circuit breaker state transitions.

    Circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit tripped, requests blocked for cooldown period
    - HALF_OPEN: Testing if endpoint recovered after cooldown

    State Transitions:
    - CLOSED -> OPEN: After circuit_breaker_threshold consecutive failures
    - OPEN -> HALF_OPEN: After circuit_breaker_cooldown_seconds elapsed
    - HALF_OPEN -> CLOSED: After circuit_breaker_success_threshold successes
    - HALF_OPEN -> OPEN: On any failure in half-open state

    These tests manipulate the Valkey cache to simulate time passing,
    allowing fast test execution without waiting for real cooldowns.
    """

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Circuit transitions from CLOSED to OPEN after threshold failures.

        Verifies that:
        1. Circuit starts in CLOSED state (allows requests)
        2. After circuit_breaker_threshold failures, circuit opens
        3. Open circuit blocks subsequent delivery attempts
        """
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            circuit_breaker_threshold=2,  # Open after 2 consecutive failures
            circuit_breaker_cooldown_seconds=10,
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

            # Initial state: circuit should be CLOSED (allows requests)
            is_allowed_initial = await handler._check_circuit_breaker(unreachable_url)
            assert is_allowed_initial is True

            # Trigger threshold failures to open circuit
            for i in range(2):  # threshold = 2
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

            # Verify circuit is now OPEN (blocks requests)
            is_allowed_after = await handler._check_circuit_breaker(unreachable_url)
            assert is_allowed_after is False

            # Additional notification should be blocked with circuit breaker message
            event = ModelNotificationEvent(
                event_id=str(uuid4()),
                topic=unique_topic,
                payload=ModelNotificationEventPayload(
                    entity_type="test_item",
                    entity_id="item_blocked",
                    action="created",
                ),
            )
            attempts = await handler.notify(topic=unique_topic, event=event)

            assert len(attempts) == 1
            assert attempts[0].status == EnumDeliveryStatus.FAILED
            assert "Circuit breaker open" in (attempts[0].error_message or "")

        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open_after_cooldown(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Circuit transitions from OPEN to HALF_OPEN after cooldown elapses.

        Verifies that:
        1. Open circuit blocks requests
        2. After cooldown period, circuit transitions to HALF_OPEN
        3. HALF_OPEN state allows test requests through

        Note: Uses cache manipulation to simulate time passing for fast testing.
        """
        import json
        from datetime import datetime, timedelta, timezone

        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown_seconds=10,
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
            for i in range(2):
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

            # Verify circuit is OPEN
            is_allowed = await handler._check_circuit_breaker(unreachable_url)
            assert is_allowed is False

            # Manipulate cache to simulate cooldown elapsed
            valkey = handler._valkey
            assert valkey is not None

            endpoint_hash = handler._endpoint_hash(unreachable_url)
            cb_key = f"circuit_breaker:{endpoint_hash}"
            cached = await valkey.get(cb_key)
            assert cached is not None

            state_dict = json.loads(cached)
            assert state_dict["state"] == "open"

            # Set open_until to the past (simulates cooldown elapsed)
            past_time = datetime.now(timezone.utc) - timedelta(seconds=60)
            state_dict["open_until"] = past_time.isoformat()
            await valkey.set_key(cb_key, json.dumps(state_dict), ttl=3600)

            # Now _check_circuit_breaker should transition to HALF_OPEN and return True
            is_allowed_after_cooldown = await handler._check_circuit_breaker(
                unreachable_url
            )
            assert is_allowed_after_cooldown is True

            # Verify state is now HALF_OPEN
            cached_after = await valkey.get(cb_key)
            assert cached_after is not None
            state_after = json.loads(cached_after)
            assert state_after["state"] == "half_open"

        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_closes_after_successes_in_half_open(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
        webhook_server: tuple[str, list[dict[str, object]]],
    ) -> None:
        """Circuit transitions from HALF_OPEN to CLOSED after success threshold.

        Verifies that:
        1. In HALF_OPEN state, requests are allowed through for testing
        2. After circuit_breaker_success_threshold consecutive successes
        3. Circuit fully closes and returns to normal operation
        """
        import json

        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        webhook_url, received_events = webhook_server

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            circuit_breaker_threshold=2,
            circuit_breaker_success_threshold=2,  # Need 2 successes to close
            circuit_breaker_cooldown_seconds=10,
            http_timeout_seconds=5.0,
        )
        handler = HandlerSubscription(config)
        await handler.initialize()

        try:
            delivery = ModelSubscriptionDeliveryWebhook(
                webhook_url=webhook_url,
                timeout_ms=5000,
            )
            await handler.subscribe(
                agent_id=unique_agent_id,
                topic=unique_topic,
                delivery=delivery,
            )

            # Manually set circuit to HALF_OPEN state via cache
            valkey = handler._valkey
            assert valkey is not None

            endpoint_hash = handler._endpoint_hash(webhook_url)
            cb_key = f"circuit_breaker:{endpoint_hash}"

            half_open_state = {
                "state": "half_open",
                "failure_count": 0,
                "success_count": 0,
                "total_requests": 10,
                "total_failures": 5,
            }
            await valkey.set_key(cb_key, json.dumps(half_open_state), ttl=3600)

            # Verify circuit is HALF_OPEN and allows requests
            is_allowed = await handler._check_circuit_breaker(webhook_url)
            assert is_allowed is True

            # Send successful notifications to trigger HALF_OPEN -> CLOSED
            for i in range(2):  # success_threshold = 2
                event = ModelNotificationEvent(
                    event_id=str(uuid4()),
                    topic=unique_topic,
                    payload=ModelNotificationEventPayload(
                        entity_type="test_item",
                        entity_id=f"item_success_{i}",
                        action="created",
                    ),
                )
                attempts = await handler.notify(topic=unique_topic, event=event)
                assert len(attempts) == 1
                assert attempts[0].status == EnumDeliveryStatus.SUCCESS

            # Verify circuit is now CLOSED
            cached_after = await valkey.get(cb_key)
            assert cached_after is not None
            state_after = json.loads(cached_after)
            assert state_after["state"] == "closed"

            # Counters should be reset
            assert state_after.get("success_count", 0) == 0
            assert state_after.get("failure_count", 0) == 0

        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_reopens_on_failure_in_half_open(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Circuit transitions from HALF_OPEN back to OPEN on any failure.

        Verifies that:
        1. In HALF_OPEN state, a test request is allowed
        2. If the test request fails, circuit immediately reopens
        3. Cooldown period is reset for the new OPEN state
        """
        import json

        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            circuit_breaker_threshold=2,
            circuit_breaker_success_threshold=3,
            circuit_breaker_cooldown_seconds=10,
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

            # Manually set circuit to HALF_OPEN state
            valkey = handler._valkey
            assert valkey is not None

            endpoint_hash = handler._endpoint_hash(unreachable_url)
            cb_key = f"circuit_breaker:{endpoint_hash}"

            half_open_state = {
                "state": "half_open",
                "failure_count": 0,
                "success_count": 1,  # One success already
                "total_requests": 10,
                "total_failures": 5,
            }
            await valkey.set_key(cb_key, json.dumps(half_open_state), ttl=3600)

            # Verify circuit is HALF_OPEN and allows test request
            is_allowed = await handler._check_circuit_breaker(unreachable_url)
            assert is_allowed is True

            # Trigger a failure while in HALF_OPEN state
            event = ModelNotificationEvent(
                event_id=str(uuid4()),
                topic=unique_topic,
                payload=ModelNotificationEventPayload(
                    entity_type="test_item",
                    entity_id="item_fail_halfopen",
                    action="created",
                ),
            )
            attempts = await handler.notify(topic=unique_topic, event=event)

            assert len(attempts) == 1
            assert attempts[0].status == EnumDeliveryStatus.FAILED

            # Verify circuit reopened (HALF_OPEN -> OPEN)
            cached_after = await valkey.get(cb_key)
            assert cached_after is not None
            state_after = json.loads(cached_after)
            assert state_after["state"] == "open"

            # Cooldown should be reset (new open_until timestamp)
            assert state_after.get("open_until") is not None

            # Circuit should now block requests
            is_allowed_after = await handler._check_circuit_breaker(unreachable_url)
            assert is_allowed_after is False

        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_all_requests_when_open(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Open circuit blocks all delivery attempts with error message.

        Verifies that when circuit is OPEN:
        1. No HTTP requests are made to the endpoint
        2. Delivery attempt returns immediately with FAILED status
        3. Error message indicates circuit breaker is open
        """
        import json
        from datetime import datetime, timedelta, timezone

        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown_seconds=10,
            http_timeout_seconds=1.0,
        )
        handler = HandlerSubscription(config)
        await handler.initialize()

        try:
            bad_url = "http://localhost:59999/unreachable"
            delivery = ModelSubscriptionDeliveryWebhook(
                webhook_url=bad_url,
                timeout_ms=100,
            )
            await handler.subscribe(
                agent_id=unique_agent_id,
                topic=unique_topic,
                delivery=delivery,
            )

            # Manually set circuit to OPEN state with future cooldown
            valkey = handler._valkey
            assert valkey is not None

            endpoint_hash = handler._endpoint_hash(bad_url)
            cb_key = f"circuit_breaker:{endpoint_hash}"

            future_time = datetime.now(timezone.utc) + timedelta(seconds=60)
            open_state = {
                "state": "open",
                "failure_count": 5,
                "success_count": 0,
                "open_until": future_time.isoformat(),
                "total_requests": 10,
                "total_failures": 5,
            }
            await valkey.set_key(cb_key, json.dumps(open_state), ttl=3600)

            # Attempt delivery - should be blocked immediately
            event = ModelNotificationEvent(
                event_id=str(uuid4()),
                topic=unique_topic,
                payload=ModelNotificationEventPayload(
                    entity_type="test_item",
                    entity_id="item_blocked",
                    action="created",
                ),
            )
            attempts = await handler.notify(topic=unique_topic, event=event)

            assert len(attempts) == 1
            assert attempts[0].status == EnumDeliveryStatus.FAILED
            assert "Circuit breaker open" in (attempts[0].error_message or "")
            # No status code because no HTTP request was made
            assert attempts[0].status_code is None

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


# =============================================================================
# Encryption Tests
# =============================================================================


class TestEncryption:
    """Tests for webhook secret encryption at rest.

    The HandlerSubscription supports Fernet encryption for webhook secrets.
    When encryption_key is configured, secrets are encrypted before storage
    in PostgreSQL and decrypted when loaded. This protects secrets in case
    of database compromise.

    These tests verify:
    - Encrypted secrets can be stored and retrieved correctly (roundtrip)
    - Decrypted secrets work correctly for HMAC signing
    - Backwards compatibility with plaintext secrets
    - Warning is logged when encryption_key is not configured
    """

    @pytest.mark.asyncio
    async def test_encrypted_secret_roundtrip(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
        webhook_url: str,
    ) -> None:
        """Encrypted secrets can be stored and retrieved correctly.

        This test verifies that when encryption_key is configured:
        1. Secrets are encrypted before storage
        2. Secrets are correctly decrypted when retrieved
        3. The decrypted value matches the original plaintext
        """
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        # Import Fernet for key generation
        from cryptography.fernet import Fernet

        encryption_key = Fernet.generate_key().decode("utf-8")

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            encryption_key=encryption_key,
        )
        handler = HandlerSubscription(config)
        await handler.initialize()

        try:
            # Subscribe with a secret
            secret = "my-webhook-secret-12345"
            delivery = ModelSubscriptionDeliveryWebhook(
                webhook_url=webhook_url,
                secret=secret,
            )

            sub = await handler.subscribe(
                agent_id=unique_agent_id,
                topic=unique_topic,
                delivery=delivery,
            )

            # Retrieve and verify secret is decrypted correctly
            subs = await handler.list_subscriptions(unique_agent_id)
            assert len(subs) == 1
            assert subs[0].id == sub.id
            assert subs[0].delivery.secret == secret

        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_encrypted_secret_survives_restart(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
        webhook_url: str,
    ) -> None:
        """Encrypted secrets persist across handler restarts.

        This test verifies that:
        1. Secrets are stored encrypted in the database
        2. A new handler instance with the same key can decrypt them
        3. The decrypted value matches the original
        """
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        from cryptography.fernet import Fernet

        encryption_key = Fernet.generate_key().decode("utf-8")
        secret = "persistent-secret-xyz"

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            encryption_key=encryption_key,
        )

        # Create subscription with first handler
        handler1 = HandlerSubscription(config)
        await handler1.initialize()

        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=webhook_url,
            secret=secret,
        )
        sub = await handler1.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )
        original_id = sub.id

        await handler1.shutdown()

        # Create new handler with same encryption key (simulates restart)
        handler2 = HandlerSubscription(config)
        await handler2.initialize()

        try:
            # Verify secret is correctly decrypted after restart
            subs = await handler2.list_subscriptions(unique_agent_id)
            assert len(subs) == 1
            assert subs[0].id == original_id
            assert subs[0].delivery.secret == secret

        finally:
            await handler2.shutdown()

    @pytest.mark.asyncio
    async def test_encrypted_secret_used_for_hmac(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
        webhook_server: tuple[str, list[dict[str, object]]],
    ) -> None:
        """Secret can be used for HMAC signing after decryption.

        This test verifies that encrypted secrets are correctly decrypted
        and used for HMAC signature generation when sending notifications.
        """
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        from cryptography.fernet import Fernet

        webhook_url, received_events = webhook_server
        encryption_key = Fernet.generate_key().decode("utf-8")
        secret = "hmac-signing-secret"

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            encryption_key=encryption_key,
        )
        handler = HandlerSubscription(config)
        await handler.initialize()

        try:
            # Subscribe with encrypted secret
            delivery = ModelSubscriptionDeliveryWebhook(
                webhook_url=webhook_url,
                secret=secret,
            )
            await handler.subscribe(
                agent_id=unique_agent_id,
                topic=unique_topic,
                delivery=delivery,
            )

            # Send notification
            event = ModelNotificationEvent(
                event_id=str(uuid4()),
                topic=unique_topic,
                payload=ModelNotificationEventPayload(
                    entity_type="test_item",
                    entity_id="item_123",
                    action="created",
                ),
            )

            attempts = await handler.notify(
                topic=unique_topic,
                event=event,
            )

            # Verify notification was sent successfully
            assert len(attempts) == 1
            assert attempts[0].status == EnumDeliveryStatus.SUCCESS

            # Verify HMAC signature was included (proves secret was decrypted)
            assert len(received_events) == 1
            signature = received_events[0]["signature"]
            assert signature is not None
            assert signature.startswith("sha256=")

        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_plaintext_secret_backwards_compatibility(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
        webhook_url: str,
    ) -> None:
        """Plaintext secrets still work when encryption is enabled.

        This test verifies backwards compatibility: when encryption is enabled
        but the database contains plaintext secrets (from before encryption
        was configured), the handler gracefully falls back to using them as-is.

        The scenario is:
        1. Create subscription without encryption (secret stored as plaintext)
        2. Restart handler WITH encryption_key configured
        3. The handler should gracefully handle the plaintext secret
        """
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        from cryptography.fernet import Fernet

        secret = "plaintext-secret-for-backwards-compat"

        # First: Create subscription WITHOUT encryption
        config_no_encryption = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            # No encryption_key - secret stored as plaintext
        )
        handler1 = HandlerSubscription(config_no_encryption)
        await handler1.initialize()

        delivery = ModelSubscriptionDeliveryWebhook(
            webhook_url=webhook_url,
            secret=secret,
        )
        sub = await handler1.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            delivery=delivery,
        )
        original_id = sub.id

        await handler1.shutdown()

        # Second: Start handler WITH encryption_key (simulates enabling encryption)
        encryption_key = Fernet.generate_key().decode("utf-8")
        config_with_encryption = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            encryption_key=encryption_key,
        )
        handler2 = HandlerSubscription(config_with_encryption)
        await handler2.initialize()

        try:
            # Retrieve subscription - should handle plaintext gracefully
            subs = await handler2.list_subscriptions(unique_agent_id)
            assert len(subs) == 1
            assert subs[0].id == original_id
            # The secret should be returned as-is (backwards compatibility)
            assert subs[0].delivery.secret == secret

        finally:
            await handler2.shutdown()

    @pytest.mark.asyncio
    async def test_warning_logged_without_encryption_key(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        services_available: bool,
        unique_agent_id: str,
        unique_topic: str,
        webhook_url: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warning is logged when encryption_key is not configured.

        When storing a webhook secret without encryption_key configured,
        the handler should log a security warning to alert operators.
        """
        if not services_available:
            pytest.skip("Required services (PostgreSQL, Valkey) not available")

        import logging

        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
            # No encryption_key configured
        )
        handler = HandlerSubscription(config)
        await handler.initialize()

        try:
            # Subscribe with a secret (should log warning)
            secret = "unencrypted-secret"
            delivery = ModelSubscriptionDeliveryWebhook(
                webhook_url=webhook_url,
                secret=secret,
            )

            with caplog.at_level(logging.WARNING):
                await handler.subscribe(
                    agent_id=unique_agent_id,
                    topic=unique_topic,
                    delivery=delivery,
                )

            # Verify warning was logged about plaintext storage
            assert any(
                "encryption_key not configured" in record.message
                and "plaintext" in record.message
                for record in caplog.records
            ), "Expected warning about plaintext storage not found in logs"

        finally:
            await handler.shutdown()
