# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerSubscription with Kafka-based notification.

This module tests the subscription handler which manages agent subscriptions
and publishes notification events to Kafka for subscriber consumption.

Test Categories:
    - TestSubscribe: Create subscription, topic validation, idempotency, metadata
    - TestUnsubscribe: Remove subscription, nonexistent returns False
    - TestListSubscriptions: List all, empty for new agent
    - TestNotify: Returns subscriber count, no subscribers returns 0
    - TestSurviveRestart: Subscriptions persist across handler restart
    - TestHealthCheck: Returns component status
    - TestMetrics: Returns counters

Prerequisites:
    - PostgreSQL running at TEST_DB_DSN
    - Valkey running at TEST_VALKEY_HOST:TEST_VALKEY_PORT
    - Kafka running for notify tests (graceful skip if unavailable)
    - omnibase_infra installed (dev dependency)

Usage:
    # Run subscription tests
    pytest tests/integration/test_handler_subscription.py -v

    # Run with markers
    pytest -m "integration and subscription" -v

Environment Variables:
    TEST_DB_DSN: PostgreSQL connection string
    TEST_VALKEY_HOST: Valkey hostname (default: localhost)
    TEST_VALKEY_PORT: Valkey port (default: 6379)
    TEST_KAFKA_BOOTSTRAP_SERVERS: Kafka servers (default: localhost:9092)

.. versionadded:: 0.2.0
    Refactored for Kafka-based notification (OMN-1393).
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest

# Check if dependencies are available
_DEPENDENCIES_AVAILABLE = False
_SKIP_REASON = "Required dependencies not installed"

try:
    from omnimemory.enums.enum_subscription_status import EnumSubscriptionStatus
    from omnimemory.handlers import (
        HandlerSubscription,
        ModelHandlerSubscriptionConfig,
        ModelSubscriptionHealth,
        ModelSubscriptionMetrics,
    )
    from omnimemory.models.subscription import (
        ModelNotificationEvent,
        ModelNotificationEventPayload,
        ModelSubscription,
    )

    _DEPENDENCIES_AVAILABLE = True
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
        not _DEPENDENCIES_AVAILABLE,
        reason=_SKIP_REASON,
    ),
]


# =============================================================================
# Test Configuration
# =============================================================================

DEFAULT_KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"


def get_test_kafka_bootstrap_servers() -> str:
    """Get Kafka bootstrap servers from environment or default.

    Returns:
        Kafka bootstrap servers string.
    """
    return os.environ.get(
        "TEST_KAFKA_BOOTSTRAP_SERVERS",
        DEFAULT_KAFKA_BOOTSTRAP_SERVERS,
    )


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
    """Create and initialize subscription handler for tests.

    Yields:
        Initialized HandlerSubscription instance.
    """
    if not services_available:
        pytest.skip("Required services (PostgreSQL, Valkey) not available")

    config = ModelHandlerSubscriptionConfig(
        db_dsn=test_db_dsn,
        valkey_host=test_valkey_host,
        valkey_port=test_valkey_port,
        kafka_bootstrap_servers=get_test_kafka_bootstrap_servers(),
    )
    handler = HandlerSubscription(config)

    try:
        await handler.initialize()
    except RuntimeError as e:
        pytest.skip(f"Failed to initialize handler: {e}")

    yield handler

    await handler.shutdown()


@pytest.fixture
def handler_config(
    test_db_dsn: str,
    test_valkey_host: str,
    test_valkey_port: int,
) -> ModelHandlerSubscriptionConfig:
    """Provide handler configuration for tests.

    Returns:
        Handler configuration.
    """
    return ModelHandlerSubscriptionConfig(
        db_dsn=test_db_dsn,
        valkey_host=test_valkey_host,
        valkey_port=test_valkey_port,
        kafka_bootstrap_servers=get_test_kafka_bootstrap_servers(),
    )


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


@pytest.fixture
def sample_event(unique_topic: str) -> ModelNotificationEvent:
    """Create a sample notification event for testing.

    Args:
        unique_topic: The topic for the event.

    Returns:
        Sample ModelNotificationEvent.
    """
    return ModelNotificationEvent(
        event_id=str(uuid4()),
        topic=unique_topic,
        payload=ModelNotificationEventPayload(
            entity_type="test_item",
            entity_id=f"item_{uuid4().hex[:8]}",
            action="created",
        ),
    )


# =============================================================================
# TestSubscribe
# =============================================================================


class TestSubscribe:
    """Tests for subscribe() method."""

    @pytest.mark.asyncio
    async def test_subscribe_creates_subscription(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Subscribe creates a new subscription."""
        subscription = await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        assert subscription is not None
        assert isinstance(subscription, ModelSubscription)
        assert subscription.agent_id == unique_agent_id
        assert subscription.topic == unique_topic
        assert subscription.status == EnumSubscriptionStatus.ACTIVE
        assert subscription.id is not None
        assert subscription.created_at is not None

    @pytest.mark.asyncio
    async def test_subscribe_with_metadata(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Subscribe stores metadata correctly."""
        metadata = {"source": "test", "priority": "high"}

        subscription = await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
            metadata=metadata,
        )

        assert subscription.metadata == metadata

    @pytest.mark.asyncio
    async def test_subscribe_idempotent(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Subscribing twice to same topic is idempotent (upsert behavior)."""
        # First subscription
        sub1 = await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        # Second subscription to same topic
        sub2 = await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        # Should return same subscription ID (upsert)
        assert sub1.id == sub2.id
        assert sub2.status == EnumSubscriptionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_subscribe_invalid_topic_raises(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
    ) -> None:
        """Subscribe with invalid topic format raises ValueError."""
        with pytest.raises(ValueError, match="Topic must match pattern"):
            await subscription_handler.subscribe(
                agent_id=unique_agent_id,
                topic="invalid-topic-format",
            )

    @pytest.mark.asyncio
    async def test_subscribe_multiple_topics(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
    ) -> None:
        """Agent can subscribe to multiple topics."""
        topics = [
            f"memory.test_{uuid4().hex[:8]}.created",
            f"memory.test_{uuid4().hex[:8]}.updated",
            f"memory.test_{uuid4().hex[:8]}.deleted",
        ]

        subscriptions = []
        for topic in topics:
            sub = await subscription_handler.subscribe(
                agent_id=unique_agent_id,
                topic=topic,
            )
            subscriptions.append(sub)

        assert len(subscriptions) == 3
        assert all(s.agent_id == unique_agent_id for s in subscriptions)
        assert {s.topic for s in subscriptions} == set(topics)


# =============================================================================
# TestUnsubscribe
# =============================================================================


class TestUnsubscribe:
    """Tests for unsubscribe() method."""

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_subscription(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Unsubscribe removes an existing subscription."""
        # First subscribe
        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        # Then unsubscribe
        result = await subscription_handler.unsubscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        assert result is True

        # Verify subscription is gone
        subscriptions = await subscription_handler.list_subscriptions(unique_agent_id)
        assert not any(s.topic == unique_topic for s in subscriptions)

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_returns_false(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
    ) -> None:
        """Unsubscribe for non-existent subscription returns False."""
        result = await subscription_handler.unsubscribe(
            agent_id=unique_agent_id,
            topic="memory.nonexistent.topic",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_unsubscribe_only_affects_specified_topic(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
    ) -> None:
        """Unsubscribe only removes the specified topic subscription."""
        topic1 = f"memory.test_{uuid4().hex[:8]}.created"
        topic2 = f"memory.test_{uuid4().hex[:8]}.updated"

        # Subscribe to both topics
        await subscription_handler.subscribe(agent_id=unique_agent_id, topic=topic1)
        await subscription_handler.subscribe(agent_id=unique_agent_id, topic=topic2)

        # Unsubscribe from topic1 only
        await subscription_handler.unsubscribe(agent_id=unique_agent_id, topic=topic1)

        # topic2 should still exist
        subscriptions = await subscription_handler.list_subscriptions(unique_agent_id)
        assert len(subscriptions) == 1
        assert subscriptions[0].topic == topic2


# =============================================================================
# TestListSubscriptions
# =============================================================================


class TestListSubscriptions:
    """Tests for list_subscriptions() method."""

    @pytest.mark.asyncio
    async def test_list_subscriptions_returns_all(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
    ) -> None:
        """List subscriptions returns all agent subscriptions."""
        topics = [
            f"memory.test_{uuid4().hex[:8]}.created",
            f"memory.test_{uuid4().hex[:8]}.updated",
        ]

        for topic in topics:
            await subscription_handler.subscribe(
                agent_id=unique_agent_id,
                topic=topic,
            )

        subscriptions = await subscription_handler.list_subscriptions(unique_agent_id)

        assert len(subscriptions) == 2
        assert all(isinstance(s, ModelSubscription) for s in subscriptions)
        assert {s.topic for s in subscriptions} == set(topics)

    @pytest.mark.asyncio
    async def test_list_subscriptions_empty_for_new_agent(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
    ) -> None:
        """List subscriptions returns empty list for agent with no subscriptions."""
        subscriptions = await subscription_handler.list_subscriptions(unique_agent_id)

        assert subscriptions == []

    @pytest.mark.asyncio
    async def test_list_subscriptions_excludes_other_agents(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """List subscriptions only returns subscriptions for specified agent."""
        agent1 = f"test_agent_{uuid4().hex[:8]}"
        agent2 = f"test_agent_{uuid4().hex[:8]}"
        topic1 = f"memory.test_{uuid4().hex[:8]}.created"
        topic2 = f"memory.test_{uuid4().hex[:8]}.updated"

        await subscription_handler.subscribe(agent_id=agent1, topic=topic1)
        await subscription_handler.subscribe(agent_id=agent2, topic=topic2)

        agent1_subs = await subscription_handler.list_subscriptions(agent1)

        assert len(agent1_subs) == 1
        assert agent1_subs[0].topic == topic1
        assert agent1_subs[0].agent_id == agent1


# =============================================================================
# TestNotify
# =============================================================================


class TestNotify:
    """Tests for notify() method."""

    @pytest.mark.asyncio
    async def test_notify_returns_subscriber_count(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
        sample_event: ModelNotificationEvent,
    ) -> None:
        """Notify returns count of active subscribers."""
        # Subscribe agent
        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        try:
            count = await subscription_handler.notify(
                topic=unique_topic,
                event=sample_event,
            )
            assert count == 1
        except RuntimeError as e:
            # Kafka may not be available - skip gracefully
            if "Kafka" in str(e) or "kafka" in str(e).lower():
                pytest.skip(f"Kafka not available: {e}")
            raise

    @pytest.mark.asyncio
    async def test_notify_no_subscribers_returns_zero(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Notify with no subscribers returns 0."""
        topic = f"memory.orphan_{uuid4().hex[:8]}.created"
        event = ModelNotificationEvent(
            event_id=str(uuid4()),
            topic=topic,
            payload=ModelNotificationEventPayload(
                entity_type="orphan",
                entity_id="orphan_123",
                action="created",
            ),
        )

        try:
            count = await subscription_handler.notify(topic=topic, event=event)
            assert count == 0
        except RuntimeError as e:
            if "Kafka" in str(e) or "kafka" in str(e).lower():
                pytest.skip(f"Kafka not available: {e}")
            raise

    @pytest.mark.asyncio
    async def test_notify_topic_mismatch_raises(
        self,
        subscription_handler: HandlerSubscription,
        unique_topic: str,
    ) -> None:
        """Notify raises ValueError when event.topic does not match topic argument."""
        event = ModelNotificationEvent(
            event_id=str(uuid4()),
            topic="memory.different.topic",
            payload=ModelNotificationEventPayload(
                entity_type="test",
                entity_id="test_123",
                action="created",
            ),
        )

        with pytest.raises(ValueError, match="Event topic mismatch"):
            await subscription_handler.notify(topic=unique_topic, event=event)

    @pytest.mark.asyncio
    async def test_notify_multiple_subscribers(
        self,
        subscription_handler: HandlerSubscription,
        unique_topic: str,
    ) -> None:
        """Notify returns count of all subscribers for topic."""
        # Subscribe multiple agents to same topic
        agents = [f"test_agent_{uuid4().hex[:8]}" for _ in range(3)]
        for agent_id in agents:
            await subscription_handler.subscribe(
                agent_id=agent_id,
                topic=unique_topic,
            )

        event = ModelNotificationEvent(
            event_id=str(uuid4()),
            topic=unique_topic,
            payload=ModelNotificationEventPayload(
                entity_type="test",
                entity_id="test_123",
                action="created",
            ),
        )

        try:
            count = await subscription_handler.notify(topic=unique_topic, event=event)
            assert count == 3
        except RuntimeError as e:
            if "Kafka" in str(e) or "kafka" in str(e).lower():
                pytest.skip(f"Kafka not available: {e}")
            raise


# =============================================================================
# TestSurviveRestart
# =============================================================================


class TestSurviveRestart:
    """Tests for subscription persistence across handler restart."""

    @pytest.mark.asyncio
    async def test_subscriptions_persist_across_restart(
        self,
        handler_config: ModelHandlerSubscriptionConfig,
        services_available: bool,
    ) -> None:
        """Subscriptions survive handler shutdown and restart."""
        if not services_available:
            pytest.skip("Required services not available")

        agent_id = f"test_agent_{uuid4().hex[:8]}"
        topic = f"memory.test_{uuid4().hex[:8]}.created"

        # Create handler and subscription
        handler1 = HandlerSubscription(handler_config)
        try:
            await handler1.initialize()
        except RuntimeError as e:
            pytest.skip(f"Failed to initialize handler: {e}")

        await handler1.subscribe(agent_id=agent_id, topic=topic)
        await handler1.shutdown()

        # Create new handler instance and verify subscription exists
        handler2 = HandlerSubscription(handler_config)
        try:
            await handler2.initialize()
        except RuntimeError as e:
            pytest.skip(f"Failed to initialize handler: {e}")

        try:
            subscriptions = await handler2.list_subscriptions(agent_id)

            assert len(subscriptions) == 1
            assert subscriptions[0].topic == topic
            assert subscriptions[0].agent_id == agent_id
            assert subscriptions[0].status == EnumSubscriptionStatus.ACTIVE
        finally:
            await handler2.shutdown()

    @pytest.mark.asyncio
    async def test_deleted_subscriptions_stay_deleted_after_restart(
        self,
        handler_config: ModelHandlerSubscriptionConfig,
        services_available: bool,
    ) -> None:
        """Deleted subscriptions remain deleted after restart."""
        if not services_available:
            pytest.skip("Required services not available")

        agent_id = f"test_agent_{uuid4().hex[:8]}"
        topic = f"memory.test_{uuid4().hex[:8]}.created"

        # Create, subscribe, then unsubscribe
        handler1 = HandlerSubscription(handler_config)
        try:
            await handler1.initialize()
        except RuntimeError as e:
            pytest.skip(f"Failed to initialize handler: {e}")

        await handler1.subscribe(agent_id=agent_id, topic=topic)
        await handler1.unsubscribe(agent_id=agent_id, topic=topic)
        await handler1.shutdown()

        # Verify subscription stays deleted
        handler2 = HandlerSubscription(handler_config)
        try:
            await handler2.initialize()
        except RuntimeError as e:
            pytest.skip(f"Failed to initialize handler: {e}")

        try:
            subscriptions = await handler2.list_subscriptions(agent_id)
            assert len(subscriptions) == 0
        finally:
            await handler2.shutdown()


# =============================================================================
# TestHealthCheck
# =============================================================================


class TestHealthCheck:
    """Tests for health_check() method."""

    @pytest.mark.asyncio
    async def test_health_check_returns_status(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Health check returns component status."""
        health = await subscription_handler.health_check()

        assert isinstance(health, ModelSubscriptionHealth)
        assert health.initialized is True
        assert health.db_healthy is not None
        assert health.valkey_healthy is not None
        assert health.kafka_healthy is not None

    @pytest.mark.asyncio
    async def test_health_check_before_initialize(
        self,
        handler_config: ModelHandlerSubscriptionConfig,
    ) -> None:
        """Health check before initialize returns uninitialized status."""
        handler = HandlerSubscription(handler_config)

        health = await handler.health_check()

        assert health.is_healthy is False
        assert health.initialized is False
        assert health.error_message == "Handler not initialized"

    @pytest.mark.asyncio
    async def test_health_check_includes_metrics(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Health check includes metrics in response."""
        health = await subscription_handler.health_check()

        assert health.metrics is not None
        assert isinstance(health.metrics, ModelSubscriptionMetrics)


# =============================================================================
# TestMetrics
# =============================================================================


class TestMetrics:
    """Tests for get_metrics() method."""

    @pytest.mark.asyncio
    async def test_get_metrics_returns_counters(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Get metrics returns counter values."""
        metrics = subscription_handler.get_metrics()

        assert isinstance(metrics, ModelSubscriptionMetrics)
        assert metrics.subscriptions_created >= 0
        assert metrics.subscriptions_deleted >= 0
        assert metrics.notifications_published >= 0

    @pytest.mark.asyncio
    async def test_metrics_increment_on_subscribe(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Subscribe increments subscriptions_created counter."""
        initial_metrics = subscription_handler.get_metrics()

        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        final_metrics = subscription_handler.get_metrics()

        assert (
            final_metrics.subscriptions_created
            == initial_metrics.subscriptions_created + 1
        )

    @pytest.mark.asyncio
    async def test_metrics_increment_on_unsubscribe(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Unsubscribe increments subscriptions_deleted counter."""
        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )
        initial_metrics = subscription_handler.get_metrics()

        await subscription_handler.unsubscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        final_metrics = subscription_handler.get_metrics()

        assert (
            final_metrics.subscriptions_deleted
            == initial_metrics.subscriptions_deleted + 1
        )

    @pytest.mark.asyncio
    async def test_metrics_increment_on_notify(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
        sample_event: ModelNotificationEvent,
    ) -> None:
        """Notify increments notifications_published counter."""
        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )
        initial_metrics = subscription_handler.get_metrics()

        try:
            await subscription_handler.notify(
                topic=unique_topic,
                event=sample_event,
            )
        except RuntimeError as e:
            if "Kafka" in str(e) or "kafka" in str(e).lower():
                pytest.skip(f"Kafka not available: {e}")
            raise

        final_metrics = subscription_handler.get_metrics()

        assert (
            final_metrics.notifications_published
            == initial_metrics.notifications_published + 1
        )


# =============================================================================
# TestInitialization
# =============================================================================


class TestInitialization:
    """Tests for handler initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_handler_not_initialized_raises(
        self,
        handler_config: ModelHandlerSubscriptionConfig,
    ) -> None:
        """Operations before initialize raise RuntimeError."""
        handler = HandlerSubscription(handler_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await handler.subscribe(
                agent_id="test_agent",
                topic="memory.test.created",
            )

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(
        self,
        handler_config: ModelHandlerSubscriptionConfig,
        services_available: bool,
    ) -> None:
        """Multiple initialize calls are safe (idempotent)."""
        if not services_available:
            pytest.skip("Required services not available")

        handler = HandlerSubscription(handler_config)
        try:
            await handler.initialize()
            await handler.initialize()  # Should not raise
            assert handler.is_initialized is True
        except RuntimeError as e:
            pytest.skip(f"Failed to initialize handler: {e}")
        finally:
            await handler.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(
        self,
        handler_config: ModelHandlerSubscriptionConfig,
        services_available: bool,
    ) -> None:
        """Multiple shutdown calls are safe (idempotent)."""
        if not services_available:
            pytest.skip("Required services not available")

        handler = HandlerSubscription(handler_config)
        try:
            await handler.initialize()
        except RuntimeError as e:
            pytest.skip(f"Failed to initialize handler: {e}")

        await handler.shutdown()
        await handler.shutdown()  # Should not raise
        assert handler.is_initialized is False
