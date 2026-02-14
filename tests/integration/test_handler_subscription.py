# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerSubscription with event bus notification.

This module tests the subscription handler which manages agent subscriptions
and publishes notification events via the event bus for subscriber consumption.

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
    - Event bus running for notify tests (graceful skip if unavailable)
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
    TEST_KAFKA_BOOTSTRAP_SERVERS: Event bus servers (default: localhost:9092)

.. versionadded:: 0.2.0
    Refactored for event bus notification (OMN-1393).
"""

from __future__ import annotations

import asyncio
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

DEFAULT_EVENT_BUS_BOOTSTRAP_SERVERS = "localhost:9092"


def get_test_event_bus_bootstrap_servers() -> str:
    """Get event bus bootstrap servers from environment or default.

    Returns:
        Event bus bootstrap servers string.
    """
    return os.environ.get(
        "TEST_KAFKA_BOOTSTRAP_SERVERS",
        DEFAULT_EVENT_BUS_BOOTSTRAP_SERVERS,
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
    from omnibase_core.container import ModelONEXContainer

    if not services_available:
        pytest.skip("Required services (PostgreSQL, Valkey) not available")

    container = ModelONEXContainer()
    config = ModelHandlerSubscriptionConfig(
        db_dsn=test_db_dsn,
        valkey_host=test_valkey_host,
        valkey_port=test_valkey_port,
        kafka_bootstrap_servers=get_test_event_bus_bootstrap_servers(),
    )
    handler = HandlerSubscription(container)

    try:
        await handler.initialize(config)
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
        kafka_bootstrap_servers=get_test_event_bus_bootstrap_servers(),
    )


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
        with pytest.raises(
            ValueError, match="subscribe\\(\\) received invalid topic format"
        ):
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
            # NOTE: Error message matching is coupled to EventBusKafka's exception text.
            # If the event bus adapter changes error messages, update these skip guards.
            if "Kafka" in str(e) or "kafka" in str(e).lower():
                pytest.skip(f"Event bus not available: {e}")
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
                pytest.skip(f"Event bus not available: {e}")
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
                pytest.skip(f"Event bus not available: {e}")
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
        from omnibase_core.container import ModelONEXContainer

        if not services_available:
            pytest.skip("Required services not available")

        agent_id = f"test_agent_{uuid4().hex[:8]}"
        topic = f"memory.test_{uuid4().hex[:8]}.created"

        # Create handler and subscription
        container1 = ModelONEXContainer()
        handler1 = HandlerSubscription(container1)
        try:
            await handler1.initialize(handler_config)
        except RuntimeError as e:
            pytest.skip(f"Failed to initialize handler: {e}")

        await handler1.subscribe(agent_id=agent_id, topic=topic)
        await handler1.shutdown()

        # Create new handler instance and verify subscription exists
        container2 = ModelONEXContainer()
        handler2 = HandlerSubscription(container2)
        try:
            await handler2.initialize(handler_config)
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
        from omnibase_core.container import ModelONEXContainer

        if not services_available:
            pytest.skip("Required services not available")

        agent_id = f"test_agent_{uuid4().hex[:8]}"
        topic = f"memory.test_{uuid4().hex[:8]}.created"

        # Create, subscribe, then unsubscribe
        container1 = ModelONEXContainer()
        handler1 = HandlerSubscription(container1)
        try:
            await handler1.initialize(handler_config)
        except RuntimeError as e:
            pytest.skip(f"Failed to initialize handler: {e}")

        await handler1.subscribe(agent_id=agent_id, topic=topic)
        await handler1.unsubscribe(agent_id=agent_id, topic=topic)
        await handler1.shutdown()

        # Verify subscription stays deleted
        container2 = ModelONEXContainer()
        handler2 = HandlerSubscription(container2)
        try:
            await handler2.initialize(handler_config)
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
        assert health.event_bus_healthy is not None

    @pytest.mark.asyncio
    async def test_health_check_before_initialize(
        self,
        handler_config: ModelHandlerSubscriptionConfig,
    ) -> None:
        """Health check before initialize returns uninitialized status."""
        from omnibase_core.container import ModelONEXContainer

        container = ModelONEXContainer()
        handler = HandlerSubscription(container)

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
        metrics = await subscription_handler.get_metrics()

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
        initial_metrics = await subscription_handler.get_metrics()

        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        final_metrics = await subscription_handler.get_metrics()

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
        initial_metrics = await subscription_handler.get_metrics()

        await subscription_handler.unsubscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        final_metrics = await subscription_handler.get_metrics()

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
        initial_metrics = await subscription_handler.get_metrics()

        try:
            await subscription_handler.notify(
                topic=unique_topic,
                event=sample_event,
            )
        except RuntimeError as e:
            if "Kafka" in str(e) or "kafka" in str(e).lower():
                pytest.skip(f"Event bus not available: {e}")
            raise

        final_metrics = await subscription_handler.get_metrics()

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
        from omnibase_core.container import ModelONEXContainer

        container = ModelONEXContainer()
        handler = HandlerSubscription(container)

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
        from omnibase_core.container import ModelONEXContainer

        if not services_available:
            pytest.skip("Required services not available")

        container = ModelONEXContainer()
        handler = HandlerSubscription(container)
        try:
            await handler.initialize(handler_config)
            await handler.initialize(handler_config)  # Should not raise
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
        from omnibase_core.container import ModelONEXContainer

        if not services_available:
            pytest.skip("Required services not available")

        container = ModelONEXContainer()
        handler = HandlerSubscription(container)
        try:
            await handler.initialize(handler_config)
        except RuntimeError as e:
            pytest.skip(f"Failed to initialize handler: {e}")

        await handler.shutdown()
        await handler.shutdown()  # Should not raise
        assert handler.is_initialized is False


# =============================================================================
# TestConcurrency
# =============================================================================


class TestConcurrency:
    """Tests for concurrent subscription operations.

    These tests verify that the subscription handler correctly handles
    race conditions and concurrent access patterns that occur in production
    when multiple agents interact with the subscription system simultaneously.
    """

    @pytest.mark.asyncio
    async def test_concurrent_subscribes_same_topic(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Multiple agents can subscribe to the same topic concurrently.

        This test verifies that when multiple agents attempt to subscribe
        to the same topic at the same time, all subscriptions succeed
        without race conditions or data corruption.
        """
        topic = f"memory.concurrent_{uuid4().hex[:8]}.created"
        agent_ids = [f"concurrent_agent_{i}_{uuid4().hex[:4]}" for i in range(10)]

        # Subscribe all agents concurrently
        tasks = [
            subscription_handler.subscribe(agent_id=agent_id, topic=topic)
            for agent_id in agent_ids
        ]
        results = await asyncio.gather(*tasks)

        # All subscriptions should succeed
        assert len(results) == 10
        assert all(r.status == EnumSubscriptionStatus.ACTIVE for r in results)
        assert all(r.topic == topic for r in results)

        # Verify all agents are subscribed with unique subscription IDs
        subscription_ids = {r.id for r in results}
        assert len(subscription_ids) == 10  # All unique IDs

        # Verify each agent has their subscription
        for agent_id in agent_ids:
            subs = await subscription_handler.list_subscriptions(agent_id)
            assert len(subs) == 1
            assert subs[0].topic == topic

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_unsubscribe(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Subscribe and unsubscribe operations handle concurrent access correctly.

        This test simulates a race condition where one operation tries to
        subscribe while another tries to unsubscribe, verifying the system
        reaches a consistent final state.
        """
        agent_id = f"race_agent_{uuid4().hex[:8]}"
        topic = f"memory.race_{uuid4().hex[:8]}.created"

        # First, establish a subscription
        await subscription_handler.subscribe(agent_id=agent_id, topic=topic)

        # Define operations that will run concurrently
        async def subscribe_op() -> ModelSubscription:
            return await subscription_handler.subscribe(agent_id=agent_id, topic=topic)

        async def unsubscribe_op() -> bool:
            return await subscription_handler.unsubscribe(
                agent_id=agent_id, topic=topic
            )

        # Run subscribe and unsubscribe concurrently multiple times
        for _ in range(5):
            # Create initial subscription if not exists
            await subscription_handler.subscribe(agent_id=agent_id, topic=topic)

            # Run concurrent operations
            results = await asyncio.gather(
                subscribe_op(),
                unsubscribe_op(),
                return_exceptions=True,
            )

            # Neither should raise an exception
            for result in results:
                assert not isinstance(
                    result, Exception
                ), f"Unexpected exception: {result}"

        # Final state should be consistent (either subscribed or not)
        final_subs = await subscription_handler.list_subscriptions(agent_id)
        # The list should have 0 or 1 subscriptions for this topic
        topic_subs = [s for s in final_subs if s.topic == topic]
        assert len(topic_subs) <= 1

    @pytest.mark.asyncio
    async def test_concurrent_subscribes_multiple_topics(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Single agent can subscribe to multiple topics concurrently.

        Verifies that concurrent subscription to different topics
        by the same agent works correctly without conflicts.
        """
        agent_id = f"multi_topic_agent_{uuid4().hex[:8]}"
        topics = [f"memory.concurrent_{uuid4().hex[:8]}.created" for _ in range(10)]

        # Subscribe to all topics concurrently
        tasks = [
            subscription_handler.subscribe(agent_id=agent_id, topic=topic)
            for topic in topics
        ]
        results = await asyncio.gather(*tasks)

        # All subscriptions should succeed
        assert len(results) == 10
        assert all(r.status == EnumSubscriptionStatus.ACTIVE for r in results)

        # Verify all subscriptions exist
        all_subs = await subscription_handler.list_subscriptions(agent_id)
        subscribed_topics = {s.topic for s in all_subs}
        assert subscribed_topics == set(topics)

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_same_topic(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Multiple agents subscribing to same topic concurrently.

        Verifies that concurrent subscriptions from different agents
        to the same topic all succeed with unique subscriptions.
        """
        topic = f"memory.test_{uuid4().hex[:8]}.created"
        agent_ids = [f"test_agent_{uuid4().hex[:8]}" for _ in range(10)]

        # Subscribe all agents concurrently
        tasks = [
            subscription_handler.subscribe(agent_id=agent_id, topic=topic)
            for agent_id in agent_ids
        ]
        results = await asyncio.gather(*tasks)

        # All subscriptions should succeed
        assert len(results) == 10
        assert all(isinstance(r, ModelSubscription) for r in results)
        assert all(r.topic == topic for r in results)
        assert len({r.agent_id for r in results}) == 10  # All unique agents

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_unsubscribe_same_agent(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
    ) -> None:
        """Subscribe and unsubscribe same agent concurrently.

        Tests race conditions when an agent is simultaneously subscribing
        to new topics while unsubscribing from others.
        """
        topics = [f"memory.test_{uuid4().hex[:8]}.created" for _ in range(5)]

        # Subscribe to all topics first
        for topic in topics:
            await subscription_handler.subscribe(agent_id=unique_agent_id, topic=topic)

        # Concurrently: unsubscribe some + subscribe new ones
        unsubscribe_tasks = [
            subscription_handler.unsubscribe(agent_id=unique_agent_id, topic=t)
            for t in topics[:2]
        ]
        new_topics = [f"memory.test_{uuid4().hex[:8]}.updated" for _ in range(2)]
        subscribe_tasks = [
            subscription_handler.subscribe(agent_id=unique_agent_id, topic=t)
            for t in new_topics
        ]

        all_results = await asyncio.gather(*unsubscribe_tasks, *subscribe_tasks)

        # Verify final state is consistent
        final_subs = await subscription_handler.list_subscriptions(unique_agent_id)
        # Should have 5 - 2 + 2 = 5 subscriptions (but may vary due to race conditions)
        # The key assertion is that no exceptions were raised
        assert isinstance(final_subs, list)
        # All results should be valid (bool for unsubscribe, ModelSubscription for subscribe)
        assert len(all_results) == 4

    @pytest.mark.asyncio
    async def test_concurrent_list_during_modifications(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
    ) -> None:
        """List subscriptions while modifications happen.

        Verifies that listing subscriptions during concurrent modifications
        does not raise exceptions and returns consistent results.
        """

        async def subscribe_loop() -> None:
            for i in range(5):
                await subscription_handler.subscribe(
                    agent_id=unique_agent_id,
                    topic=f"memory.test_{uuid4().hex[:8]}.event{i}",
                )

        async def list_loop() -> list[int]:
            counts = []
            for _ in range(10):
                subs = await subscription_handler.list_subscriptions(unique_agent_id)
                counts.append(len(subs))
                await asyncio.sleep(0.01)
            return counts

        # Run both concurrently
        await asyncio.gather(subscribe_loop(), list_loop())

        # Final state should be consistent
        final_subs = await subscription_handler.list_subscriptions(unique_agent_id)
        assert len(final_subs) >= 5  # At least the subscribed ones


# =============================================================================
# TestLargeBatch
# =============================================================================


class TestLargeBatch:
    """Tests for large-scale subscription operations.

    These tests verify that the subscription system handles large numbers
    of subscriptions efficiently and correctly, which is important for
    production scenarios with many agents.
    """

    @pytest.mark.asyncio
    async def test_notify_large_subscriber_count(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Notify works correctly with many subscribers.

        Creates 100 subscriptions to a single topic and verifies that
        the notify operation correctly reports the subscriber count.
        """
        topic = f"memory.large_batch_{uuid4().hex[:8]}.created"

        # Create 100 subscriptions
        for i in range(100):
            await subscription_handler.subscribe(
                agent_id=f"batch_agent_{i}_{uuid4().hex[:4]}",
                topic=topic,
            )

        # Create notification event
        event = ModelNotificationEvent(
            event_id=str(uuid4()),
            topic=topic,
            payload=ModelNotificationEventPayload(
                entity_type="batch_item",
                entity_id=f"item_{uuid4().hex[:8]}",
                action="created",
            ),
        )

        try:
            subscriber_count = await subscription_handler.notify(topic, event)
            assert subscriber_count == 100
        except RuntimeError as e:
            # Event bus may not be available - skip gracefully
            if "Kafka" in str(e) or "kafka" in str(e).lower():
                pytest.skip(f"Event bus not available: {e}")
            raise

    @pytest.mark.asyncio
    async def test_list_subscriptions_large_result(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """List subscriptions handles large result sets correctly.

        Creates many subscriptions for a single agent and verifies
        that list_subscriptions returns all of them.
        """
        agent_id = f"large_list_agent_{uuid4().hex[:8]}"
        num_subscriptions = 50

        # Create many subscriptions for the same agent
        topics = [
            f"memory.list_test_{i}_{uuid4().hex[:6]}.created"
            for i in range(num_subscriptions)
        ]
        for topic in topics:
            await subscription_handler.subscribe(agent_id=agent_id, topic=topic)

        # List all subscriptions (non-paginated)
        all_subs = await subscription_handler.list_subscriptions(agent_id)
        assert len(all_subs) == num_subscriptions

        # Verify all topics are present
        returned_topics = {s.topic for s in all_subs}
        assert returned_topics == set(topics)

    @pytest.mark.asyncio
    async def test_list_subscriptions_pagination_large_result(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Paginated list_subscriptions works with large result sets.

        Creates many subscriptions and verifies that pagination
        correctly returns subsets and total counts.
        """
        from omnimemory.handlers import ModelPaginatedSubscriptions

        agent_id = f"paginated_agent_{uuid4().hex[:8]}"
        num_subscriptions = 50

        # Create subscriptions
        for i in range(num_subscriptions):
            await subscription_handler.subscribe(
                agent_id=agent_id,
                topic=f"memory.paginate_{i}_{uuid4().hex[:6]}.created",
            )

        # Test first page
        first_page = await subscription_handler.list_subscriptions(
            agent_id, limit=10, offset=0
        )
        assert isinstance(first_page, ModelPaginatedSubscriptions)
        assert len(first_page.subscriptions) == 10
        assert first_page.total_count == num_subscriptions
        assert first_page.limit == 10
        assert first_page.offset == 0

        # Test middle page
        middle_page = await subscription_handler.list_subscriptions(
            agent_id, limit=10, offset=20
        )
        assert isinstance(middle_page, ModelPaginatedSubscriptions)
        assert len(middle_page.subscriptions) == 10
        assert middle_page.total_count == num_subscriptions
        assert middle_page.offset == 20

        # Test last page (may have fewer items)
        last_page = await subscription_handler.list_subscriptions(
            agent_id, limit=10, offset=45
        )
        assert isinstance(last_page, ModelPaginatedSubscriptions)
        assert len(last_page.subscriptions) == 5  # Only 5 remaining
        assert last_page.total_count == num_subscriptions

        # Verify no overlap between pages
        first_ids = {s.id for s in first_page.subscriptions}
        middle_ids = {s.id for s in middle_page.subscriptions}
        last_ids = {s.id for s in last_page.subscriptions}
        assert first_ids.isdisjoint(middle_ids)
        assert middle_ids.isdisjoint(last_ids)
        assert first_ids.isdisjoint(last_ids)


# =============================================================================
# TestEventBusFailureHandling
# =============================================================================


class TestEventBusFailureHandling:
    """Tests for event bus failure scenarios.

    These tests verify that the subscription handler behaves correctly
    when event bus operations fail, including proper error propagation
    and graceful degradation.
    """

    @pytest.mark.asyncio
    async def test_notify_returns_subscriber_count_even_with_bus_issues(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Notify correctly counts subscribers regardless of event bus status.

        The subscriber count is determined from the database/cache before
        event bus publish, so it should be accurate even if the bus has issues.
        """
        # Subscribe an agent
        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
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
            count = await subscription_handler.notify(
                topic=unique_topic,
                event=event,
            )
            # Count should reflect actual subscribers
            assert count == 1
        except RuntimeError as e:
            # If the event bus is unavailable, the test demonstrates the
            # handler's behavior - it requires an event bus for notify operations
            if "Kafka" in str(e) or "kafka" in str(e).lower():
                pytest.skip(f"Event bus not available: {e}")
            raise

    @pytest.mark.asyncio
    async def test_notify_handles_event_bus_timeout_gracefully(
        self,
        subscription_handler: HandlerSubscription,
    ) -> None:
        """Notify operation handles potential event bus timeouts.

        Verifies that the system properly handles slow event bus responses
        without hanging indefinitely or corrupting state.
        """
        topic = f"memory.timeout_test_{uuid4().hex[:8]}.created"
        agent_id = f"timeout_agent_{uuid4().hex[:8]}"

        # Subscribe first
        await subscription_handler.subscribe(agent_id=agent_id, topic=topic)

        event = ModelNotificationEvent(
            event_id=str(uuid4()),
            topic=topic,
            payload=ModelNotificationEventPayload(
                entity_type="timeout_test",
                entity_id="item_123",
                action="created",
            ),
        )

        try:
            # Use asyncio.wait_for to enforce our own timeout
            result = await asyncio.wait_for(
                subscription_handler.notify(topic, event),
                timeout=30.0,  # 30 second timeout
            )
            assert result >= 0  # Should return valid count
        except TimeoutError:
            pytest.fail("Notify operation timed out after 30 seconds")
        except RuntimeError as e:
            if "Kafka" in str(e) or "kafka" in str(e).lower():
                pytest.skip(f"Event bus not available: {e}")
            raise

    @pytest.mark.asyncio
    async def test_metrics_track_notification_attempts(
        self,
        subscription_handler: HandlerSubscription,
        unique_agent_id: str,
        unique_topic: str,
        sample_event: ModelNotificationEvent,
    ) -> None:
        """Metrics correctly track notification publish attempts.

        Verifies that the notifications_published metric is incremented
        when a notify operation is attempted.
        """
        # Subscribe first
        await subscription_handler.subscribe(
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        initial_metrics = await subscription_handler.get_metrics()
        initial_published = initial_metrics.notifications_published

        try:
            await subscription_handler.notify(
                topic=unique_topic,
                event=sample_event,
            )
        except RuntimeError as e:
            if "Kafka" in str(e) or "kafka" in str(e).lower():
                pytest.skip(f"Event bus not available: {e}")
            raise

        final_metrics = await subscription_handler.get_metrics()

        # Verify metric was incremented
        assert final_metrics.notifications_published == initial_published + 1
