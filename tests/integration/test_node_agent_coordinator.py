# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Integration tests for NodeAgentCoordinatorOrchestrator.

This module tests the agent coordinator orchestrator node which wraps
HandlerSubscription to provide subscription management operations.

Test Categories:
    - Subscribe: Orchestrator subscribe action
    - Unsubscribe: Orchestrator unsubscribe action
    - ListSubscriptions: Orchestrator list subscriptions action
    - Notify: Orchestrator notify action

Prerequisites:
    - PostgreSQL running at TEST_DB_DSN
    - Valkey running at TEST_VALKEY_HOST:TEST_VALKEY_PORT
    - omnibase_infra installed (dev dependency)

Note:
    The NodeAgentCoordinatorOrchestrator node implementation is pending.
    These tests define the expected behavior for when it is implemented.
    Currently, tests use HandlerSubscription directly as a reference.

Usage:
    # Run orchestrator tests
    pytest tests/integration/test_node_agent_coordinator.py -v

    # Run with markers
    pytest -m "integration and orchestrator" -v

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

# Check if dependencies are available
_DEPENDENCIES_AVAILABLE = False
_SKIP_REASON = "Required dependencies not installed"

try:
    from omnibase_core.container import ModelONEXContainer

    from omnimemory.enums.enum_subscription_status import EnumSubscriptionStatus
    from omnimemory.handlers import (
        HandlerSubscription,
        ModelHandlerSubscriptionConfig,
    )
    from omnimemory.models.subscription import (
        ModelNotificationEvent,
        ModelNotificationEventPayload,
    )
    from omnimemory.nodes.agent_coordinator_orchestrator import (
        EnumAgentCoordinatorAction,
        ModelAgentCoordinatorRequest,
        ModelAgentCoordinatorResponse,
    )

    _DEPENDENCIES_AVAILABLE = True
    _SKIP_REASON = ""
except ImportError as e:
    ModelONEXContainer = None  # type: ignore[assignment, misc]
    _SKIP_REASON = f"Required dependencies not available: {e}"


# =============================================================================
# Skip Conditions
# =============================================================================

# Skip all tests if dependencies are not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.orchestrator,
    pytest.mark.skipif(
        not _DEPENDENCIES_AVAILABLE,
        reason=_SKIP_REASON,
    ),
]


# =============================================================================
# Node Implementation
# =============================================================================

# Note: The actual NodeAgentCoordinatorOrchestrator is pending implementation.
# For now, we provide a reference implementation for testing purposes that
# wraps HandlerSubscription. This will be replaced when the node is implemented.


class NodeAgentCoordinatorOrchestrator:
    """Reference implementation for testing agent coordinator orchestrator.

    This class wraps HandlerSubscription to provide the orchestrator API
    as defined by the request/response models. It will be replaced by the
    actual node implementation once available.

    Note:
        This is a test-only implementation. The production node should be
        located at omnimemory/nodes/agent_coordinator_orchestrator/node.py.
    """

    def __init__(self, config: ModelHandlerSubscriptionConfig) -> None:
        """Initialize the orchestrator with configuration.

        Args:
            config: Handler subscription configuration.
        """
        self._config = config
        self._handler: HandlerSubscription | None = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the orchestrator has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the underlying handler."""
        if self._initialized:
            return

        container = ModelONEXContainer()
        self._handler = HandlerSubscription(container)
        await self._handler.initialize(self._config)
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the underlying handler."""
        if self._handler:
            await self._handler.shutdown()
            self._handler = None
        self._initialized = False

    async def execute(
        self,
        request: ModelAgentCoordinatorRequest,
    ) -> ModelAgentCoordinatorResponse:
        """Execute an agent coordinator action.

        Args:
            request: The coordinator request.

        Returns:
            The coordinator response. Returns error response if not initialized.
        """
        if not self._initialized or self._handler is None:
            return ModelAgentCoordinatorResponse(
                success=False,
                action=request.action,
                correlation_id=request.correlation_id,
                error_message=(
                    "NodeAgentCoordinatorOrchestrator not initialized. "
                    "Call initialize() first."
                ),
                error_code="NOT_INITIALIZED",
            )

        try:
            if request.action == EnumAgentCoordinatorAction.SUBSCRIBE:
                return await self._handle_subscribe(request)
            elif request.action == EnumAgentCoordinatorAction.UNSUBSCRIBE:
                return await self._handle_unsubscribe(request)
            elif request.action == EnumAgentCoordinatorAction.LIST_SUBSCRIPTIONS:
                return await self._handle_list_subscriptions(request)
            elif request.action == EnumAgentCoordinatorAction.NOTIFY:
                return await self._handle_notify(request)
            else:
                return ModelAgentCoordinatorResponse(
                    success=False,
                    action=request.action,
                    correlation_id=request.correlation_id,
                    error_message=f"Unknown action: {request.action}",
                    error_code="UNKNOWN_ACTION",
                )
        except Exception as e:
            return ModelAgentCoordinatorResponse(
                success=False,
                action=request.action,
                correlation_id=request.correlation_id,
                error_message=str(e),
                error_code="EXECUTION_ERROR",
            )

    async def _handle_subscribe(
        self,
        request: ModelAgentCoordinatorRequest,
    ) -> ModelAgentCoordinatorResponse:
        """Handle subscribe action."""
        assert self._handler is not None
        assert request.agent_id is not None
        assert request.topic is not None

        subscription = await self._handler.subscribe(
            agent_id=request.agent_id,
            topic=request.topic,
            metadata=request.metadata,
        )

        return ModelAgentCoordinatorResponse(
            success=True,
            action=request.action,
            correlation_id=request.correlation_id,
            subscription=subscription,
        )

    async def _handle_unsubscribe(
        self,
        request: ModelAgentCoordinatorRequest,
    ) -> ModelAgentCoordinatorResponse:
        """Handle unsubscribe action."""
        assert self._handler is not None
        assert request.agent_id is not None
        assert request.topic is not None

        result = await self._handler.unsubscribe(
            agent_id=request.agent_id,
            topic=request.topic,
        )

        if result:
            return ModelAgentCoordinatorResponse(
                success=True,
                action=request.action,
                correlation_id=request.correlation_id,
            )
        else:
            return ModelAgentCoordinatorResponse(
                success=False,
                action=request.action,
                correlation_id=request.correlation_id,
                error_message="Subscription not found",
                error_code="NOT_FOUND",
            )

    async def _handle_list_subscriptions(
        self,
        request: ModelAgentCoordinatorRequest,
    ) -> ModelAgentCoordinatorResponse:
        """Handle list_subscriptions action."""
        assert self._handler is not None
        assert request.agent_id is not None

        subscriptions = await self._handler.list_subscriptions(request.agent_id)

        return ModelAgentCoordinatorResponse(
            success=True,
            action=request.action,
            correlation_id=request.correlation_id,
            subscriptions=subscriptions,
        )

    async def _handle_notify(
        self,
        request: ModelAgentCoordinatorRequest,
    ) -> ModelAgentCoordinatorResponse:
        """Handle notify action."""
        assert self._handler is not None
        assert request.topic is not None
        assert request.event is not None

        subscriber_count = await self._handler.notify(
            topic=request.topic,
            event=request.event,
        )

        return ModelAgentCoordinatorResponse(
            success=True,
            action=request.action,
            correlation_id=request.correlation_id,
            subscriber_count=subscriber_count,
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def orchestrator_node(
    test_db_dsn: str,
    test_valkey_host: str,
    test_valkey_port: int,
    services_available: bool,
) -> AsyncGenerator[NodeAgentCoordinatorOrchestrator, None]:
    """Create and initialize orchestrator node for tests.

    Yields:
        Initialized NodeAgentCoordinatorOrchestrator instance.
    """
    if not services_available:
        pytest.skip("Required services (PostgreSQL, Valkey) not available")

    config = ModelHandlerSubscriptionConfig(
        db_dsn=test_db_dsn,
        valkey_host=test_valkey_host,
        valkey_port=test_valkey_port,
    )
    node = NodeAgentCoordinatorOrchestrator(config)
    await node.initialize()

    yield node

    await node.shutdown()


# =============================================================================
# Subscribe Action Tests
# =============================================================================


class TestOrchestratorSubscribe:
    """Tests for orchestrator subscribe action."""

    @pytest.mark.asyncio
    async def test_subscribe_action(
        self,
        orchestrator_node: NodeAgentCoordinatorOrchestrator,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Subscribe action creates subscription."""
        request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.SUBSCRIBE,
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        response = await orchestrator_node.execute(request)

        assert response.success is True
        assert response.action == EnumAgentCoordinatorAction.SUBSCRIBE
        assert response.subscription is not None
        assert response.subscription.agent_id == unique_agent_id
        assert response.subscription.topic == unique_topic
        assert response.subscription.status == EnumSubscriptionStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_subscribe_action_with_metadata(
        self,
        orchestrator_node: NodeAgentCoordinatorOrchestrator,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Subscribe action with metadata."""
        request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.SUBSCRIBE,
            agent_id=unique_agent_id,
            topic=unique_topic,
            metadata={"source": "test", "version": "1.0"},
        )

        response = await orchestrator_node.execute(request)

        assert response.success is True
        assert response.subscription is not None
        assert response.subscription.metadata == {"source": "test", "version": "1.0"}

    @pytest.mark.asyncio
    async def test_subscribe_action_invalid_topic_fails(
        self,
        orchestrator_node: NodeAgentCoordinatorOrchestrator,
        unique_agent_id: str,
    ) -> None:
        """Subscribe action with invalid topic fails."""
        request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.SUBSCRIBE,
            agent_id=unique_agent_id,
            topic="invalid-topic-format",
        )

        response = await orchestrator_node.execute(request)

        assert response.success is False
        assert response.error_message is not None
        assert "subscribe() received invalid topic format" in response.error_message


# =============================================================================
# Unsubscribe Action Tests
# =============================================================================


class TestOrchestratorUnsubscribe:
    """Tests for orchestrator unsubscribe action."""

    @pytest.mark.asyncio
    async def test_unsubscribe_action(
        self,
        orchestrator_node: NodeAgentCoordinatorOrchestrator,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Unsubscribe action removes subscription."""
        # First subscribe
        subscribe_request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.SUBSCRIBE,
            agent_id=unique_agent_id,
            topic=unique_topic,
        )
        await orchestrator_node.execute(subscribe_request)

        # Then unsubscribe
        unsubscribe_request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.UNSUBSCRIBE,
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        response = await orchestrator_node.execute(unsubscribe_request)

        assert response.success is True
        assert response.action == EnumAgentCoordinatorAction.UNSUBSCRIBE

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_fails(
        self,
        orchestrator_node: NodeAgentCoordinatorOrchestrator,
        unique_agent_id: str,
    ) -> None:
        """Unsubscribe action for non-existent subscription fails."""
        request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.UNSUBSCRIBE,
            agent_id=unique_agent_id,
            topic="memory.nonexistent.topic",
        )

        response = await orchestrator_node.execute(request)

        assert response.success is False
        assert response.error_code == "NOT_FOUND"


# =============================================================================
# List Subscriptions Action Tests
# =============================================================================


class TestOrchestratorListSubscriptions:
    """Tests for orchestrator list_subscriptions action."""

    @pytest.mark.asyncio
    async def test_list_subscriptions_action(
        self,
        orchestrator_node: NodeAgentCoordinatorOrchestrator,
        unique_agent_id: str,
    ) -> None:
        """List subscriptions returns all agent subscriptions."""
        # Create subscriptions
        topics = [
            f"memory.test_{uuid4().hex[:8]}.created",
            f"memory.test_{uuid4().hex[:8]}.updated",
        ]

        for topic in topics:
            request = ModelAgentCoordinatorRequest(
                action=EnumAgentCoordinatorAction.SUBSCRIBE,
                agent_id=unique_agent_id,
                topic=topic,
            )
            await orchestrator_node.execute(request)

        # List subscriptions
        list_request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.LIST_SUBSCRIPTIONS,
            agent_id=unique_agent_id,
        )

        response = await orchestrator_node.execute(list_request)

        assert response.success is True
        assert response.subscriptions is not None
        assert len(response.subscriptions) == 2

        returned_topics = {s.topic for s in response.subscriptions}
        assert returned_topics == set(topics)

    @pytest.mark.asyncio
    async def test_list_subscriptions_empty(
        self,
        orchestrator_node: NodeAgentCoordinatorOrchestrator,
        unique_agent_id: str,
    ) -> None:
        """List subscriptions returns empty for new agent."""
        request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.LIST_SUBSCRIPTIONS,
            agent_id=unique_agent_id,
        )

        response = await orchestrator_node.execute(request)

        assert response.success is True
        assert response.subscriptions is not None
        assert len(response.subscriptions) == 0


# =============================================================================
# Notify Action Tests
# =============================================================================


class TestOrchestratorNotify:
    """Tests for orchestrator notify action.

    Note: Notifications are published to Kafka. These tests verify
    the subscriber count is returned correctly. Actual event consumption
    by agents happens via Kafka consumer groups.
    """

    @pytest.mark.asyncio
    async def test_notify_action(
        self,
        orchestrator_node: NodeAgentCoordinatorOrchestrator,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Notify action publishes to Kafka and returns subscriber count."""
        # Subscribe
        subscribe_request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.SUBSCRIBE,
            agent_id=unique_agent_id,
            topic=unique_topic,
        )
        await orchestrator_node.execute(subscribe_request)

        # Notify
        event_id = str(uuid4())
        notify_request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.NOTIFY,
            topic=unique_topic,
            event=ModelNotificationEvent(
                event_id=event_id,
                topic=unique_topic,
                payload=ModelNotificationEventPayload(
                    entity_type="test_item",
                    entity_id="item_123",
                    action="created",
                ),
            ),
        )

        response = await orchestrator_node.execute(notify_request)

        assert response.success is True
        assert response.subscriber_count == 1

    @pytest.mark.asyncio
    async def test_notify_action_no_subscribers(
        self,
        orchestrator_node: NodeAgentCoordinatorOrchestrator,
    ) -> None:
        """Notify action with no subscribers succeeds with zero count."""
        request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.NOTIFY,
            topic="memory.orphan.topic",
            event=ModelNotificationEvent(
                event_id=str(uuid4()),
                topic="memory.orphan.topic",
                payload=ModelNotificationEventPayload(
                    entity_type="orphan",
                    entity_id="orphan_123",
                    action="created",
                ),
            ),
        )

        response = await orchestrator_node.execute(request)

        assert response.success is True
        assert response.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_notify_action_multiple_subscribers(
        self,
        orchestrator_node: NodeAgentCoordinatorOrchestrator,
        unique_topic: str,
    ) -> None:
        """Notify action reports correct subscriber count for multiple subscribers."""
        # Subscribe multiple agents
        agents = [f"agent_{uuid4().hex[:8]}" for _ in range(3)]
        for agent_id in agents:
            subscribe_request = ModelAgentCoordinatorRequest(
                action=EnumAgentCoordinatorAction.SUBSCRIBE,
                agent_id=agent_id,
                topic=unique_topic,
            )
            await orchestrator_node.execute(subscribe_request)

        # Notify
        notify_request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.NOTIFY,
            topic=unique_topic,
            event=ModelNotificationEvent(
                event_id=str(uuid4()),
                topic=unique_topic,
                payload=ModelNotificationEventPayload(
                    entity_type="test_item",
                    entity_id="item_123",
                    action="created",
                ),
            ),
        )

        response = await orchestrator_node.execute(notify_request)

        assert response.success is True
        assert response.subscriber_count == 3


# =============================================================================
# Request Validation Tests
# =============================================================================


class TestRequestValidation:
    """Tests for request validation via Pydantic."""

    def test_subscribe_requires_agent_id(self) -> None:
        """Subscribe action requires agent_id."""
        with pytest.raises(ValueError, match="agent_id"):
            ModelAgentCoordinatorRequest(
                action=EnumAgentCoordinatorAction.SUBSCRIBE,
                topic="memory.item.created",
            )

    def test_subscribe_requires_topic(
        self,
        unique_agent_id: str,
    ) -> None:
        """Subscribe action requires topic."""
        with pytest.raises(ValueError, match="topic"):
            ModelAgentCoordinatorRequest(
                action=EnumAgentCoordinatorAction.SUBSCRIBE,
                agent_id=unique_agent_id,
            )

    def test_notify_requires_topic(self) -> None:
        """Notify action requires topic."""
        with pytest.raises(ValueError, match="topic"):
            ModelAgentCoordinatorRequest(
                action=EnumAgentCoordinatorAction.NOTIFY,
                event=ModelNotificationEvent(
                    event_id=str(uuid4()),
                    topic="memory.item.created",
                    payload=ModelNotificationEventPayload(
                        entity_type="item",
                        entity_id="item_123",
                        action="created",
                    ),
                ),
            )

    def test_notify_requires_event(self) -> None:
        """Notify action requires event."""
        with pytest.raises(ValueError, match="event"):
            ModelAgentCoordinatorRequest(
                action=EnumAgentCoordinatorAction.NOTIFY,
                topic="memory.item.created",
            )

    def test_notify_topic_consistency_validation(self) -> None:
        """Notify action validates that event.topic matches request topic."""
        with pytest.raises(ValueError, match="Event topic mismatch"):
            ModelAgentCoordinatorRequest(
                action=EnumAgentCoordinatorAction.NOTIFY,
                topic="memory.item.created",
                event=ModelNotificationEvent(
                    event_id=str(uuid4()),
                    topic="memory.item.updated",  # Mismatch with request topic
                    payload=ModelNotificationEventPayload(
                        entity_type="item",
                        entity_id="item_123",
                        action="updated",
                    ),
                ),
            )

    def test_notify_topic_consistency_valid(self) -> None:
        """Notify action accepts matching topic in event and request."""
        # Should not raise - topics match
        request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.NOTIFY,
            topic="memory.item.created",
            event=ModelNotificationEvent(
                event_id=str(uuid4()),
                topic="memory.item.created",  # Matches request topic
                payload=ModelNotificationEventPayload(
                    entity_type="item",
                    entity_id="item_123",
                    action="created",
                ),
            ),
        )
        assert request.topic == request.event.topic

    def test_list_subscriptions_requires_agent_id(self) -> None:
        """List subscriptions action requires agent_id."""
        with pytest.raises(ValueError, match="agent_id"):
            ModelAgentCoordinatorRequest(
                action=EnumAgentCoordinatorAction.LIST_SUBSCRIPTIONS,
            )


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestOrchestratorErrorHandling:
    """Tests for orchestrator error handling."""

    @pytest.mark.asyncio
    async def test_execute_before_initialize_returns_error(
        self,
        test_db_dsn: str,
        test_valkey_host: str,
        test_valkey_port: int,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Execute before initialize returns error response."""
        config = ModelHandlerSubscriptionConfig(
            db_dsn=test_db_dsn,
            valkey_host=test_valkey_host,
            valkey_port=test_valkey_port,
        )
        node = NodeAgentCoordinatorOrchestrator(config)

        request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.SUBSCRIBE,
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        response = await node.execute(request)

        assert response.success is False
        assert response.error_code == "NOT_INITIALIZED"
        assert response.error_message is not None
        assert "not initialized" in response.error_message

    @pytest.mark.asyncio
    async def test_correlation_id_preserved_in_response(
        self,
        orchestrator_node: NodeAgentCoordinatorOrchestrator,
        unique_agent_id: str,
        unique_topic: str,
    ) -> None:
        """Correlation ID from request is preserved in response."""
        from uuid import UUID

        request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.SUBSCRIBE,
            agent_id=unique_agent_id,
            topic=unique_topic,
        )

        response = await orchestrator_node.execute(request)

        assert response.correlation_id == request.correlation_id
        assert isinstance(response.correlation_id, UUID)
