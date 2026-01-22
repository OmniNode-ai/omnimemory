# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Agent Coordinator Orchestrator Node - ONEX 4.0 Compliant.

Orchestrates cross-agent memory coordination through subscription management
and notification delivery.

Node Type: ORCHESTRATOR
- Workflow coordination and routing
- Cross-node coordination via handlers
- Lifecycle management for subscriptions

This node is a thin wrapper around HandlerSubscription - all business logic
lives in the handler. The node provides the ONEX interface and error handling.

Example::

    from omnimemory.nodes.agent_coordinator_orchestrator import (
        NodeAgentCoordinatorOrchestrator,
        ModelAgentCoordinatorRequest,
        EnumAgentCoordinatorAction,
    )
    from omnimemory.handlers import ModelHandlerSubscriptionConfig
    from omnimemory.models.subscription import ModelSubscriptionDeliveryWebhook

    config = ModelHandlerSubscriptionConfig(
        db_dsn="postgresql://user:pass@localhost:5432/omnimemory",
        valkey_host="localhost",
        valkey_port=6379,
    )
    node = NodeAgentCoordinatorOrchestrator(config=config)
    await node.initialize()

    # Subscribe an agent
    request = ModelAgentCoordinatorRequest(
        action=EnumAgentCoordinatorAction.SUBSCRIBE,
        agent_id="agent_123",
        topic="memory.item.created",
        delivery=ModelSubscriptionDeliveryWebhook(
            webhook_url="https://example.com/webhook",
        ),
    )
    response = await node.execute(request)
    print(f"Subscription created: {response.subscription.id}")

    await node.shutdown()

.. versionadded:: 0.1.0
    Initial implementation for OMN-1393.
"""

from __future__ import annotations

import logging

from omnimemory.handlers import (
    HandlerSubscription,
    ModelHandlerSubscriptionConfig,
    ModelSubscriptionHealth,
)

from ..base import BaseOrchestratorNode, ContainerType
from .models import (
    EnumAgentCoordinatorAction,
    ModelAgentCoordinatorRequest,
    ModelAgentCoordinatorResponse,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NodeAgentCoordinatorOrchestrator",
]


class NodeAgentCoordinatorOrchestrator(BaseOrchestratorNode):
    """ONEX Orchestrator node for agent subscription management.

    This node orchestrates cross-agent memory coordination through:
    - Subscription registration and management
    - Notification delivery to subscribers
    - Circuit breaker protection for failing endpoints

    Following ONEX patterns:
        - Node is a thin wrapper (minimal logic)
        - All business logic is in HandlerSubscription
        - Error handling converts exceptions to error responses
        - Async operations for I/O-bound work

    Operations:
        - subscribe: Register an agent's subscription to a memory topic
        - unsubscribe: Remove an agent's subscription
        - list_subscriptions: Get all subscriptions for an agent
        - notify: Send notifications to all topic subscribers

    Attributes:
        _config: Handler configuration for database, cache, and HTTP settings.
        _handler: The HandlerSubscription instance for all operations.
        _initialized: Whether the node has been initialized.

    Example::

        config = ModelHandlerSubscriptionConfig(
            db_dsn="postgresql://...",
            valkey_host="localhost",
        )
        node = NodeAgentCoordinatorOrchestrator(config=config)
        await node.initialize()

        request = ModelAgentCoordinatorRequest(
            action=EnumAgentCoordinatorAction.SUBSCRIBE,
            agent_id="agent_alpha",
            topic="memory.item.created",
            delivery=delivery_config,
        )
        response = await node.execute(request)
    """

    def __init__(
        self,
        config: ModelHandlerSubscriptionConfig,
        container: ContainerType | None = None,
    ) -> None:
        """Initialize the orchestrator node.

        Args:
            config: Configuration for the subscription handler including
                database, cache, and HTTP settings.
            container: Optional ONEX container for dependency injection.
                If provided, config may be resolved from container.

        Raises:
            ValueError: If config is None and container doesn't provide one.
        """
        # Create a minimal container if not provided
        if container is None:
            from omnimemory.compat import ModelOnexContainer

            container = ModelOnexContainer()

        super().__init__(container)
        self._config = config
        self._handler: HandlerSubscription | None = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the node has been initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the orchestrator and its handler.

        Creates and initializes the HandlerSubscription with all required
        connections (database, cache, HTTP).

        Raises:
            RuntimeError: If initialization fails.
        """
        if self._initialized:
            logger.debug("NodeAgentCoordinatorOrchestrator already initialized")
            return

        try:
            self._handler = HandlerSubscription(self._config)
            await self._handler.initialize()
            self._initialized = True
            logger.info("NodeAgentCoordinatorOrchestrator initialized successfully")

        except Exception as e:
            logger.exception("Failed to initialize NodeAgentCoordinatorOrchestrator")
            self._handler = None
            self._initialized = False
            raise RuntimeError(f"Node initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """Shutdown the orchestrator and cleanup resources.

        Gracefully shuts down the handler and releases all connections.
        """
        if self._handler is not None:
            try:
                await self._handler.shutdown()
            except Exception as e:
                logger.warning("Error during handler shutdown: %s", e)
            finally:
                self._handler = None

        self._initialized = False
        logger.info("NodeAgentCoordinatorOrchestrator shutdown complete")

    async def execute(
        self,
        request: ModelAgentCoordinatorRequest,
    ) -> ModelAgentCoordinatorResponse:
        """Execute a coordination action.

        Dispatches to the appropriate handler method based on action type.
        All errors are caught and converted to error responses.

        Args:
            request: The coordination request with action and parameters.

        Returns:
            Response with results or error information.
        """
        if not self._initialized or self._handler is None:
            return ModelAgentCoordinatorResponse(
                success=False,
                action=request.action,
                correlation_id=request.correlation_id,
                error_message="Orchestrator not initialized. Call initialize() first.",
                error_code="NOT_INITIALIZED",
            )

        try:
            match request.action:
                case EnumAgentCoordinatorAction.SUBSCRIBE:
                    return await self._handle_subscribe(request)

                case EnumAgentCoordinatorAction.UNSUBSCRIBE:
                    return await self._handle_unsubscribe(request)

                case EnumAgentCoordinatorAction.LIST_SUBSCRIPTIONS:
                    return await self._handle_list_subscriptions(request)

                case EnumAgentCoordinatorAction.NOTIFY:
                    return await self._handle_notify(request)

        except Exception as e:
            logger.exception(
                "Error executing action %s for correlation_id %s",
                request.action.value,
                request.correlation_id,
            )
            return ModelAgentCoordinatorResponse(
                success=False,
                action=request.action,
                correlation_id=request.correlation_id,
                error_message=str(e)[:2048],
                error_code="EXECUTION_ERROR",
            )

    async def _handle_subscribe(
        self,
        request: ModelAgentCoordinatorRequest,
    ) -> ModelAgentCoordinatorResponse:
        """Handle subscribe action.

        Creates or updates a subscription for the agent on the specified topic.

        Args:
            request: Request with agent_id, topic, and delivery configuration.

        Returns:
            Response with created/updated subscription.
        """
        # Validation already done by request model_validator
        assert self._handler is not None
        assert request.agent_id is not None
        assert request.topic is not None
        assert request.delivery is not None

        subscription = await self._handler.subscribe(
            agent_id=request.agent_id,
            topic=request.topic,
            delivery=request.delivery,
        )

        logger.info(
            "Subscription %s created for agent %s on topic %s",
            subscription.id,
            request.agent_id,
            request.topic,
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
        """Handle unsubscribe action.

        Removes an agent's subscription from the specified topic.

        Args:
            request: Request with agent_id and topic.

        Returns:
            Response indicating success or failure.
        """
        assert self._handler is not None
        assert request.agent_id is not None
        assert request.topic is not None

        success = await self._handler.unsubscribe(
            agent_id=request.agent_id,
            topic=request.topic,
        )

        if success:
            logger.info(
                "Subscription removed for agent %s on topic %s",
                request.agent_id,
                request.topic,
            )
            return ModelAgentCoordinatorResponse(
                success=True,
                action=request.action,
                correlation_id=request.correlation_id,
            )
        else:
            logger.warning(
                "No subscription found for agent %s on topic %s",
                request.agent_id,
                request.topic,
            )
            return ModelAgentCoordinatorResponse(
                success=False,
                action=request.action,
                correlation_id=request.correlation_id,
                error_message=f"No subscription found for agent {request.agent_id} on topic {request.topic}",
                error_code="SUBSCRIPTION_NOT_FOUND",
            )

    async def _handle_list_subscriptions(
        self,
        request: ModelAgentCoordinatorRequest,
    ) -> ModelAgentCoordinatorResponse:
        """Handle list_subscriptions action.

        Retrieves all active subscriptions for an agent.

        Args:
            request: Request with agent_id.

        Returns:
            Response with list of subscriptions.
        """
        assert self._handler is not None
        assert request.agent_id is not None

        subscriptions = await self._handler.list_subscriptions(
            agent_id=request.agent_id,
        )

        logger.debug(
            "Listed %d subscriptions for agent %s",
            len(subscriptions),
            request.agent_id,
        )

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
        """Handle notify action.

        Sends notification event to all subscribers of the topic.

        Args:
            request: Request with topic and event.

        Returns:
            Response with delivery attempt details.
        """
        assert self._handler is not None
        assert request.topic is not None
        assert request.event is not None

        attempts = await self._handler.notify(
            topic=request.topic,
            event=request.event,
        )

        # Calculate success/failure counts
        success_count = sum(1 for a in attempts if a.status.value == "success")
        failure_count = len(attempts) - success_count

        logger.info(
            "Notification for topic %s event %s: %d success, %d failed",
            request.topic,
            request.event.event_id,
            success_count,
            failure_count,
        )

        return ModelAgentCoordinatorResponse(
            success=True,  # Operation succeeded even if some deliveries failed
            action=request.action,
            correlation_id=request.correlation_id,
            delivery_attempts=attempts,
            success_count=success_count,
            failure_count=failure_count,
        )

    def describe(self) -> dict[str, object]:
        """Return node metadata for introspection.

        Returns:
            Dictionary with node metadata including name, version,
            type, status, and supported actions.
        """
        return {
            "name": "agent_coordinator_orchestrator",
            "version": "0.1.0",
            "node_type": "ORCHESTRATOR",
            "initialized": self._initialized,
            "actions": [
                EnumAgentCoordinatorAction.SUBSCRIBE.value,
                EnumAgentCoordinatorAction.UNSUBSCRIBE.value,
                EnumAgentCoordinatorAction.LIST_SUBSCRIPTIONS.value,
                EnumAgentCoordinatorAction.NOTIFY.value,
            ],
            "dependencies": [
                "HandlerSubscription",
                "HandlerDb",
                "HandlerHttpRest",
                "AdapterValkey",
            ],
        }

    async def health_check(self) -> ModelSubscriptionHealth:
        """Check health of node and all dependencies.

        Verifies connectivity to:
        - PostgreSQL database
        - Valkey cache
        - HTTP handler

        Returns:
            Health status with detailed component information.
        """
        if not self._initialized or self._handler is None:
            return ModelSubscriptionHealth(
                is_healthy=False,
                initialized=False,
                error_message="Node not initialized",
            )

        return await self._handler.health_check()
