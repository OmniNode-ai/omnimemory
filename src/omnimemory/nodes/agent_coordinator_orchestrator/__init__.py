# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Agent Coordinator Orchestrator - ONEX Node (Core 8 Foundation).

Cross-agent memory coordination and sharing through subscription management
and notification delivery.

Node Type: ORCHESTRATOR
- Workflow coordination for agent subscriptions
- Cross-agent notification delivery
- Circuit breaker protection for webhook endpoints

Example::

    from omnimemory.nodes.agent_coordinator_orchestrator import (
        NodeAgentCoordinatorOrchestrator,
        EnumAgentCoordinatorAction,
        ModelAgentCoordinatorRequest,
        ModelAgentCoordinatorResponse,
    )
    from omnimemory.handlers import ModelHandlerSubscriptionConfig
    from omnimemory.models.subscription import ModelSubscriptionDeliveryWebhook

    config = ModelHandlerSubscriptionConfig(
        db_dsn="postgresql://user:pass@localhost:5432/omnimemory",
        valkey_host="localhost",
    )
    node = NodeAgentCoordinatorOrchestrator(config=config)
    await node.initialize()

    request = ModelAgentCoordinatorRequest(
        action=EnumAgentCoordinatorAction.SUBSCRIBE,
        agent_id="agent_123",
        topic="memory.item.created",
        delivery=ModelSubscriptionDeliveryWebhook(
            webhook_url="https://example.com/webhook",
        ),
    )
    response = await node.execute(request)

.. versionadded:: 0.1.0
    Initial implementation for OMN-1393.
"""

from .models import (
    EnumAgentCoordinatorAction,
    ModelAgentCoordinatorRequest,
    ModelAgentCoordinatorResponse,
)
from .node import NodeAgentCoordinatorOrchestrator

__all__ = [
    # Node
    "NodeAgentCoordinatorOrchestrator",
    # Models
    "EnumAgentCoordinatorAction",
    "ModelAgentCoordinatorRequest",
    "ModelAgentCoordinatorResponse",
]
