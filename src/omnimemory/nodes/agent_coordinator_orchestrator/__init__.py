# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Agent Coordinator Orchestrator - ONEX Node (Core 8 Foundation).

Cross-agent memory coordination and sharing through subscription management
and notification delivery.

Node Type: ORCHESTRATOR
- Workflow coordination for agent subscriptions
- Cross-agent notification delivery
- Circuit breaker protection for webhook endpoints

ONEX 4.0 Declarative Pattern:
    This node follows the fully declarative ONEX pattern:
    - contract.yaml defines the node type, inputs, outputs, and dependencies
    - Business logic lives in HandlerSubscription (omnimemory.handlers)
    - No node.py class needed - the contract IS the node definition

Models::

    from omnimemory.nodes.agent_coordinator_orchestrator import (
        EnumAgentCoordinatorAction,
        ModelAgentCoordinatorRequest,
        ModelAgentCoordinatorResponse,
    )

Handler Integration::

    from omnimemory.handlers import (
        HandlerSubscription,
        ModelHandlerSubscriptionConfig,
    )

    config = ModelHandlerSubscriptionConfig(
        db_dsn="postgresql://user:pass@localhost:5432/omnimemory",
        valkey_host="localhost",
    )
    handler = HandlerSubscription(config)
    await handler.initialize()

    # Subscribe an agent
    subscription = await handler.subscribe(
        agent_id="agent_123",
        topic="memory.item.created",
        delivery=delivery_config,
    )

.. versionadded:: 0.1.0
    Initial implementation for OMN-1393.

.. versionchanged:: 0.2.0
    Migrated to ONEX 4.0 fully declarative pattern.
"""

from .models import (
    EnumAgentCoordinatorAction,
    ModelAgentCoordinatorRequest,
    ModelAgentCoordinatorResponse,
)

__all__ = [
    # Models
    "EnumAgentCoordinatorAction",
    "ModelAgentCoordinatorRequest",
    "ModelAgentCoordinatorResponse",
]
