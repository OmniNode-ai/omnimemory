# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory domain message type registration.

Registers all memory wire models (Kafka event payloads, command inputs, and
response envelopes) with ``RegistryMessageType``.  This enables type-based
envelope routing and startup validation for the memory domain.

The registration list is intentionally explicit rather than derived from
contract YAML files.  Contract-driven discovery is acceptable as a future
enhancement, but an explicit list keeps the registration deterministic and
auditable.

Design:
    - All registrations use ``domain="memory"``
    - ``handler_id`` matches the node directory name
    - ``category`` follows topic naming: ``.cmd.`` -> COMMAND, ``.evt.`` -> EVENT
    - Consumed events from external domains use EVENT category

Related:
    - OMN-2217: Phase 6 -- Wire model registration & entry point declaration
    - OMN-937: Central Message Type Registry implementation
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumMessageCategory

if TYPE_CHECKING:
    from omnibase_infra.runtime.registry import RegistryMessageType

logger = logging.getLogger(__name__)

MEMORY_DOMAIN = "memory"
"""Owning domain for all memory message types."""

# Total number of unique message types registered.
# Used in tests and validate_startup logging.
EXPECTED_MESSAGE_TYPE_COUNT = 10


def register_memory_message_types(
    registry: RegistryMessageType,
) -> list[str]:
    """Register all memory wire models with the message type registry.

    This function registers 10 message types spanning:
    - 1 consumed Kafka event (intent classification from omniintelligence)
    - 3 Kafka command models (effect node inputs)
    - 3 Kafka event/response models (effect node outputs)
    - 1 notification event model (published by orchestrator)
    - 1 orchestrator command model (coordinator input)
    - 1 orchestrator response model (coordinator output)

    The registry is NOT frozen by this function.  The caller is responsible
    for calling ``registry.freeze()`` after all domains have registered.

    Args:
        registry: An unfrozen RegistryMessageType instance.

    Returns:
        List of registered message type names (for logging).

    Raises:
        ModelOnexError: If registry is already frozen.
        MessageTypeRegistryError: If any registration fails validation.
    """
    registered: list[str] = []

    # =========================================================================
    # Consumed Kafka Event (from omniintelligence) -- EVENT category
    # =========================================================================

    # 1. Intent classification event consumed from omniintelligence
    registry.register_simple(
        message_type="ModelIntentClassifiedEvent",
        handler_id="intent_event_consumer_effect",
        category=EnumMessageCategory.EVENT,
        domain=MEMORY_DOMAIN,
        description=(
            "Intent classification event consumed from omniintelligence "
            "for memory storage"
        ),
    )
    registered.append("ModelIntentClassifiedEvent")

    # =========================================================================
    # Intent Storage Effect (orchestrator-invoked) -- COMMAND/EVENT
    # =========================================================================

    # 2. Intent storage request (command input)
    registry.register_simple(
        message_type="ModelIntentStorageRequest",
        handler_id="intent_storage_effect",
        category=EnumMessageCategory.COMMAND,
        domain=MEMORY_DOMAIN,
        description="Intent storage request command input",
    )
    registered.append("ModelIntentStorageRequest")

    # 3. Intent storage response (event output)
    registry.register_simple(
        message_type="ModelIntentStorageResponse",
        handler_id="intent_storage_effect",
        category=EnumMessageCategory.EVENT,
        domain=MEMORY_DOMAIN,
        description="Intent storage response event output",
    )
    registered.append("ModelIntentStorageResponse")

    # =========================================================================
    # Memory Retrieval Effect -- COMMAND/EVENT
    # =========================================================================

    # 4. Memory retrieval request (command input)
    registry.register_simple(
        message_type="ModelMemoryRetrievalRequest",
        handler_id="memory_retrieval_effect",
        category=EnumMessageCategory.COMMAND,
        domain=MEMORY_DOMAIN,
        description="Memory retrieval request command consumed from Kafka",
    )
    registered.append("ModelMemoryRetrievalRequest")

    # 5. Memory retrieval response (event output)
    registry.register_simple(
        message_type="ModelMemoryRetrievalResponse",
        handler_id="memory_retrieval_effect",
        category=EnumMessageCategory.EVENT,
        domain=MEMORY_DOMAIN,
        description="Memory retrieval response event output",
    )
    registered.append("ModelMemoryRetrievalResponse")

    # =========================================================================
    # Memory Storage Effect (orchestrator-invoked) -- COMMAND/EVENT
    # =========================================================================

    # 6. Memory storage request (command input)
    registry.register_simple(
        message_type="ModelMemoryStorageRequest",
        handler_id="memory_storage_effect",
        category=EnumMessageCategory.COMMAND,
        domain=MEMORY_DOMAIN,
        description="Memory storage CRUD request command input",
    )
    registered.append("ModelMemoryStorageRequest")

    # 7. Memory storage response (event output)
    registry.register_simple(
        message_type="ModelMemoryStorageResponse",
        handler_id="memory_storage_effect",
        category=EnumMessageCategory.EVENT,
        domain=MEMORY_DOMAIN,
        description="Memory storage CRUD response event output",
    )
    registered.append("ModelMemoryStorageResponse")

    # =========================================================================
    # Agent Coordinator Orchestrator -- COMMAND/EVENT
    # =========================================================================

    # 8. Agent coordinator request (command input)
    registry.register_simple(
        message_type="ModelAgentCoordinatorRequest",
        handler_id="agent_coordinator_orchestrator",
        category=EnumMessageCategory.COMMAND,
        domain=MEMORY_DOMAIN,
        description=(
            "Agent coordinator request for subscription management "
            "and notification dispatch"
        ),
    )
    registered.append("ModelAgentCoordinatorRequest")

    # 9. Agent coordinator response (event output)
    registry.register_simple(
        message_type="ModelAgentCoordinatorResponse",
        handler_id="agent_coordinator_orchestrator",
        category=EnumMessageCategory.EVENT,
        domain=MEMORY_DOMAIN,
        description="Agent coordinator response with operation result",
    )
    registered.append("ModelAgentCoordinatorResponse")

    # 10. Notification event (published by coordinator to Kafka)
    registry.register_simple(
        message_type="ModelNotificationEvent",
        handler_id="agent_coordinator_orchestrator",
        category=EnumMessageCategory.EVENT,
        domain=MEMORY_DOMAIN,
        description=(
            "Notification event published to Kafka for cross-agent "
            "memory change notifications"
        ),
    )
    registered.append("ModelNotificationEvent")

    logger.info(
        "Registered %d memory message types with RegistryMessageType",
        len(registered),
    )

    return registered


__all__ = [
    "EXPECTED_MESSAGE_TYPE_COUNT",
    "MEMORY_DOMAIN",
    "register_memory_message_types",
]
