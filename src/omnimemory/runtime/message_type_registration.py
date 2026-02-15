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

Observability:
    - ``_registry_ready`` module-level flag tracks registration readiness
    - ``_registered_count`` / ``_registration_failure_count`` track metrics
    - ``is_registry_ready()`` exposes readiness for health checks

Thread-safety:
    This module is designed to be called **once** during single-threaded plugin
    startup.  The ``_registry_lock`` protects individual reads and writes to
    the module-level observability globals, but it does NOT serialize the entire
    ``register_memory_message_types`` operation.  Concurrent calls from
    multiple threads are **not supported** and will produce undefined results.

Related:
    - OMN-2217: Phase 6 -- Wire model registration & entry point declaration
    - OMN-937: Central Message Type Registry implementation
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumMessageCategory

if TYPE_CHECKING:
    # Annotation-only import: the concrete RegistryMessageType instance is
    # passed in at runtime by the caller (plugin.py), so no runtime import
    # is needed here.
    from omnibase_infra.runtime.registry import RegistryMessageType

logger = logging.getLogger(__name__)

MEMORY_DOMAIN = "memory"
"""Owning domain for all memory message types."""

# Total number of unique message types registered.
# Used in tests and validate_startup logging.
EXPECTED_MESSAGE_TYPE_COUNT = 10

# ---------------------------------------------------------------------------
# Observability: readiness flag and metric counters
# ---------------------------------------------------------------------------
_registry_ready: bool = False
"""Module-level readiness flag.  True after successful registration."""

_registered_count: int = 0
"""Number of message types successfully registered in the most recent call.

Reset to ``0`` at the start of each invocation of
``register_memory_message_types``, so this reflects only the most recent
call, not a cumulative total across multiple invocations.
"""

_registration_failure_count: int = 0
"""Failure indicator for the most recent call to ``register_memory_message_types``.

Set to ``1`` if the last call failed, ``0`` if it succeeded (or if no call
has been made yet).  Reset at the start of each invocation, so this reflects
only the most recent call, not a cumulative total.
"""

_registry_lock = threading.Lock()
"""Guards concurrent access to _registry_ready, _registered_count,
and _registration_failure_count."""


def is_registry_ready() -> bool:
    """Return whether the memory message type registry is ready.

    This function is intended for health-check integrations.  It returns
    ``True`` only after ``register_memory_message_types`` has completed
    successfully (i.e., all expected types registered without error).
    """
    with _registry_lock:
        return _registry_ready


def get_registration_metrics() -> dict[str, int]:
    """Return current registration metrics for observability.

    Returns:
        Dict with ``registered_count``, ``failure_count``, and
        ``expected_count`` keys.
    """
    with _registry_lock:
        return {
            "registered_count": _registered_count,
            "failure_count": _registration_failure_count,
            "expected_count": EXPECTED_MESSAGE_TYPE_COUNT,
        }


def _reset_for_testing() -> None:
    """Reset all module-level observability globals to their initial state.

    This function is intended **exclusively for test fixtures** that need a
    clean slate between test cases.  It acquires ``_registry_lock`` and
    resets ``_registry_ready``, ``_registered_count``, and
    ``_registration_failure_count`` to their default values.

    Production code should never call this function.
    """
    global _registry_ready, _registered_count, _registration_failure_count  # noqa: PLW0603

    with _registry_lock:
        _registry_ready = False
        _registered_count = 0
        _registration_failure_count = 0


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

    .. note:: This function is **not thread-safe** for concurrent callers.
       It is designed to be invoked once during single-threaded plugin startup.

    On success the module-level readiness flag is set to ``True`` and
    ``_registered_count`` is set to the number of types registered.  On
    failure the readiness flag is set to ``False`` and
    ``_registration_failure_count`` is set to ``1``.  Both counters are
    reset at the start of each invocation, so they always reflect the
    outcome of the most recent call only.

    Args:
        registry: An unfrozen RegistryMessageType instance.

    Returns:
        List of registered message type names (for logging).

    Raises:
        ModelOnexError: If registry is already frozen.
        MessageTypeRegistryError: If any registration fails validation.
    """
    global _registry_ready, _registered_count, _registration_failure_count  # noqa: PLW0603

    # Reset readiness and failure counter at the start of each registration
    # attempt so that metrics reflect the last call only, and a retry after
    # a previous failure does not report stale readiness.
    # Lock protects reader functions (is_registry_ready, get_registered_count,
    # etc.) from seeing partially-written globals.  This does NOT make the
    # function safe for concurrent callers -- see module docstring.
    with _registry_lock:
        _registry_ready = False
        _registered_count = 0
        _registration_failure_count = 0

    registered: list[str] = []

    try:
        # =====================================================================
        # Consumed Kafka Event (from omniintelligence) -- EVENT category
        # =====================================================================

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

        # =====================================================================
        # Intent Storage Effect (orchestrator-invoked) -- COMMAND/EVENT
        # =====================================================================

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

        # =====================================================================
        # Memory Retrieval Effect -- COMMAND/EVENT
        # =====================================================================

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

        # =====================================================================
        # Memory Storage Effect (orchestrator-invoked) -- COMMAND/EVENT
        # =====================================================================

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

        # =====================================================================
        # Agent Coordinator Orchestrator -- COMMAND/EVENT
        # =====================================================================

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

    except Exception:
        with _registry_lock:
            _registration_failure_count = 1
            _registry_ready = False
            failure_count_snapshot = _registration_failure_count
        logger.exception(
            "Memory message type registration failed (registered=%d, failures=%d)",
            len(registered),
            failure_count_snapshot,
        )
        raise

    # Update metrics on success
    with _registry_lock:
        _registered_count = len(registered)
        _registry_ready = True
        count_snapshot = _registered_count
        ready_snapshot = _registry_ready
        failure_snapshot = _registration_failure_count

    logger.info(
        "Registered %d memory message types with RegistryMessageType "
        "(ready=%s, failures=%d)",
        count_snapshot,
        ready_snapshot,
        failure_snapshot,
    )

    return registered


__all__ = [
    "EXPECTED_MESSAGE_TYPE_COUNT",
    "MEMORY_DOMAIN",
    "_reset_for_testing",
    "get_registration_metrics",
    "is_registry_ready",
    "register_memory_message_types",
]
