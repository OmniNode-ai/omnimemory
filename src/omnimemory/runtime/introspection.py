# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory node introspection registration.

Publishes introspection events for all memory nodes during plugin
initialization, enabling them to be discovered by the registration
orchestrator in omnibase_infra.

This module bridges the gap between memory nodes (which run inside
the plugin lifecycle, not as standalone processes) and the platform
registration system (which discovers nodes via introspection events on
``onex.evt.platform.node-introspection.v1``).

Design Decisions:
    - Introspection is published per-node, not per-plugin. Each memory
      node gets its own introspection event with its own node_id.
    - The plugin owns the event bus reference and passes it to this module.
    - Heartbeat is enabled for effect nodes only (they have long-running
      consumers). Compute nodes are stateless and do not need heartbeats.
    - Node IDs are deterministic UUIDs derived from the node name using
      uuid5(NAMESPACE_DNS, node_name) for stable registration across restarts.

Related:
    - OMN-2216: Wire memory nodes into registration + introspection
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import NAMESPACE_DNS, UUID, uuid5

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.enums import EnumIntrospectionReason
from omnibase_infra.mixins.mixin_node_introspection import MixinNodeIntrospection
from omnibase_infra.models.discovery import ModelIntrospectionConfig

if TYPE_CHECKING:
    from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus

logger = logging.getLogger(__name__)

# Guard for single-call invariant on publish_memory_introspection.
# See the function docstring for rationale: calling it more than once orphans
# heartbeat tasks from the first call, leaking asyncio tasks.
# Async-safety: guarded by _introspection_lock (asyncio.Lock). A threading.Lock
# cannot prevent race conditions in async code because two coroutines on the
# same thread can both acquire a threading.Lock before either releases it.
# asyncio.Lock correctly serializes coroutines sharing an event loop.
_introspection_lock = asyncio.Lock()
_introspection_published: bool = False

# Standard DNS namespace for deterministic UUID5 generation.
# Node name prefixed with "omnimemory." ensures uniqueness across domains.
_NAMESPACE_MEMORY = NAMESPACE_DNS


# =============================================================================
# Memory Node Descriptors
# =============================================================================
# Each entry describes a node that should publish introspection events.
# Fields: node_name, node_type, version


class _NodeDescriptor:
    """Describes a memory node for introspection registration."""

    __slots__ = ("name", "node_type", "version")

    def __init__(
        self,
        name: str,
        node_type: EnumNodeKind,
        version: str = "1.0.0",
    ) -> None:
        self.name = name
        self.node_type = node_type
        self.version = version

    @property
    def node_id(self) -> UUID:
        """Deterministic node ID derived from node name."""
        return uuid5(_NAMESPACE_MEMORY, f"omnimemory.{self.name}")


MEMORY_NODES: tuple[_NodeDescriptor, ...] = (
    # Orchestrators
    _NodeDescriptor("node_memory_lifecycle_orchestrator", EnumNodeKind.ORCHESTRATOR),
    _NodeDescriptor("node_agent_coordinator_orchestrator", EnumNodeKind.ORCHESTRATOR),
    # Compute nodes
    _NodeDescriptor("node_similarity_compute", EnumNodeKind.COMPUTE),
    _NodeDescriptor("node_semantic_analyzer_compute", EnumNodeKind.COMPUTE),
    # Effect nodes
    _NodeDescriptor("node_intent_event_consumer_effect", EnumNodeKind.EFFECT),
    _NodeDescriptor("node_intent_query_effect", EnumNodeKind.EFFECT),
    _NodeDescriptor("node_intent_storage_effect", EnumNodeKind.EFFECT),
    _NodeDescriptor("node_memory_retrieval_effect", EnumNodeKind.EFFECT),
    _NodeDescriptor("node_memory_storage_effect", EnumNodeKind.EFFECT),
)


# =============================================================================
# Introspection Publisher
# =============================================================================


class MemoryNodeIntrospectionProxy(MixinNodeIntrospection):  # type: ignore[misc]  # omnibase_infra does not export py.typed
    """Proxy that uses MixinNodeIntrospection to publish on behalf of a node.

    Memory nodes are thin shells that run inside the plugin lifecycle.
    They do not own event bus references or background tasks. This proxy
    creates a lightweight MixinNodeIntrospection instance for each node
    to publish startup introspection and optionally run heartbeats.

    The proxy is intentionally minimal: it only provides the introspection
    mixin surface. It does not implement any node business logic.

    Note on mixin usage: This is an intentional proxy pattern, not a proper
    mixin usage. The proxy deliberately provides only the subset of the node
    interface that the mixin requires (the ``initialize_introspection`` call
    and the ``name`` property). The ``# type: ignore[misc]`` suppresses the
    mypy error from inheriting a mixin without a full node base class.
    """

    def __init__(
        self,
        descriptor: _NodeDescriptor,
        event_bus: ProtocolEventBus | None,
    ) -> None:
        config = ModelIntrospectionConfig(
            node_id=descriptor.node_id,
            node_type=descriptor.node_type,
            node_name=descriptor.name,
            event_bus=event_bus,
            version=descriptor.version,
        )
        self.initialize_introspection(config)
        self._descriptor = descriptor

    @property
    def name(self) -> str:
        """Return the node name from the descriptor."""
        return self._descriptor.name


@dataclass
class IntrospectionResult:
    """Result of memory introspection publishing.

    Holds both the list of registered node names and the proxy references
    needed to stop heartbeat tasks during shutdown.

    Design constraint on ``proxies``:
        The ``proxies`` list contains **only effect node proxies** that have
        running heartbeat background tasks (started via
        ``start_introspection_tasks``). Compute, orchestrator, and reducer
        nodes publish a one-shot STARTUP introspection event but do not start
        heartbeat tasks, so their proxies are not retained here.

        During shutdown (``publish_memory_shutdown``), fresh proxy
        instances are created for ALL nodes to publish SHUTDOWN events.
        Identity correlation between STARTUP and SHUTDOWN events is maintained
        through deterministic ``node_id`` values (UUID5 derived from the node
        name via ``_NodeDescriptor.node_id``), not through object identity.
        The registration orchestrator matches events by ``node_id``, so the
        distinct proxy instances produce correct correlation.

    Attributes:
        registered_nodes: Names of nodes that successfully published
            STARTUP introspection events.
        proxies: Effect node proxies with active heartbeat tasks. These
            must be passed to ``publish_memory_shutdown`` so their
            background tasks are stopped before the process exits.
    """

    registered_nodes: list[str] = field(default_factory=list)
    proxies: list[MemoryNodeIntrospectionProxy] = field(default_factory=list)


async def publish_memory_introspection(
    event_bus: ProtocolEventBus | None,
    *,
    correlation_id: UUID | None = None,
    enable_heartbeat: bool = True,
    heartbeat_interval_seconds: float = 30.0,
) -> IntrospectionResult:
    """Publish introspection events for all memory nodes.

    Creates a proxy MixinNodeIntrospection instance for each memory
    node and publishes a STARTUP introspection event. For effect nodes,
    optionally starts heartbeat tasks.

    **Single-call invariant**: This function MUST only be called once per
    process lifecycle with a real event bus. Each call creates new proxy
    instances and starts new heartbeat background tasks for effect nodes.
    Calling it more than once would orphan the previous proxies and their
    running heartbeat tasks, leading to leaked asyncio tasks and duplicate
    introspection events. The caller is responsible for retaining the
    returned ``IntrospectionResult`` and passing its ``proxies`` to
    ``publish_memory_shutdown()`` during teardown.

    If ``event_bus`` is None, the function is a no-op and does not set the
    single-call guard, allowing a subsequent call with a real event bus to
    succeed. This is intentional: a no-op call creates no proxies or
    heartbeat tasks, so there is nothing to orphan.

    Args:
        event_bus: Event bus implementing ProtocolEventBus for publishing
            introspection events. If None, introspection is skipped.
        correlation_id: Optional correlation ID for tracing.
        enable_heartbeat: Whether to start heartbeat tasks for effect nodes.
        heartbeat_interval_seconds: Interval between heartbeats in seconds.

    Returns:
        IntrospectionResult with registered node names and proxy references
        for lifecycle management.
    """
    global _introspection_published  # noqa: PLW0603
    async with _introspection_lock:
        if _introspection_published:
            raise RuntimeError(
                "publish_memory_introspection() has already been called "
                "with a real event bus. Calling it again would orphan heartbeat "
                "tasks from the first invocation. This violates the single-call "
                "invariant documented in the function docstring. "
                "(Note: calls with event_bus=None are exempt from this guard "
                "because they are no-ops that create no proxies or tasks.)"
            )

        if event_bus is None:
            # No-op path: intentionally does NOT set _introspection_published.
            # A no-op call creates no proxies or heartbeat tasks, so there is
            # nothing to orphan. A later call with a real event bus must still
            # be allowed to proceed.
            logger.info(
                "Skipping memory introspection: no event bus available "
                "(correlation_id=%s)",
                correlation_id,
            )
            return IntrospectionResult()

        # Set the guard atomically with the check to eliminate the TOCTOU
        # race window. If the subsequent work fails, the guard is reset in
        # the except block below so a retry is still possible.
        _introspection_published = True

    result = IntrospectionResult()

    try:
        for descriptor in MEMORY_NODES:
            try:
                proxy = MemoryNodeIntrospectionProxy(
                    descriptor=descriptor,
                    event_bus=event_bus,
                )

                success = await proxy.publish_introspection(
                    reason=EnumIntrospectionReason.STARTUP,
                    correlation_id=correlation_id,
                )

                if success:
                    result.registered_nodes.append(descriptor.name)
                    logger.debug(
                        "Published introspection for %s (node_id=%s, type=%s, "
                        "correlation_id=%s)",
                        descriptor.name,
                        descriptor.node_id,
                        descriptor.node_type,
                        correlation_id,
                    )

                    # Start heartbeat for effect nodes only
                    if enable_heartbeat and descriptor.node_type == EnumNodeKind.EFFECT:
                        await proxy.start_introspection_tasks(
                            enable_heartbeat=True,
                            heartbeat_interval_seconds=heartbeat_interval_seconds,
                            enable_registry_listener=False,
                        )
                        result.proxies.append(proxy)
                else:
                    logger.warning(
                        "Failed to publish introspection for %s (correlation_id=%s)",
                        descriptor.name,
                        correlation_id,
                    )

            except Exception as e:
                logger.warning(
                    "Error publishing introspection for %s: %s (correlation_id=%s)",
                    descriptor.name,
                    str(e),
                    correlation_id,
                    exc_info=True,
                    extra={
                        "error_type": type(e).__name__,
                        "node_name": descriptor.name,
                        "node_type": descriptor.node_type.value
                        if hasattr(descriptor.node_type, "value")
                        else str(descriptor.node_type),
                        "correlation_id": str(correlation_id),
                    },
                )
    except Exception:
        # Reset guard on failure so a retry is possible instead of
        # permanently blocking all future calls.
        async with _introspection_lock:
            _introspection_published = False
        raise

    logger.info(
        "Memory introspection published: %d/%d nodes (correlation_id=%s)",
        len(result.registered_nodes),
        len(MEMORY_NODES),
        correlation_id,
    )

    return result


async def publish_memory_shutdown(
    event_bus: ProtocolEventBus | None,
    *,
    proxies: list[MemoryNodeIntrospectionProxy] | None = None,
    correlation_id: UUID | None = None,
) -> None:
    """Publish shutdown introspection events for all memory nodes.

    Called during plugin shutdown to notify the registration orchestrator
    that memory nodes are going offline. Also stops any running
    heartbeat tasks on the provided proxies.

    Shutdown is best-effort: heartbeat tasks are stopped unconditionally
    (so nodes stop advertising liveness), but if ``event_bus`` is None
    the SHUTDOWN introspection events are skipped. This means nodes will
    appear offline from the heartbeat perspective while the registration
    orchestrator never receives an explicit SHUTDOWN event. The
    orchestrator handles this via heartbeat TTL expiry.

    Args:
        event_bus: Event bus for publishing shutdown events. If None,
            heartbeat tasks are still stopped but SHUTDOWN events are
            not published.
        proxies: Proxy instances from startup that may have running
            heartbeat tasks. If provided, their tasks are stopped
            before publishing shutdown events.
        correlation_id: Optional correlation ID for tracing.
    """
    # Stop heartbeat tasks on proxies that were started at init time
    if proxies:
        for proxy in proxies:
            try:
                await proxy.stop_introspection_tasks()
            except Exception as e:
                logger.debug(
                    "Error stopping introspection tasks for %s: %s",
                    proxy.name,
                    str(e),
                    exc_info=True,
                )

    if event_bus is None:
        await reset_introspection_guard()
        return

    # New proxies are created here because startup only retains proxies for
    # effect nodes (those with heartbeat tasks) in IntrospectionResult.proxies.
    # Non-effect nodes (compute, orchestrator, reducer) have no background tasks,
    # so their startup proxies are not stored. Creating lightweight proxies here
    # is simpler than refactoring startup to retain all proxies.
    #
    # Identity correlation: node_id (deterministic UUID5 from descriptor.name)
    # is the sole identity key used by the registration orchestrator to
    # correlate STARTUP and SHUTDOWN events for the same logical node. The
    # shutdown proxies created below are distinct object instances from the
    # startup proxies, but they produce the same node_id per descriptor,
    # ensuring the registration orchestrator correctly matches them.
    for descriptor in MEMORY_NODES:
        try:
            proxy = MemoryNodeIntrospectionProxy(
                descriptor=descriptor,
                event_bus=event_bus,
            )
            await proxy.publish_introspection(
                reason=EnumIntrospectionReason.SHUTDOWN,
                correlation_id=correlation_id,
            )
        except Exception as e:
            logger.debug(
                "Error publishing shutdown introspection for %s: %s",
                descriptor.name,
                str(e),
                exc_info=True,
            )

    # Reset the single-call guard so the plugin can be re-initialized in the
    # same process (tests, hot-reload).  This MUST happen after all shutdown
    # events have been published so the guard remains set while shutdown is
    # in progress, preventing a concurrent re-init from racing with shutdown.
    await reset_introspection_guard()


async def reset_introspection_guard() -> None:
    """Reset the single-call guard for publish_memory_introspection.

    Called during shutdown to allow re-initialization, and in tests for
    isolation between test cases that invoke
    ``publish_memory_introspection``.

    Async-safety: guarded by ``_introspection_lock`` (asyncio.Lock).
    """
    global _introspection_published  # noqa: PLW0603
    async with _introspection_lock:
        _introspection_published = False


__all__ = [
    "MEMORY_NODES",
    "IntrospectionResult",
    "MemoryNodeIntrospectionProxy",
    "publish_memory_introspection",
    "publish_memory_shutdown",
    "reset_introspection_guard",
]
