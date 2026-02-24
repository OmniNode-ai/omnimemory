# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory domain plugin for kernel-level initialization.

This module provides the PluginMemory class, which implements
ProtocolDomainPlugin for the Memory domain. It encapsulates all
Memory-specific initialization code for kernel bootstrap.

The plugin handles:
    - Handler importability verification during wiring
    - MessageDispatchEngine wiring for topic-based routing (OMN-2215)
    - Kafka topic subscriptions for memory events
    - Graceful shutdown and resource cleanup

Design Pattern:
    The plugin pattern enables the kernel to remain generic while allowing
    domain-specific initialization to be encapsulated in domain modules.
    This follows the dependency inversion principle - the kernel depends
    on the abstract ProtocolDomainPlugin protocol, not this concrete class.

Topic Discovery (OMN-2213):
    Subscribe topics are declared in individual effect node ``contract.yaml``
    files under ``event_bus.subscribe_topics`` and collected at import time
    via ``collect_subscribe_topics_from_contracts()``.  There are no
    hardcoded topic lists in this module.

Configuration:
    The plugin activates based on environment variables:
    - OMNIMEMORY_ENABLED: Set to any truthy value to activate the plugin.
      When not set, the plugin is inactive (graceful degradation).

Example Usage:
    ```python
    from omnimemory.runtime.plugin import PluginMemory
    from omnibase_infra.runtime.protocol_domain_plugin import (
        ModelDomainPluginConfig,
        RegistryDomainPlugin,
    )

    # Register plugin
    registry = RegistryDomainPlugin()
    registry.register(PluginMemory())

    # During kernel bootstrap
    config = ModelDomainPluginConfig(container=container, event_bus=event_bus, ...)
    plugin = registry.get("memory")

    if plugin and plugin.should_activate(config):
        await plugin.initialize(config)
        await plugin.wire_handlers(config)
        await plugin.wire_dispatchers(config)
        await plugin.start_consumers(config)
    ```

Related:
    - OMN-2216: Phase 5 -- Runtime plugin PluginMemory
    - OMN-2215: Phase 4 -- MessageDispatchEngine integration
    - OMN-2213: Phase 2 -- Contract-driven topic discovery
    - omniintelligence/runtime/plugin.py (reference implementation)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus
    from omnibase_core.runtime.runtime_message_dispatch import MessageDispatchEngine
    from omnibase_infra.runtime.registry import RegistryMessageType

    from omnimemory.runtime.introspection import MemoryNodeIntrospectionProxy

from omnibase_infra.runtime.protocol_domain_plugin import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
    ProtocolDomainPlugin,
)

from omnimemory.runtime.contract_topics import (
    canonical_topic_to_dispatch_alias,
    collect_subscribe_topics_from_contracts,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Memory Kafka Topics (contract-driven, OMN-2213)
# =============================================================================
# Topics are declared in individual effect node contract.yaml files under
# event_bus.subscribe_topics.  This list is populated at import time by
# scanning those contracts via importlib.resources.
#
# Source contracts:
#   - intent_event_consumer_effect/contract.yaml
#   - intent_query_effect/contract.yaml
#   - intent_storage_effect/contract.yaml
#   - memory_retrieval_effect/contract.yaml
#   - memory_storage_effect/contract.yaml
#   - memory_lifecycle_orchestrator/contract.yaml

MEMORY_SUBSCRIBE_TOPICS: list[str] = collect_subscribe_topics_from_contracts()
"""All input topics the memory plugin subscribes to (contract-driven)."""


class PluginMemory:
    """Memory domain plugin for ONEX kernel initialization.

    Provides memory management pipeline integration:
    - Intent event consumption and storage
    - Intent query processing
    - Memory lifecycle orchestration (runtime-tick, archive, expire)
    - Subscription management

    Resources Created:
        - Memory domain handlers (via wiring module)
        - MessageDispatchEngine for topic-based routing

    Thread Safety:
        This class is NOT thread-safe. The kernel calls plugin methods
        sequentially during bootstrap. Resource access during runtime
        should be via container-resolved handlers.

    Attributes:
        _unsubscribe_callbacks: Callbacks for Kafka unsubscription
        _shutdown_in_progress: Guard against concurrent shutdown calls
        _services_registered: List of registered handler names
        _dispatch_engine: MessageDispatchEngine for topic routing
    """

    def __init__(self) -> None:
        """Initialize the plugin with empty state."""
        self._unsubscribe_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._shutdown_in_progress: bool = False
        self._services_registered: list[str] = []
        self._dispatch_engine: MessageDispatchEngine | None = None
        self._message_type_registry: RegistryMessageType | None = None
        self._event_bus: ProtocolEventBus | None = None
        self._introspection_nodes: list[str] = []
        self._introspection_proxies: list[MemoryNodeIntrospectionProxy] = []

    @property
    def message_type_registry(self) -> RegistryMessageType | None:
        """Return the message type registry (for external access)."""
        return self._message_type_registry

    @property
    def plugin_id(self) -> str:
        """Return unique identifier for this plugin."""
        return "memory"

    @property
    def display_name(self) -> str:
        """Return human-readable name for this plugin."""
        return "Memory"

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        """Check if Memory should activate based on environment.

        Returns True if OMNIMEMORY_ENABLED is set, indicating the memory
        domain should be activated for this kernel instance.

        Args:
            config: Plugin configuration (not used for this check).

        Returns:
            True if OMNIMEMORY_ENABLED environment variable is set.
        """
        enabled = os.getenv("OMNIMEMORY_ENABLED")
        if not enabled:
            logger.debug(
                "Memory plugin inactive: OMNIMEMORY_ENABLED not set "
                "(correlation_id=%s)",
                config.correlation_id,
            )
            return False
        return True

    async def initialize(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Initialize Memory resources.

        The memory domain does not require a dedicated database pool at
        plugin level (unlike intelligence which uses asyncpg directly).
        Memory handlers use container-injected adapters for storage.

        This method validates the environment and prepares for handler
        wiring.

        Args:
            config: Plugin configuration with container and correlation_id.

        Returns:
            Result with resources_created list on success.
        """
        start_time = time.time()
        correlation_id = config.correlation_id

        try:
            resources_created: list[str] = []

            # Register memory message types (OMN-2217)
            # NOTE: This creates a plugin-local registry for the memory domain
            # only.  Cross-domain validation and duplicate detection require a
            # kernel-level shared registry (future enhancement).
            from omnibase_infra.runtime.registry import RegistryMessageType

            from omnimemory.runtime.message_type_registration import (
                register_memory_message_types,
            )

            registry = RegistryMessageType()
            registered_types = register_memory_message_types(registry)
            registry.freeze()

            # Validate startup -- log warnings but do not fail init
            warnings = registry.validate_startup()
            if warnings:
                for warning in warnings:
                    logger.warning(
                        "Message type registry warning: %s (correlation_id=%s)",
                        warning,
                        correlation_id,
                    )

            self._message_type_registry = registry
            resources_created.append("message_type_registry")

            logger.info(
                "Memory message type registry created and frozen "
                "(types=%d, warnings=%d, correlation_id=%s)",
                len(registered_types),
                len(warnings),
                correlation_id,
            )

            duration = time.time() - start_time

            logger.info(
                "Memory plugin initialized (correlation_id=%s)",
                correlation_id,
            )

            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Memory plugin initialized",
                resources_created=resources_created,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "Failed to initialize Memory plugin (correlation_id=%s)",
                correlation_id,
                extra={"error_type": type(e).__name__},
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=str(e),
                duration_seconds=duration,
            )

    async def wire_handlers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Register Memory handlers with the container.

        Delegates to wire_memory_handlers from the wiring module to
        verify and register memory domain handlers.

        Args:
            config: Plugin configuration with container.

        Returns:
            Result with services_registered list on success.
        """
        from omnimemory.runtime.wiring import wire_memory_handlers

        start_time = time.time()
        correlation_id = config.correlation_id

        try:
            self._services_registered = await wire_memory_handlers(
                config=config,
            )
            duration = time.time() - start_time

            logger.info(
                "Memory handlers wired (correlation_id=%s)",
                correlation_id,
                extra={"services": self._services_registered},
            )

            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Memory handlers wired",
                services_registered=self._services_registered,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "Failed to wire Memory handlers (correlation_id=%s)",
                correlation_id,
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=str(e),
                duration_seconds=duration,
            )

    async def wire_dispatchers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Wire memory domain dispatchers with MessageDispatchEngine.

        Creates the dispatch engine with all memory domain handlers and
        routes registered. The engine handles topic-based routing for
        incoming events.

        Dispatchers registered:
            1. intent-classified handler (1 route: intent-classified.v1 events)
            2. intent-query handler (1 route: intent-query-requested.v1 commands)
            3. memory-retrieval handler (1 route: memory-retrieval-requested.v1 -- fail-fast)
            4. lifecycle handler (3 routes: runtime-tick, archive, expire -- fail-fast)

        The dispatch engine requires intent_consumer and intent_query_handler
        protocol implementations. These are created as stub instances for
        initial wiring; full handler instances are resolved from the container
        at dispatch time.

        Args:
            config: Plugin configuration.

        Returns:
            Result indicating success/failure and dispatchers registered.
        """
        from omnimemory.runtime.contract_topics import (
            collect_publish_topics_for_dispatch,
        )
        from omnimemory.runtime.dispatch_handlers import (
            create_memory_dispatch_engine,
        )

        start_time = time.time()
        correlation_id = config.correlation_id

        try:
            # Create stub protocol implementations for dispatch engine
            # The dispatch handlers bridge to the actual handler logic
            intent_consumer = _StubIntentEventConsumer()
            intent_query_handler = _StubIntentQueryHandler()

            # Kafka publisher: optional (graceful degradation in handlers).
            # config.event_bus may be None; hasattr safely returns False for None.
            publish_callback = None
            if hasattr(config.event_bus, "publish"):
                from omnimemory.runtime.adapters import AdapterKafkaPublisher

                kafka_publisher = AdapterKafkaPublisher(config.event_bus)

                async def _publish(topic: str, value: dict[str, object]) -> None:
                    await kafka_publisher.publish(topic, None, value)

                publish_callback = _publish

            # Read publish topics from contract.yaml declarations.
            # Runs in a thread because the helper performs synchronous
            # filesystem I/O (importlib.resources.files + yaml.safe_load).
            publish_topics = await asyncio.to_thread(
                collect_publish_topics_for_dispatch,
            )

            self._dispatch_engine = create_memory_dispatch_engine(
                intent_consumer=intent_consumer,
                intent_query_handler=intent_query_handler,
                publish_callback=publish_callback,
                publish_topics=publish_topics,
            )

            # Store event_bus reference for introspection publishing.
            # NOTE: This reference is captured at wire time and used during
            # shutdown. The caller is responsible for keeping the event bus
            # alive until shutdown completes.
            self._event_bus = config.event_bus

            # Publish introspection events for all memory nodes
            from omnimemory.runtime.introspection import (
                publish_memory_introspection,
            )

            introspection_result = await publish_memory_introspection(
                event_bus=config.event_bus,
                correlation_id=correlation_id,
            )
            self._introspection_nodes = introspection_result.registered_nodes
            self._introspection_proxies = introspection_result.proxies

            duration = time.time() - start_time
            logger.info(
                "Memory dispatch engine wired "
                "(routes=%d, handlers=%d, kafka=%s, introspection=%d, "
                "correlation_id=%s)",
                self._dispatch_engine.route_count,
                self._dispatch_engine.handler_count,
                publish_callback is not None,
                len(self._introspection_nodes),
                correlation_id,
                extra={"publish_topics": publish_topics},
            )

            resources_created = ["dispatch_engine"]
            if self._introspection_nodes:
                resources_created.append("node_introspection")

            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Memory dispatch engine wired",
                resources_created=resources_created,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "Failed to wire memory dispatch engine (correlation_id=%s)",
                correlation_id,
            )
            # Clean up partially-captured state to avoid stale references.
            # Stop heartbeat tasks on any introspection proxies that were
            # started before the failure, then reset the single-call guard
            # so a retry is not permanently blocked.
            for proxy in self._introspection_proxies:
                try:
                    await proxy.stop_introspection_tasks()
                except Exception as stop_error:
                    logger.debug(
                        "Error stopping introspection tasks for %s during "
                        "wire_dispatchers cleanup: %s (correlation_id=%s)",
                        proxy.name,
                        str(stop_error),
                        correlation_id,
                    )

            from omnimemory.runtime.introspection import (
                reset_introspection_guard,
            )

            await reset_introspection_guard()

            self._event_bus = None
            self._introspection_nodes = []
            self._introspection_proxies = []
            self._dispatch_engine = None
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=str(e),
                duration_seconds=duration,
            )

    async def start_consumers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Start memory event consumers.

        Subscribes to memory input topics via MessageDispatchEngine.
        All topics are routed through the dispatch engine. If the
        dispatch engine is not wired, consumers are not started
        (returns skipped).

        Args:
            config: Plugin configuration with event_bus.

        Returns:
            Result with unsubscribe_callbacks for cleanup.
        """
        start_time = time.time()
        correlation_id = config.correlation_id

        # Strict gating: no dispatch engine = no consumers
        if self._dispatch_engine is None:
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="Dispatch engine not wired; consumers not started",
            )

        # Duck typing: check for subscribe capability
        if not hasattr(config.event_bus, "subscribe"):
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="Event bus does not support subscribe",
            )

        try:
            # Build per-topic handler map (dispatch engine guaranteed non-None)
            topic_handlers = self._build_topic_handlers(correlation_id)

            unsubscribe_callbacks: list[Callable[[], Awaitable[None]]] = []

            for topic in MEMORY_SUBSCRIBE_TOPICS:
                handler = topic_handlers[topic]
                logger.info(
                    "Subscribing to memory topic: %s "
                    "(mode=dispatch_engine, correlation_id=%s)",
                    topic,
                    correlation_id,
                )
                unsub = await config.event_bus.subscribe(
                    topic=topic,
                    group_id=f"{config.consumer_group}-memory",
                    on_message=handler,
                )
                unsubscribe_callbacks.append(unsub)

            self._unsubscribe_callbacks = unsubscribe_callbacks

            duration = time.time() - start_time
            logger.info(
                "Memory consumers started: %d topics "
                "(all dispatched, correlation_id=%s)",
                len(MEMORY_SUBSCRIBE_TOPICS),
                correlation_id,
            )

            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message=(
                    f"Memory consumers started "
                    f"({len(MEMORY_SUBSCRIBE_TOPICS)} dispatched)"
                ),
                duration_seconds=duration,
                unsubscribe_callbacks=unsubscribe_callbacks,
            )

        except Exception as e:
            # Rollback any successful subscriptions before returning failure.
            for unsub in unsubscribe_callbacks:
                try:
                    await unsub()
                except Exception as rollback_err:
                    logger.debug(
                        "Failed to rollback subscription during "
                        "start_consumers error handling: %s "
                        "(correlation_id=%s)",
                        rollback_err,
                        correlation_id,
                    )

            duration = time.time() - start_time
            logger.exception(
                "Failed to start memory consumers (correlation_id=%s)",
                correlation_id,
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=str(e),
                duration_seconds=duration,
            )

    def _build_topic_handlers(
        self,
        correlation_id: object,
    ) -> dict[str, Callable[[object], Awaitable[None]]]:
        """Build handler map for each memory topic.

        Returns a dict mapping topic -> async callback. All memory
        topics are routed through the dispatch engine. This method must
        only be called when ``self._dispatch_engine`` is not None
        (enforced by ``start_consumers()``).

        Topic -> dispatch alias conversion is handled generically by
        ``canonical_topic_to_dispatch_alias`` (OMN-2213).

        Args:
            correlation_id: Correlation ID for tracing.

        Returns:
            Dict mapping each MEMORY_SUBSCRIBE_TOPICS entry to a handler.

        Raises:
            RuntimeError: If dispatch engine is not wired (invariant violation).
        """
        if self._dispatch_engine is None:
            raise RuntimeError(
                "_build_topic_handlers called without dispatch engine "
                f"(correlation_id={correlation_id})"
            )

        from omnimemory.runtime.dispatch_handlers import (
            create_dispatch_callback,
        )

        handlers: dict[str, Callable[[object], Awaitable[None]]] = {}

        for topic in MEMORY_SUBSCRIBE_TOPICS:
            dispatch_alias = canonical_topic_to_dispatch_alias(topic)
            handlers[topic] = create_dispatch_callback(
                engine=self._dispatch_engine,
                dispatch_topic=dispatch_alias,
            )

        return handlers

    async def shutdown(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Clean up Memory resources.

        Unsubscribes from topics and clears internal state. Guards
        against concurrent shutdown calls via _shutdown_in_progress flag.

        Args:
            config: Plugin configuration.

        Returns:
            Result indicating cleanup success/failure.
        """
        # Re-entrancy guard (not concurrency-safe; class is single-threaded per docstring)
        if self._shutdown_in_progress:
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="Shutdown already in progress",
            )
        self._shutdown_in_progress = True

        try:
            return await self._do_shutdown(config)
        finally:
            self._shutdown_in_progress = False

    async def _do_shutdown(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Internal shutdown implementation.

        Args:
            config: Plugin configuration.

        Returns:
            Result indicating cleanup success/failure.
        """
        start_time = time.time()
        correlation_id = config.correlation_id
        errors: list[str] = []

        # Publish shutdown introspection for all memory nodes.
        # Gate on _event_bus (set in wire_dispatchers before introspection
        # is attempted), NOT on _introspection_nodes.
        if self._event_bus is not None:
            try:
                from omnimemory.runtime.introspection import (
                    publish_memory_shutdown,
                )

                await publish_memory_shutdown(
                    event_bus=self._event_bus,
                    proxies=self._introspection_proxies,
                    correlation_id=correlation_id,
                )
            except Exception as shutdown_intro_error:
                errors.append(f"introspection_shutdown: {shutdown_intro_error}")
                logger.warning(
                    "Failed to publish shutdown introspection: %s (correlation_id=%s)",
                    shutdown_intro_error,
                    correlation_id,
                    exc_info=True,
                    extra={"error_type": type(shutdown_intro_error).__name__},
                )
        else:
            logger.debug(
                "Introspection shutdown skipped: wire_dispatchers was never "
                "called or did not capture event_bus "
                "(correlation_id=%s)",
                correlation_id,
            )

        # Unsubscribe from topics
        for unsub in self._unsubscribe_callbacks:
            try:
                await unsub()
            except Exception as unsub_error:
                errors.append(f"unsubscribe: {unsub_error}")
                logger.warning(
                    "Failed to unsubscribe memory consumer: %s (correlation_id=%s)",
                    unsub_error,
                    correlation_id,
                    exc_info=True,
                    extra={"error_type": type(unsub_error).__name__},
                )
        self._unsubscribe_callbacks = []

        self._services_registered = []
        self._dispatch_engine = None
        self._message_type_registry = None
        self._event_bus = None
        self._introspection_nodes = []
        self._introspection_proxies = []

        duration = time.time() - start_time

        if errors:
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message="; ".join(errors),
                duration_seconds=duration,
            )

        logger.debug(
            "Memory resources cleaned up (correlation_id=%s)",
            correlation_id,
        )

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message="Memory resources cleaned up",
            duration_seconds=duration,
        )

    def get_status_line(self) -> str:
        """Get status line for kernel banner.

        Returns:
            Status string indicating enabled state.
        """
        enabled = os.getenv("OMNIMEMORY_ENABLED", "")
        if not enabled:
            return "disabled"
        topics_count = len(MEMORY_SUBSCRIBE_TOPICS)
        return f"enabled ({topics_count} topics)"


# =============================================================================
# Stub Implementations for Dispatch Engine Wiring
# =============================================================================
# The dispatch engine requires protocol implementations at creation time.
# These stubs satisfy the protocol interface.  The actual dispatch handlers
# (create_intent_classified_dispatch_handler, etc.) contain the real business
# logic that processes events.


class _StubIntentEventConsumer:
    """Stub intent event consumer for dispatch engine wiring.

    The dispatch engine's bridge handler
    (create_intent_classified_dispatch_handler) delegates to
    consumer._handle_message().  This stub provides that interface.
    The real processing happens in the bridge handler itself.
    """

    async def _handle_message(
        self, message: dict[str, object], *, retry_count: int = 0
    ) -> None:
        """Process an intent-classified event message.

        In production, this would delegate to the full
        HandlerIntentEventConsumer.  For initial plugin wiring, the
        bridge handler in dispatch_handlers.py handles the core logic.
        """
        logger.warning(
            "Stub intent event consumer received message (keys=%s)",
            list(message.keys()) if isinstance(message, dict) else "N/A",
        )


class _StubIntentQueryHandler:
    """Stub intent query handler for dispatch engine wiring.

    The dispatch engine's bridge handler
    (create_intent_query_dispatch_handler) delegates to
    handler.execute().  This stub provides that interface.
    """

    async def execute(self, request: object) -> object:
        """Process an intent query request.

        In production, this would delegate to the full
        HandlerIntentQuery.  For initial plugin wiring, the bridge
        handler in dispatch_handlers.py handles the core logic.

        Returns:
            Empty dict as placeholder response.
        """
        logger.warning(
            "Stub intent query handler received request (type=%s)",
            type(request).__name__,
        )
        return {}


# Verify protocol compliance at module load time
_: ProtocolDomainPlugin = PluginMemory()

__all__: list[str] = [
    "MEMORY_SUBSCRIBE_TOPICS",
    "PluginMemory",
]
