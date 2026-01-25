# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for intent_query_effect node dependency injection.

This registry provides factory methods for creating HandlerIntentQuery
instances with their required dependencies resolved.

Following ONEX naming conventions:
    - File: registry_<node_name>.py
    - Class: Registry<NodeName>

The registry serves as the entry point for creating properly configured
handler instances, documenting required adapters, and providing
node metadata for introspection.

Related:
    - models/: Configuration and mapping utilities
    - handlers/: Handler implementation

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnimemory.handlers.adapters import AdapterIntentGraph
    from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery
    from omnimemory.nodes.intent_query_effect.models import (
        ModelHandlerIntentQueryConfig,
    )

__all__ = ["RegistryIntentQueryEffect"]


class RegistryIntentQueryEffect:
    """Infrastructure registry for intent_query_effect node.

    Provides factory methods for creating handler instances with
    proper dependency injection.

    This registry follows the ONEX infrastructure registry pattern:
        - Factory methods for handler creation with adapter injection
        - Adapter requirements documentation for validation
        - Node type classification for routing decisions
        - Capability listing for service discovery

    Example:
        >>> from omnimemory.handlers.adapters import (
        ...     AdapterIntentGraph,
        ...     ModelAdapterIntentGraphConfig,
        ... )
        >>> from omnimemory.nodes.intent_query_effect.registry import (
        ...     RegistryIntentQueryEffect,
        ... )
        >>>
        >>> # Create and initialize adapter
        >>> adapter_config = ModelAdapterIntentGraphConfig()
        >>> adapter = AdapterIntentGraph(adapter_config)
        >>> await adapter.initialize(connection_uri="bolt://localhost:7687")
        >>>
        >>> # Create handler via registry
        >>> handler = await RegistryIntentQueryEffect.create_and_initialize(adapter)

    .. versionadded:: 0.1.0
    """

    @staticmethod
    def create_handler(
        config: ModelHandlerIntentQueryConfig | None = None,
    ) -> HandlerIntentQuery:
        """Create a HandlerIntentQuery with configuration.

        Factory method that creates a HandlerIntentQuery instance
        with optional configuration. The handler must be initialized
        separately with an adapter before use.

        Args:
            config: Optional handler configuration. Uses defaults if not provided.

        Returns:
            Configured HandlerIntentQuery instance (not yet initialized).

        Example:
            >>> handler = RegistryIntentQueryEffect.create_handler()
            >>> await handler.initialize(adapter)

        .. versionadded:: 0.1.0
        """
        from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery

        return HandlerIntentQuery(config=config)

    @staticmethod
    async def create_and_initialize(
        adapter: AdapterIntentGraph,
        config: ModelHandlerIntentQueryConfig | None = None,
    ) -> HandlerIntentQuery:
        """Create and initialize a HandlerIntentQuery.

        Convenience method that creates and initializes in one call.

        Args:
            adapter: Initialized AdapterIntentGraph instance for database
                operations. Must be initialized before passing.
            config: Optional handler configuration. Uses defaults if not provided.

        Returns:
            Initialized HandlerIntentQuery ready to execute queries.

        Example:
            >>> handler = await RegistryIntentQueryEffect.create_and_initialize(
            ...     adapter=adapter,
            ...     config=ModelHandlerIntentQueryConfig(timeout_seconds=30.0),
            ... )
            >>> response = await handler.execute(request)

        .. versionadded:: 0.1.0
        """
        from omnimemory.nodes.intent_query_effect.handlers import HandlerIntentQuery

        handler = HandlerIntentQuery(config=config)
        await handler.initialize(adapter)
        return handler

    @staticmethod
    def get_required_adapters() -> list[str]:
        """Get list of adapters required by this node.

        Returns the adapter class names that must be provided when
        initializing handlers from this registry.

        Returns:
            List of adapter class names required for node operation.

        Example:
            >>> adapters = RegistryIntentQueryEffect.get_required_adapters()
            >>> print(adapters)
            ['AdapterIntentGraph']

        .. versionadded:: 0.1.0
        """
        return ["AdapterIntentGraph"]

    @staticmethod
    def get_node_type() -> str:
        """Get the ONEX node type classification.

        Returns the ONEX node archetype for this node, used for
        routing decisions and execution context selection.

        Returns:
            Node type string ("EFFECT").

        Note:
            EFFECT nodes perform external I/O operations and should
            be treated as side-effecting by the runtime.

        .. versionadded:: 0.1.0
        """
        return "EFFECT"

    @staticmethod
    def get_node_name() -> str:
        """Get the canonical node name.

        Returns:
            The node name identifier.

        .. versionadded:: 0.1.0
        """
        return "intent_query_effect"

    @staticmethod
    def get_capabilities() -> list[str]:
        """Get list of capabilities provided by this node.

        Returns capability identifiers that can be used for service
        discovery and feature detection.

        Returns:
            List of capability identifiers.

        .. versionadded:: 0.1.0
        """
        return [
            "intent_distribution_query",
            "intent_session_query",
            "intent_recent_query",
            "event_driven_response",
        ]

    @staticmethod
    def get_supported_query_types() -> list[str]:
        """Get list of supported query types.

        Returns the query type values that can be passed in
        intent query request events.

        Returns:
            List of supported query type strings.

        .. versionadded:: 0.1.0
        """
        return ["distribution", "session", "recent"]

    @staticmethod
    def get_topic_suffixes() -> dict[str, str]:
        """Get Kafka topic suffixes for this node.

        Returns topic SUFFIXES (not full topics). Runtime composes
        full topics by adding env prefix:
            full_topic = f"{topic_env_prefix}.{suffix}"

        Example full topics (with "dev" env prefix):
            - dev.onex.cmd.omnimemory.intent-query-requested.v1
            - dev.onex.evt.omnimemory.intent-query-response.v1

        Returns:
            Dictionary with 'subscribe' and 'publish' topic suffixes.

        .. versionadded:: 0.1.0
        """
        return {
            "subscribe": "onex.cmd.omnimemory.intent-query-requested.v1",
            "publish": "onex.evt.omnimemory.intent-query-response.v1",
        }

    @staticmethod
    def get_invocation_mode() -> str:
        """Get the invocation mode for this node.

        Returns:
            "subscription" - node subscribes to topic, runtime dispatches
            "orchestrator" - node is invoked by orchestrator, no subscription

        .. versionadded:: 0.1.0
        """
        return "subscription"

    @staticmethod
    def get_supported_operations() -> list[str]:
        """Get list of operations supported by this node.

        Returns:
            List of operation identifiers.

        .. versionadded:: 0.1.0
        """
        return [
            "query_intent_distribution",
            "query_session_intents",
            "query_recent_intents",
        ]

    @staticmethod
    def get_backends() -> list[str]:
        """Get list of backend types this node interacts with.

        Returns:
            List of backend identifiers.

        .. versionadded:: 0.1.0
        """
        return ["memgraph", "kafka"]
