# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""OmniMemory Runtime Package.

Provides contract-driven topic discovery, protocol adapters, dispatch
handlers, and the PluginMemory domain plugin for the OmniMemory domain.

This package contains:
    - Contract-driven topic discovery (contract_topics module)
    - Protocol adapters bridging infrastructure to handler protocols (adapters module)
    - Dispatch handlers routing events through MessageDispatchEngine (dispatch_handlers module)
    - PluginMemory domain plugin for kernel bootstrap (plugin module)
    - Handler wiring for plugin initialization (wiring module)

Usage:
    from omnimemory.runtime.contract_topics import (
        collect_subscribe_topics_from_contracts,
        collect_publish_topics_for_dispatch,
        collect_all_publish_topics,
        canonical_topic_to_dispatch_alias,
    )

    from omnimemory.runtime.adapters import (
        AdapterKafkaPublisher,
        ProtocolEventBusPublish,
        ProtocolEventBusHealthCheck,
        ProtocolEventBusLifecycle,
    )

    from omnimemory.runtime.dispatch_handlers import (
        create_memory_dispatch_engine,
        create_dispatch_callback,
    )

    from omnimemory.runtime.plugin import PluginMemory
    from omnimemory.runtime.wiring import wire_memory_handlers
"""

# Eager imports: adapters module is lightweight and used by all handler consumers.
from omnimemory.runtime.adapters import (
    AdapterKafkaPublisher,
    ProtocolEventBusHealthCheck,
    ProtocolEventBusLifecycle,
    ProtocolEventBusPublish,
    create_default_event_bus,
)
from omnimemory.runtime.contract_topics import (
    canonical_topic_to_dispatch_alias,
    collect_all_publish_topics,
    collect_publish_topics_for_dispatch,
    collect_subscribe_topics_from_contracts,
)

# Dispatch handlers are lazy-imported to avoid pulling in omnibase_core.runtime
# at package import time. Import from omnimemory.runtime.dispatch_handlers directly,
# or access via __all__ (the symbols are provided by __getattr__ below).


def __getattr__(name: str) -> object:
    """Lazy-import runtime symbols on first access.

    This avoids pulling heavier modules into the namespace at package import
    time while still allowing direct imports of dispatch, plugin, wiring,
    introspection, and message-type registration symbols.
    """
    _dispatch_symbols = {
        "DISPATCH_ALIAS_ARCHIVE_MEMORY",
        "DISPATCH_ALIAS_EXPIRE_MEMORY",
        "DISPATCH_ALIAS_INTENT_CLASSIFIED",
        "DISPATCH_ALIAS_INTENT_QUERY_REQUESTED",
        "DISPATCH_ALIAS_MEMORY_RETRIEVAL_REQUESTED",
        "DISPATCH_ALIAS_RUNTIME_TICK",
        "create_dispatch_callback",
        "create_memory_dispatch_engine",
    }
    if name in _dispatch_symbols:
        from omnimemory.runtime import dispatch_handlers

        return getattr(dispatch_handlers, name)

    _plugin_symbols = {
        "MEMORY_SUBSCRIBE_TOPICS",
        "PluginMemory",
    }
    if name in _plugin_symbols:
        from omnimemory.runtime import plugin

        return getattr(plugin, name)

    _wiring_symbols = {
        "wire_memory_handlers",
    }
    if name in _wiring_symbols:
        from omnimemory.runtime import wiring

        return getattr(wiring, name)

    _introspection_symbols = {
        "MEMORY_NODES",
        "IntrospectionResult",
        "MemoryNodeIntrospectionProxy",
        "publish_memory_introspection",
        "publish_memory_shutdown",
        "reset_introspection_guard",
    }
    if name in _introspection_symbols:
        from omnimemory.runtime import introspection

        return getattr(introspection, name)

    _message_type_symbols = {
        "EXPECTED_MESSAGE_TYPE_COUNT",
        "MEMORY_DOMAIN",
        "get_registration_metrics",
        "is_registry_ready",
        "register_memory_message_types",
    }
    if name in _message_type_symbols:
        from omnimemory.runtime import message_type_registration

        return getattr(message_type_registration, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Protocol adapters (ARCH-002)
    "AdapterKafkaPublisher",
    "ProtocolEventBusHealthCheck",
    "ProtocolEventBusLifecycle",
    "ProtocolEventBusPublish",
    "create_default_event_bus",
    # Contract-driven topic discovery
    "canonical_topic_to_dispatch_alias",
    "collect_all_publish_topics",
    "collect_publish_topics_for_dispatch",
    "collect_subscribe_topics_from_contracts",
    # Dispatch handlers (lazy-imported via __getattr__)
    "DISPATCH_ALIAS_ARCHIVE_MEMORY",
    "DISPATCH_ALIAS_EXPIRE_MEMORY",
    "DISPATCH_ALIAS_INTENT_CLASSIFIED",
    "DISPATCH_ALIAS_INTENT_QUERY_REQUESTED",
    "DISPATCH_ALIAS_MEMORY_RETRIEVAL_REQUESTED",
    "DISPATCH_ALIAS_RUNTIME_TICK",
    "create_dispatch_callback",
    "create_memory_dispatch_engine",
    # Plugin (lazy-imported via __getattr__)
    "MEMORY_SUBSCRIBE_TOPICS",
    "PluginMemory",
    # Wiring (lazy-imported via __getattr__)
    "wire_memory_handlers",
    # Introspection (lazy-imported via __getattr__)
    "MEMORY_NODES",
    "IntrospectionResult",
    "MemoryNodeIntrospectionProxy",
    "publish_memory_introspection",
    "publish_memory_shutdown",
    "reset_introspection_guard",
    # Message type registration (lazy-imported via __getattr__)
    "EXPECTED_MESSAGE_TYPE_COUNT",
    "MEMORY_DOMAIN",
    "get_registration_metrics",
    "is_registry_ready",
    "register_memory_message_types",
]
