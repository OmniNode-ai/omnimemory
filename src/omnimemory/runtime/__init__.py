# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""OmniMemory Runtime Package.

Provides contract-driven topic discovery, protocol adapters, and dispatch
handlers for the OmniMemory domain.

This package contains:
    - Contract-driven topic discovery (contract_topics module)
    - Protocol adapters bridging infrastructure to handler protocols (adapters module)
    - Dispatch handlers routing events through MessageDispatchEngine (dispatch_handlers module)

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
    """Lazy-import dispatch handler symbols on first access.

    This avoids pulling omnibase_core.runtime into the module namespace at
    package import time while still allowing ``from omnimemory.runtime import
    create_memory_dispatch_engine`` to work.
    """
    _dispatch_symbols = {
        "DISPATCH_ALIAS_ARCHIVE_MEMORY",
        "DISPATCH_ALIAS_EXPIRE_MEMORY",
        "DISPATCH_ALIAS_INTENT_CLASSIFIED",
        "DISPATCH_ALIAS_INTENT_QUERY_REQUESTED",
        "DISPATCH_ALIAS_RUNTIME_TICK",
        "create_dispatch_callback",
        "create_memory_dispatch_engine",
    }
    if name in _dispatch_symbols:
        from omnimemory.runtime import dispatch_handlers

        return getattr(dispatch_handlers, name)
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
    "DISPATCH_ALIAS_RUNTIME_TICK",
    "create_dispatch_callback",
    "create_memory_dispatch_engine",
]
