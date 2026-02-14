# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""OmniMemory Runtime Package.

Provides contract-driven topic discovery and protocol adapters for the
OmniMemory domain.

This package contains:
    - Contract-driven topic discovery (contract_topics module)
    - Protocol adapters bridging infrastructure to handler protocols (adapters module)

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
]
