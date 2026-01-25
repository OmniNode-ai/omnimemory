# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for intent_query_effect node.

Provides factory methods and metadata for the intent_query_effect node,
following ONEX registry patterns for dependency injection and service discovery.

Example::

    from omnimemory.nodes.intent_query_effect.registry import (
        RegistryIntentQueryEffect,
    )

    # Create handler via registry
    handler = await RegistryIntentQueryEffect.create_and_initialize(adapter)

    # Query node metadata
    node_type = RegistryIntentQueryEffect.get_node_type()  # "EFFECT"
    topics = RegistryIntentQueryEffect.get_kafka_topics()

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.
"""

from omnimemory.nodes.intent_query_effect.registry.registry_intent_query_effect import (
    RegistryIntentQueryEffect,
)

__all__ = ["RegistryIntentQueryEffect"]
