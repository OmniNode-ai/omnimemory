# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for intent_query_effect node.

Provides factory methods and metadata for the intent_query_effect node,
following ONEX registry patterns for dependency injection and service discovery.

Topic Naming:
    This node uses topic SUFFIXES (not full topics). Runtime composes
    full topics by adding env prefix::

        full_topic = f"{topic_env_prefix}.{suffix}"

    Example with "dev" env prefix:
        - dev.onex.cmd.omnimemory.intent-query-requested.v1
        - dev.onex.evt.omnimemory.intent-query-response.v1

Example::

    from omnimemory.nodes.intent_query_effect.registry import (
        RegistryIntentQueryEffect,
    )

    # Create handler via registry
    handler = await RegistryIntentQueryEffect.create_and_initialize(adapter)

    # Query node metadata
    node_type = RegistryIntentQueryEffect.get_node_type()  # "EFFECT"
    suffixes = RegistryIntentQueryEffect.get_topic_suffixes()
    # {"subscribe": "onex.cmd...", "publish": "onex.evt..."}

.. versionadded:: 0.1.0
    Initial implementation for OMN-1504.
"""

from omnimemory.nodes.intent_query_effect.registry.registry_intent_query_effect import (
    RegistryIntentQueryEffect,
)

__all__ = ["RegistryIntentQueryEffect"]
