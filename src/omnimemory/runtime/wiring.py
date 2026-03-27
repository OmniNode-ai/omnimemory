# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory domain handler wiring for kernel initialization.

validates, and (where possible) instantiates memory domain handlers
during plugin initialization.

Handlers Wired:
    - HandlerIntentEventConsumer: Intent event consumer (class, needs deps)
    - HandlerIntentQuery: Intent query handler (class, needs container)
    - HandlerSubscription: Subscription management handler (class, needs container)

Note:
    OmniMemory handlers require runtime dependencies (storage adapters,
    containers, event buses) and cannot be fully instantiated at wiring
    time.  This module verifies importability only.  Full instantiation
    happens when the kernel creates handler instances with injected deps.

Related:
    - OMN-2216: Phase 5 -- Runtime plugin PluginMemory
    - omniintelligence/runtime/wiring.py (reference implementation)
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.runtime.models import ModelDomainPluginConfig

logger = logging.getLogger(__name__)

# Handler import specifications: (module_path, attribute_name, is_class)
# Classes with no-arg constructors are instantiated; everything else is
# verified importable only.  OmniMemory handlers require runtime deps
# (containers, adapters) so most are verify-only.
_HANDLER_SPECS: list[tuple[str, str, bool]] = [
    (
        "omnimemory.nodes.node_intent_event_consumer_effect.handler_intent_event_consumer",
        "HandlerIntentEventConsumer",
        False,  # Requires config + storage_adapter
    ),
    (
        "omnimemory.nodes.node_intent_query_effect.handlers.handler_intent_query",
        "HandlerIntentQuery",
        False,  # Requires ModelONEXContainer
    ),
    (
        "omnimemory.handlers.handler_subscription",
        "HandlerSubscription",
        False,  # Requires ModelONEXContainer
    ),
    (
        "omnimemory.nodes.node_similarity_compute.handlers.handler_similarity_compute",
        "HandlerSimilarityCompute",
        False,  # Requires ModelONEXContainer -- verify-only
    ),
    # Graph adapters (OMN-6579, OMN-6580)
    (
        "omnimemory.handlers.adapters.adapter_intent_graph",
        "AdapterIntentGraph",
        False,  # Requires config + initialize()
    ),
    # Navigation history reducer (OMN-6583)
    (
        "omnimemory.nodes.node_navigation_history_reducer.handlers.handler_navigation_history_reducer",
        "HandlerNavigationHistoryReducer",
        False,  # Requires pg_dsn, qdrant, embedding params
    ),
    # Semantic compute (OMN-6585)
    (
        "omnimemory.nodes.node_semantic_analyzer_compute.handlers.handler_semantic_compute",
        "HandlerSemanticCompute",
        False,  # Requires ModelONEXContainer
    ),
    # Lifecycle handler (OMN-6588)
    (
        "omnimemory.runtime.handler_lifecycle",
        "HandlerMemoryLifecycle",
        False,  # Requires component adapters
    ),
]


async def wire_memory_handlers(
    config: ModelDomainPluginConfig,
) -> list[str]:
    """Wire memory domain handlers by verifying importability.

    Imports and validates handler modules for the memory domain.
    OmniMemory handlers require runtime dependencies (containers,
    storage adapters, event buses) so they are not instantiated here --
    only verified importable.

    Args:
        config: Plugin configuration with container and correlation_id.

    Returns:
        List of handler names successfully verified.

    Raises:
        ImportError: If any required handler module cannot be imported.
    """
    correlation_id = config.correlation_id
    services_registered: list[str] = []

    for module_path, attr_name, is_class in _HANDLER_SPECS:
        try:
            mod = await asyncio.to_thread(importlib.import_module, module_path)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Failed to import handler module '{module_path}' "
                f"(correlation_id={correlation_id})"
            ) from e
        try:
            handler_attr = getattr(mod, attr_name)
        except AttributeError as e:
            raise ImportError(f"{attr_name} not found in {module_path}") from e

        if not callable(handler_attr):
            raise ImportError(f"{attr_name} in {module_path} is not callable")

        if is_class:
            # Instantiate class-based handlers (pure compute, no deps)
            _instance = handler_attr()
            logger.debug(
                "Instantiated %s (correlation_id=%s)",
                attr_name,
                correlation_id,
            )
        else:
            logger.debug(
                "Verified %s importable (correlation_id=%s)",
                attr_name,
                correlation_id,
            )

        services_registered.append(attr_name)

    logger.info(
        "Memory handlers wired: %d services (correlation_id=%s)",
        len(services_registered),
        correlation_id,
        extra={"services": services_registered},
    )

    return services_registered


__all__: list[str] = [
    "wire_memory_handlers",
]
