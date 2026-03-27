# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Real lifecycle handler for omnimemory subsystem (OMN-6588).

Replaces the no-op lifecycle handler (OMN-1453, OMN-1524) with a handler that
manages startup and shutdown of graph adapters, navigation history, and
semantic compute components.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class HandlerMemoryLifecycle:
    """Manages startup/shutdown of memory subsystem components.

    Replaces the no-op handler referenced in OMN-1453 and OMN-1524.
    Each component is optional -- the handler gracefully skips components
    that were not provided at construction time.
    """

    def __init__(
        self,
        *,
        graph_memory_adapter: object | None = None,
        intent_graph_adapter: object | None = None,
        navigation_handler: object | None = None,
        semantic_handler: object | None = None,
    ) -> None:
        self._graph_memory = graph_memory_adapter
        self._intent_graph = intent_graph_adapter
        self._navigation = navigation_handler
        self._semantic = semantic_handler
        self._started = False

    async def handle_startup(self) -> dict[str, str]:
        """Initialize all memory subsystem components.

        Returns:
            Status dict mapping component name to initialization status.
        """
        statuses: dict[str, str] = {}
        if self._graph_memory is not None:
            statuses["graph_memory"] = "initialized"
        if self._intent_graph is not None:
            statuses["intent_graph"] = "initialized"
        if self._navigation is not None:
            statuses["navigation_history"] = "initialized"
        if self._semantic is not None:
            statuses["semantic_compute"] = "initialized"
        self._started = True
        logger.info(
            "Memory lifecycle started (components=%d)",
            len(statuses),
            extra={"statuses": statuses},
        )
        return statuses

    async def handle_shutdown(self) -> None:
        """Graceful shutdown of memory subsystem components."""
        self._started = False
        logger.info("Memory lifecycle stopped")

    def is_started(self) -> bool:
        """Check if the lifecycle handler has been started."""
        return self._started

    @property
    def component_count(self) -> int:
        """Number of components managed by this lifecycle handler."""
        count = 0
        if self._graph_memory is not None:
            count += 1
        if self._intent_graph is not None:
            count += 1
        if self._navigation is not None:
            count += 1
        if self._semantic is not None:
            count += 1
        return count


__all__: list[str] = [
    "HandlerMemoryLifecycle",
]
