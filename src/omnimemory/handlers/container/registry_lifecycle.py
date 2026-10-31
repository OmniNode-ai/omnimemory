# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Binding site for the runtime lifecycle handler (OMN-21623)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnimemory.runtime.handler_lifecycle import HandlerMemoryLifecycle

__all__ = ["bind_memory_lifecycle"]


def bind_memory_lifecycle(
    *,
    graph_memory_adapter: object | None,
    intent_graph_adapter: object | None,
    navigation_handler: object | None,
    semantic_handler: object | None,
) -> HandlerMemoryLifecycle:
    """Build the runtime startup/shutdown lifecycle handler."""
    from omnimemory.runtime.handler_lifecycle import HandlerMemoryLifecycle

    return HandlerMemoryLifecycle(
        graph_memory_adapter=graph_memory_adapter,
        intent_graph_adapter=intent_graph_adapter,
        navigation_handler=navigation_handler,
        semantic_handler=semantic_handler,
    )
