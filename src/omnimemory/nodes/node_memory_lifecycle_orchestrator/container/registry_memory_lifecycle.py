# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Binding site for the memory lifecycle handlers (OMN-21623).

The one place that builds the expire handler. The plugin and the expire adapter
take it from this binder instead of constructing the class by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

    from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers.handler_memory_expire import (
        HandlerMemoryExpire,
    )

__all__ = ["bind_memory_expire"]


def bind_memory_expire(container: ModelONEXContainer) -> HandlerMemoryExpire:
    """Build the expire handler (not yet initialized)."""
    from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers.handler_memory_expire import (
        HandlerMemoryExpire,
    )

    return HandlerMemoryExpire(container)
