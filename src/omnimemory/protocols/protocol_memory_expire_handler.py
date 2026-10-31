# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Protocol for the handler that performs the ACTIVE -> EXPIRED transition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnimemory.nodes.node_memory_lifecycle_orchestrator.handlers.handler_memory_expire import (
        ModelExpireMemoryCommand,
        ModelMemoryExpireResult,
    )


@runtime_checkable
class ProtocolMemoryExpireHandler(Protocol):
    """Protocol for the handler that performs the ACTIVE -> EXPIRED transition.

    Matches ``HandlerMemoryExpire.handle()``.
    """

    async def handle(
        self, command: ModelExpireMemoryCommand
    ) -> ModelMemoryExpireResult: ...


__all__ = ["ProtocolMemoryExpireHandler"]
