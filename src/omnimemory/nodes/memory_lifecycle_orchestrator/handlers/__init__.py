# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Memory lifecycle orchestrator handlers.

This module exports handlers for memory lifecycle operations including
archival to cold storage.

Example::

    from omnimemory.nodes.memory_lifecycle_orchestrator.handlers import (
        HandlerFileSystemArchive,
        ModelArchiveRecord,
        ModelArchiveRequest,
        ModelArchiveResult,
    )

    handler = HandlerFileSystemArchive()
    result = await handler.archive_memories(request)
"""

from omnimemory.nodes.memory_lifecycle_orchestrator.handlers.handler_filesystem_archive import (
    HandlerFileSystemArchive,
    ModelArchiveRecord,
    ModelArchiveRequest,
    ModelArchiveResult,
)

__all__ = [
    "HandlerFileSystemArchive",
    "ModelArchiveRecord",
    "ModelArchiveRequest",
    "ModelArchiveResult",
]
