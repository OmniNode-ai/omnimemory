# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Memory Storage Effect Handlers.

The primary handler is the FileSystem adapter which wraps `HandlerFileSystem`
from omnibase_infra to store memory snapshots as JSON files.

Note:
    The FileSystem adapter requires omnibase_infra (a dev dependency).
    Install with: poetry install --with dev

Available Handlers:
    - HandlerFileSystemAdapter: Stores snapshots as JSON files on disk

Example::

    import asyncio
    from omnimemory.nodes.memory_storage_effect.adapters import (
        HandlerFileSystemAdapter,
        ModelFileSystemAdapterConfig,
    )
    from pathlib import Path

    async def example():
        config = ModelFileSystemAdapterConfig(base_path=Path("/data/memory"))
        adapter = HandlerFileSystemAdapter(config)
        await adapter.initialize()

    asyncio.run(example())

.. versionadded:: 0.1.0
    Initial implementation for OMN-1384.
"""

from omnimemory.models.adapters import ModelFileSystemAdapterConfig

from .adapter_filesystem import (
    HandlerFileSystemAdapter,
)

__all__ = [
    "HandlerFileSystemAdapter",
    "ModelFileSystemAdapterConfig",
]
