# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Configuration model for the FileSystem adapter.

This module contains the Pydantic configuration model for
HandlerFileSystemAdapter.

.. versionadded:: 0.1.0
    Initial implementation for OMN-1384.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelFileSystemAdapterConfig",
]


class ModelFileSystemAdapterConfig(BaseModel):
    """Configuration for FileSystem adapter.

    Attributes:
        base_path: Base path for memory storage. All snapshots will be stored
            under this directory.
        snapshots_dir: Subdirectory name for storing snapshot JSON files.
            Defaults to "snapshots".
        max_file_size: Maximum file size in bytes for read/write operations.
            Defaults to 10MB.
    """

    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

    base_path: Path = Field(
        ...,
        description="Base path for memory storage",
    )
    snapshots_dir: str = Field(
        default="snapshots",
        description="Subdirectory for snapshots",
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024,
        gt=0,
        description="Maximum file size in bytes (default 10MB)",
    )
