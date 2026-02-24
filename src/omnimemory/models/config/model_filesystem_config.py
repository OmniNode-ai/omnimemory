# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
Filesystem storage configuration model following ONEX standards.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelFilesystemConfig(BaseModel):
    """Configuration for filesystem-based memory storage.

    This config defines how OmniMemory stores and retrieves memory files
    from the local filesystem. It is the required storage backend for Phase 1.
    """

    model_config = ConfigDict(extra="forbid")

    # Path configuration
    base_path: Path = Field(
        description="Base directory for memory file storage",
    )

    # Size limits
    max_file_size_bytes: int = Field(
        default=10_485_760,
        ge=1,
        le=1_073_741_824,
        description="Maximum file size in bytes (default 10MB, max 1GB)",
    )

    # File type restrictions
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".json", ".txt", ".md"],
        description="Allowed file extensions for memory storage",
    )

    # Directory behavior
    create_if_missing: bool = Field(
        default=True,
        description="Create base_path directory if it does not exist",
    )

    # Performance settings
    enable_compression: bool = Field(
        default=False,
        description="Enable gzip compression for stored files",
    )
    buffer_size_bytes: int = Field(
        default=65536,
        ge=4096,
        le=1_048_576,
        description="I/O buffer size in bytes (default 64KB)",
    )

    @field_validator("allowed_extensions")
    @classmethod
    def validate_extensions(cls, v: list[str]) -> list[str]:
        """Validate that extensions start with a dot."""
        for ext in v:
            if not ext.startswith("."):
                raise ValueError(f"Extension must start with '.': {ext}")
        return v

    @field_validator("base_path")
    @classmethod
    def validate_base_path(cls, v: Path) -> Path:
        """Validate that base_path is an absolute path."""
        if not v.is_absolute():
            raise ValueError("base_path must be an absolute path")
        return v
