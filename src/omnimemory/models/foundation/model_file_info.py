"""
Strongly typed file information model for OmniMemory ONEX architecture.

This module provides models for file information with type safety and validation.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ModelFileInfo(BaseModel):
    """Strongly typed file information wrapper."""

    path: Path = Field(description="File path as Path object")
    size_bytes: Optional[int] = Field(
        default=None, ge=0, description="File size in bytes"
    )
    checksum: Optional[str] = Field(
        default=None, description="File checksum for integrity"
    )
    mime_type: Optional[str] = Field(default=None, description="MIME type of the file")
    encoding: Optional[str] = Field(
        default=None, description="Text encoding if applicable"
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Validate and resolve file path."""
        if isinstance(v, str):
            v = Path(v)
        return v.resolve()

    @property
    def exists(self) -> bool:
        """Check if file exists."""
        return self.path.exists()

    @property
    def name(self) -> str:
        """Get file name."""
        return self.path.name

    @property
    def suffix(self) -> str:
        """Get file extension."""
        return self.path.suffix

    @property
    def size_mb(self) -> Optional[float]:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024) if self.size_bytes else None
