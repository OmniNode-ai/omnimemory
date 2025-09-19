"""
Memory data value model following ONEX standards.
"""

from typing import Dict, List, Union

from pydantic import BaseModel, Field

from ...enums import EnumDataType


class ModelMemoryDataValue(BaseModel):
    """Individual memory data value following ONEX standards."""

    value: Union[str, int, float, bool, Dict, List] = Field(
        description="The actual data value",
    )
    data_type: EnumDataType = Field(
        description="Type of the data value",
    )
    encoding: str | None = Field(
        default=None,
        description="Encoding format if applicable (e.g., 'utf-8', 'base64')",
    )
    size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Size of the data in bytes",
    )
    checksum: str | None = Field(
        default=None,
        description="Checksum for data integrity verification",
    )
    is_encrypted: bool = Field(
        default=False,
        description="Whether the data value is encrypted",
    )
    encryption_method: str | None = Field(
        default=None,
        description="Encryption method used if encrypted",
    )
    compression: str | None = Field(
        default=None,
        description="Compression method used if compressed",
    )
    mime_type: str | None = Field(
        default=None,
        description="MIME type for binary or media data",
    )
    validation_schema: str | None = Field(
        default=None,
        description="JSON schema or validation pattern for the value",
    )

    def get_size_mb(self) -> float | None:
        """Get size in megabytes."""
        return self.size_bytes / (1024 * 1024) if self.size_bytes else None

    def is_large_data(self, threshold_mb: float = 1.0) -> bool:
        """Check if data exceeds size threshold."""
        size_mb = self.get_size_mb()
        return size_mb is not None and size_mb > threshold_mb
