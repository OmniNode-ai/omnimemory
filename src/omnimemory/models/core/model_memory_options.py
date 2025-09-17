"""
Memory operation options model following ONEX standards.
"""

from pydantic import BaseModel, Field


class ModelMemoryOptions(BaseModel):
    """Boolean options for memory operations following ONEX standards."""

    # Validation options
    validate_input: bool = Field(
        default=True,
        description="Whether to validate input data before processing",
    )
    require_confirmation: bool = Field(
        default=False,
        description="Whether the operation requires explicit confirmation",
    )
    skip_duplicates: bool = Field(
        default=True,
        description="Whether to skip duplicate memory entries",
    )

    # Processing options
    async_processing: bool = Field(
        default=True,
        description="Whether to process the operation asynchronously",
    )
    enable_compression: bool = Field(
        default=False,
        description="Whether to enable data compression",
    )
    enable_encryption: bool = Field(
        default=True,
        description="Whether to enable data encryption",
    )

    # Intelligence options
    enable_semantic_indexing: bool = Field(
        default=True,
        description="Whether to enable semantic indexing for the memory",
    )
    auto_generate_embeddings: bool = Field(
        default=True,
        description="Whether to automatically generate embeddings",
    )
    enable_pattern_recognition: bool = Field(
        default=False,
        description="Whether to enable pattern recognition processing",
    )

    # Migration options
    preserve_timestamps: bool = Field(
        default=True,
        description="Whether to preserve original timestamps during migration",
    )
    rollback_on_failure: bool = Field(
        default=True,
        description="Whether to rollback changes if operation fails",
    )
    create_backup: bool = Field(
        default=False,
        description="Whether to create backup before destructive operations",
    )
