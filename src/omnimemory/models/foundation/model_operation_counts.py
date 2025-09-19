"""
Operation counts model following ONEX standards.
"""

from pydantic import BaseModel, Field


class ModelOperationCounts(BaseModel):
    """Count of operations by type."""

    storage_operations: int = Field(
        default=0, description="Number of storage operations"
    )
    retrieval_operations: int = Field(
        default=0, description="Number of retrieval operations"
    )
    query_operations: int = Field(default=0, description="Number of query operations")
    consolidation_operations: int = Field(
        default=0, description="Number of consolidation operations"
    )
    failed_operations: int = Field(default=0, description="Number of failed operations")
