"""
Memory request data model following ONEX standards.
"""

from uuid import UUID

from pydantic import BaseModel, Field

from .model_memory_data_content import ModelMemoryDataContent
from .model_memory_data_value import ModelMemoryDataValue


class ModelMemoryRequestData(BaseModel):
    """Memory request data following ONEX standards."""

    request_data_id: UUID = Field(
        description="Unique identifier for this request data",
    )
    operation_data: ModelMemoryDataContent = Field(
        description="Main operation data content",
    )
    supplementary_data: dict[str, ModelMemoryDataContent] = Field(
        default_factory=dict,
        description="Additional data content for the operation",
    )
    query_parameters: dict[str, ModelMemoryDataValue] = Field(
        default_factory=dict,
        description="Query parameters as typed data values",
    )
    filters: dict[str, ModelMemoryDataValue] = Field(
        default_factory=dict,
        description="Filter criteria as typed data values",
    )
    sorting_criteria: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Sorting criteria as (field, direction) tuples",
    )
    pagination: dict[str, int] = Field(
        default_factory=dict,
        description="Pagination parameters (offset, limit, etc.)",
    )
    validation_rules: list[str] = Field(
        default_factory=list,
        description="Custom validation rules for this request data",
    )

    def add_supplementary_data(self, key: str, content: ModelMemoryDataContent) -> None:
        """Add supplementary data content."""
        self.supplementary_data[key] = content

    def add_query_parameter(self, key: str, value: ModelMemoryDataValue) -> None:
        """Add a query parameter."""
        self.query_parameters[key] = value

    def add_filter(self, key: str, value: ModelMemoryDataValue) -> None:
        """Add a filter criterion."""
        self.filters[key] = value

    def set_pagination(self, offset: int = 0, limit: int = 100) -> None:
        """Set pagination parameters."""
        self.pagination = {"offset": offset, "limit": limit}

    def add_sort_criteria(self, field: str, direction: str = "asc") -> None:
        """Add sorting criteria."""
        if direction not in ["asc", "desc"]:
            raise ValueError("Sort direction must be 'asc' or 'desc'")
        self.sorting_criteria.append((field, direction))

    @property
    def total_data_size_bytes(self) -> int:
        """Calculate total size of all data content."""
        total = self.operation_data.total_size_bytes
        for content in self.supplementary_data.values():
            total += content.total_size_bytes
        return total

    @property
    def has_filters(self) -> bool:
        """Check if request has any filters."""
        return len(self.filters) > 0

    @property
    def has_sorting(self) -> bool:
        """Check if request has sorting criteria."""
        return len(self.sorting_criteria) > 0

    @property
    def has_pagination(self) -> bool:
        """Check if request has pagination."""
        return len(self.pagination) > 0
