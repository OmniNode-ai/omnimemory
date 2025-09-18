"""
Memory data models following ONEX standards.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ...enums.enum_data_type import EnumDataType


class ModelMemoryDataValue(BaseModel):
    """Individual memory data value following ONEX standards."""

    value: Any = Field(
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


class ModelMemoryDataContent(BaseModel):
    """Memory data content following ONEX standards."""

    content_id: UUID = Field(
        description="Unique identifier for this data content",
    )
    primary_data: ModelMemoryDataValue = Field(
        description="Primary data value",
    )
    metadata: dict[str, ModelMemoryDataValue] = Field(
        default_factory=dict,
        description="Additional metadata as typed data values",
    )
    relationships: dict[str, UUID] = Field(
        default_factory=dict,
        description="Relationships to other data content by UUID",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorizing the data content",
    )
    source_system: str | None = Field(
        default=None,
        description="Source system that generated this data",
    )
    source_reference: str | None = Field(
        default=None,
        description="Reference or identifier in the source system",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the data content was created",
    )
    modified_at: datetime | None = Field(
        default=None,
        description="When the data content was last modified",
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this data has been accessed",
    )
    last_accessed_at: datetime | None = Field(
        default=None,
        description="When the data was last accessed",
    )

    def add_metadata(self, key: str, value: ModelMemoryDataValue) -> None:
        """Add metadata to the data content."""
        self.metadata[key] = value
        self.modified_at = datetime.utcnow()

    def get_metadata(self, key: str) -> ModelMemoryDataValue | None:
        """Get metadata by key."""
        return self.metadata.get(key)

    def add_relationship(self, relationship_type: str, target_id: UUID) -> None:
        """Add a relationship to another data content."""
        self.relationships[relationship_type] = target_id
        self.modified_at = datetime.utcnow()

    def record_access(self) -> None:
        """Record an access to this data content."""
        self.access_count += 1
        self.last_accessed_at = datetime.utcnow()

    @property
    def total_size_bytes(self) -> int:
        """Calculate total size including metadata."""
        total = self.primary_data.size_bytes or 0
        for metadata_value in self.metadata.values():
            total += metadata_value.size_bytes or 0
        return total

    @property
    def is_recently_accessed(self, hours: int = 24) -> bool:
        """Check if data was accessed recently."""
        if not self.last_accessed_at:
            return False
        delta = datetime.utcnow() - self.last_accessed_at
        return delta.total_seconds() / 3600 < hours


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


class ModelMemoryResponseData(BaseModel):
    """Memory response data following ONEX standards."""

    response_data_id: UUID = Field(
        description="Unique identifier for this response data",
    )
    result_data: list[ModelMemoryDataContent] = Field(
        default_factory=list,
        description="Main result data content",
    )
    aggregation_data: dict[str, ModelMemoryDataValue] = Field(
        default_factory=dict,
        description="Aggregated data results as typed data values",
    )
    metadata: dict[str, ModelMemoryDataValue] = Field(
        default_factory=dict,
        description="Response metadata as typed data values",
    )
    pagination_info: dict[str, int] = Field(
        default_factory=dict,
        description="Pagination information for the response",
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics for the operation",
    )
    quality_indicators: dict[str, float] = Field(
        default_factory=dict,
        description="Quality indicators for the response data",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about the response data",
    )

    def add_result(self, content: ModelMemoryDataContent) -> None:
        """Add result data content."""
        self.result_data.append(content)

    def add_aggregation(self, key: str, value: ModelMemoryDataValue) -> None:
        """Add aggregation data."""
        self.aggregation_data[key] = value

    def add_metadata(self, key: str, value: ModelMemoryDataValue) -> None:
        """Add response metadata."""
        self.metadata[key] = value

    def set_pagination_info(
        self, total: int, offset: int = 0, limit: int = 100
    ) -> None:
        """Set pagination information."""
        self.pagination_info = {
            "total": total,
            "offset": offset,
            "limit": limit,
            "returned": len(self.result_data),
        }

    def add_performance_metric(self, metric: str, value: float) -> None:
        """Add performance metric."""
        self.performance_metrics[metric] = value

    def add_quality_indicator(self, indicator: str, value: float) -> None:
        """Add quality indicator."""
        self.quality_indicators[indicator] = value

    def add_warning(self, warning: str) -> None:
        """Add warning message."""
        self.warnings.append(warning)

    @property
    def total_results(self) -> int:
        """Get total number of result items."""
        return len(self.result_data)

    @property
    def total_response_size_bytes(self) -> int:
        """Calculate total size of response data."""
        total = sum(content.total_size_bytes for content in self.result_data)
        for metadata_value in self.metadata.values():
            total += metadata_value.size_bytes or 0
        for agg_value in self.aggregation_data.values():
            total += agg_value.size_bytes or 0
        return total

    @property
    def has_warnings(self) -> bool:
        """Check if response has any warnings."""
        return len(self.warnings) > 0

    @property
    def is_empty(self) -> bool:
        """Check if response has no results."""
        return len(self.result_data) == 0
