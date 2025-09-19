"""
Memory response data model following ONEX standards.
"""

from uuid import UUID

from pydantic import BaseModel, Field

from .model_memory_data_content import ModelMemoryDataContent
from .model_memory_data_value import ModelMemoryDataValue


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
