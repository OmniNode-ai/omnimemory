"""
Success rate metric model following ONEX standards.
"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ModelSuccessRate(BaseModel):
    """Success rate metric following ONEX standards."""

    rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Success rate as a decimal between 0.0 and 1.0",
    )
    total_operations: int = Field(
        ge=0,
        description="Total number of operations measured",
    )
    successful_operations: int = Field(
        ge=0,
        description="Number of successful operations",
    )
    calculation_window_start: datetime = Field(
        description="Start time of the calculation window",
    )
    calculation_window_end: datetime = Field(
        description="End time of the calculation window",
    )
    measurement_type: str = Field(
        description="Type of operation measured (e.g., 'memory_storage', 'retrieval')",
    )

    @field_validator("successful_operations")
    @classmethod
    def validate_successful_operations(cls, v: int, info) -> int:
        """Validate successful operations doesn't exceed total."""
        if hasattr(info, "data") and "total_operations" in info.data:
            total = info.data["total_operations"]
            if v > total:
                raise ValueError("Successful operations cannot exceed total operations")
        return v

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.rate

    @property
    def failed_operations(self) -> int:
        """Calculate number of failed operations."""
        return self.total_operations - self.successful_operations

    def to_percentage(self) -> float:
        """Convert rate to percentage."""
        return self.rate * 100.0
