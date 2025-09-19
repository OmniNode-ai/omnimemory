"""
Result collection model for ONEX Foundation Architecture.

Provides strongly typed result collection replacing List[Dict[str, Any]].
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .model_result_item import ModelResultItem
from .model_structured_data import ModelStructuredData


class ModelResultCollection(BaseModel):
    """Strongly typed result collection replacing List[Dict[str, Any]]."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    results: List[ModelResultItem] = Field(
        default_factory=list,
        description="List of operation results with structured data",
    )

    def add_result(
        self,
        id: str,
        status: str,
        message: str,
        data: Optional[ModelStructuredData] = None,
    ) -> None:
        """Add a new result to the collection."""
        result = ModelResultItem(id=id, status=status, message=message, data=data)
        self.results.append(result)

    def get_successful_results(self) -> List[ModelResultItem]:
        """Get all successful results."""
        return [result for result in self.results if result.status == "success"]

    def get_failed_results(self) -> List[ModelResultItem]:
        """Get all failed results."""
        return [result for result in self.results if result.status == "failure"]
